# src/models/vl/medclip_loader.py

import torch, contextlib
import torch.nn.functional as F
import torch.nn as nn
from typing import Sequence
from PIL import Image
from medclip import MedCLIPProcessor, MedCLIPVisionModelViT, MedCLIPVisionModel
import medclip.modeling_medclip as mcm  # import the class from its defining module
from vl_backbones import VLPack

def _l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

@contextlib.contextmanager
def medclip_relaxed_load():
    """
    Inside this context:
      - torch.load maps to CPU
      - MedCLIPModel.load_state_dict is made non-strict and drops '*position_ids*'
    """
    orig_load = torch.load
    def _load_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return orig_load(*args, **kwargs)
    torch.load = _load_cpu

    orig_load_sd = mcm.MedCLIPModel.load_state_dict
    def _loose(self, state_dict, strict=True):
        cleaned = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        return orig_load_sd(self, cleaned, strict=False)
    mcm.MedCLIPModel.load_state_dict = _loose
    try:
        yield
    finally:
        torch.load = orig_load
        mcm.MedCLIPModel.load_state_dict = orig_load_sd

def build_medclip(device: str = "cpu", backbone: str = "vit") -> "VLPack":
    """
    CPU-safe MedCLIP wrapper that:
      - loads weights on CPU with a relaxed state_dict
      - avoids model(...), which .cuda()'s internally in some versions
      - returns pooled hidden features (no projection layers required)
    Prefer backbone='vit' so text/vision dims both ~768.
    """
    proc = MedCLIPProcessor()
    vision_cls = MedCLIPVisionModelViT if backbone.lower() == "vit" else MedCLIPVisionModel

    # Construct and load weights (relaxed)
    model = mcm.MedCLIPModel(vision_cls=vision_cls)
    with medclip_relaxed_load():
        model.from_pretrained()

    model = model.to(device).eval()

    with torch.no_grad():
        t_inputs = proc(text=["probe"], return_tensors="pt", padding=True, truncation=True)
        tx = model.text_model(
            input_ids=t_inputs["input_ids"].to(device),
            attention_mask=(t_inputs.get("attention_mask", None).to(device)
                            if t_inputs.get("attention_mask") is not None else None)
        )
        if isinstance(tx, torch.Tensor):
            pooled_txt = tx  # already pooled
        elif hasattr(tx, "pooler_output") and tx.pooler_output is not None:
            pooled_txt = tx.pooler_output
        else:
            pooled_txt = tx.last_hidden_state[:, 0]
        dim = int(pooled_txt.shape[-1])

    # Preprocess (uses MedCLIPProcessor)
    def preprocess(pils_or_list):
        if torch.is_tensor(pils_or_list):
            return pils_or_list
        if isinstance(pils_or_list, Image.Image):
            return proc(images=pils_or_list, return_tensors="pt")["pixel_values"]
        return proc(images=pils_or_list, return_tensors="pt")["pixel_values"]

    class MedCLIPPack:
        def __init__(self):
            self.model = model
            self.preprocess = preprocess
            self.tokenize = lambda s: s
            self.dim = dim
            self.name = f"MedCLIP-{backbone.upper()}-noProj"
            self.device = torch.device(device)

            out_dim = dim
            self.model.visual_projection = nn.Linear(dim, out_dim, bias=False).to(device)
            self.model.text_projection   = nn.Linear(dim, out_dim, bias=False).to(device)


        @torch.no_grad()
        def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
            inp = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            tx = model.text_model(
                input_ids=inp["input_ids"].to(device),
                attention_mask=(inp.get("attention_mask", None).to(device)
                                if inp.get("attention_mask") is not None else None)
            )
            if isinstance(tx, torch.Tensor):
                pooled = tx
            elif hasattr(tx, "pooler_output") and tx.pooler_output is not None:
                pooled = tx.pooler_output
            else:
                pooled = tx.last_hidden_state[:, 0]
            return _l2(pooled)

        @torch.no_grad()
        def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
            x = pixel_values.to(device)
            try:
                vi = model.vision_model(pixel_values=x)
            except TypeError:
                vi = model.vision_model(x)
            pooled = (getattr(vi, "pooler_output", None)
                      if hasattr(vi, "pooler_output") else None)
            if pooled is None:
                last = getattr(vi, "last_hidden_state", None)
                if last is not None and last.ndim == 3:
                    pooled = last[:, 0]
                else:
                    arr = vi[0] if isinstance(vi, (list, tuple)) else vi
                    pooled = arr.mean(dim=(2, 3)) if arr.ndim == 4 else arr
            return _l2(pooled)

    return MedCLIPPack()
