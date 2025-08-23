# pubmedclip_loader.py
import torch
from typing import List
from transformers import CLIPModel, CLIPProcessor
from vl_backbones import VLPack

def build_pubmedclip(device="cuda") -> VLPack:
    name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc  = CLIPProcessor.from_pretrained(name)

    class PubMedCLIP(VLPack):
        @torch.no_grad()
        def encode_image(self, pixel_values):
            feat = model.get_image_features(pixel_values=pixel_values.to(model.device))
            return self._l2norm(feat)

        @torch.no_grad()
        def encode_text(self, texts):
            toks = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
            toks = {k: v.to(model.device) for k,v in toks.items()}
            feat = model.get_text_features(**toks)
            return self._l2norm(feat)

    # processor gives both image+text transforms; expose simple callables:
    def preprocess(pil_or_list):
        batch = proc(images=pil_or_list, return_tensors="pt")
        return batch["pixel_values"]

    def tokenize(texts: List[str]):
        return proc(text=texts, return_tensors="pt", padding=True, truncation=True)

    return PubMedCLIP(model=model, preprocess=preprocess, tokenize=tokenize, dim=model.config.projection_dim, name="PubMedCLIP")
