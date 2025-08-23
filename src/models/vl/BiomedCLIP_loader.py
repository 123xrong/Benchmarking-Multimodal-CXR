import os, math, time, json, random, gc
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_recall_curve

class BiomedCLIPMultiLabel(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 14,
        freeze_backbone: bool = False,
        l2_norm_features: bool = False,
        head_dropout: float = 0.0,
        embed_dim: int = None,            # <--- NEW: explicit override if known
    ):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes
        self.l2_norm_features = l2_norm_features
        self._frozen = freeze_backbone

        # Try static inference first
        d = embed_dim
        if d is None and getattr(getattr(self.clip, "visual", None), "output_dim", None):
            d = int(self.clip.visual.output_dim)
        if d is None and hasattr(self.clip, "embed_dim"):
            d = int(self.clip.embed_dim)
        if d is None and hasattr(self.clip, "proj") and hasattr(self.clip.proj, "shape"):
            d = int(self.clip.proj.shape[1])
        if d is None and hasattr(self.clip, "image_projection") and hasattr(self.clip.image_projection, "in_features"):
            d = int(self.clip.image_projection.in_features)

        # Build head now if we know d; else defer until we run a dummy forward
        self.head = None
        if d is not None:
            self._build_head(d, dropout=head_dropout)

        # Optionally freeze backbone (head will remain trainable)
        if freeze_backbone:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad = False

    def _build_head(self, d: int, dropout: float = 0.0):
        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(d, self.num_classes))
        self.head = nn.Sequential(*layers)
        # init
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # ensure head is trainable even if backbone is frozen
        for p in self.head.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def materialize_head_by_dummy(self, device="cpu", img_size=224, dtype=torch.float32):
        """Run a one-time dummy forward to discover embedding dim and build the head."""
        if self.head is not None:
            return  # already built
        self.eval()
        dummy = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
        if hasattr(self.clip, "encode_image"):
            z = self.clip.encode_image(dummy)
        elif hasattr(self.clip, "visual") and callable(getattr(self.clip, "visual")):
            z = self.clip.visual(dummy)
        else:
            z = self.clip(dummy)
        if z.ndim == 3:
            z = z.mean(dim=1)
        d = int(z.shape[-1])
        self._build_head(d)
        # restore mode if backbone was intended to train
        if not self._frozen:
            self.clip.train()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.clip, "encode_image"):
            z = self.clip.encode_image(x)
        elif hasattr(self.clip, "visual") and callable(getattr(self.clip, "visual")):
            z = self.clip.visual(x)
        else:
            z = self.clip(x)
        return z

    def forward(self, x, return_features: bool = False):
        z = self._encode(x)
        if z.ndim == 3:
            z = z.mean(dim=1)
        if self.l2_norm_features:
            z = F.normalize(z, dim=-1)
        if self.head is None:
            # safety: build head on the fly if somehow still missing
            self._build_head(z.shape[-1])
        logits = self.head(z)
        return (logits, z) if return_features else logits
    
class BiomedCLIPPack:
    def __init__(self, model, tokenizer, preprocess, device="cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = torch.device(device)
        self.name = "BiomedCLIP"

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, pixel_values):
        feats = self.model.encode_image(pixel_values.to(self.device))
        return feats / feats.norm(dim=-1, keepdim=True)

# Fine-tuning projection head
def biomedclip_proj_ft():
    """Build a BiomedCLIP model with a trainable projection head."""
    