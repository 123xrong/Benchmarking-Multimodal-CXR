# vl_backbones.py
from dataclasses import dataclass
from typing import List, Tuple
import torch, torch.nn.functional as F

@dataclass
class VLPack:
    model: object
    preprocess: object          # fn(images PIL|Tensor -> pixel_values Tensor[B,3,H,W])
    tokenize: object            # fn(list[str]) -> dict with input_ids/attention_mask tensors
    dim: int
    name: str

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _l2norm(x): return F.normalize(x, dim=-1)
