import torch, torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from typing import Sequence, Union
from PIL import Image
from health_multimodal.image.utils import get_image_inference, ImageModelType
from health_multimodal.text.utils import get_bert_inference, BertEncoderType
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from vl_backbones import VLPack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _l2(x): 
    return F.normalize(x, dim=-1)

def build_biovil_t(device) -> "BioViLT":
    img_engine = get_image_inference(ImageModelType.BIOVIL_T)  # ImageInferenceEngine
    txt_engine = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

    # infer joint embedding dim from the text side
    with torch.no_grad():
        dim = int(txt_engine.get_embeddings_from_prompt(["probe"], normalize=True).shape[-1])

    transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=480)

    def preprocess(pils_or_list):
        if isinstance(pils_or_list, (list, tuple)):
            return torch.stack([transform(p) for p in pils_or_list])
        elif isinstance(pils_or_list, Image.Image):
            return transform(pils_or_list).unsqueeze(0)
        elif torch.is_tensor(pils_or_list):
            return pils_or_list
        else:
            raise TypeError("Unsupported input to preprocess")

    class BioViLT(nn.Module):
        def __init__(self, img_engine, txt_engine, dim, device):
            super().__init__()
            self.img_engine = img_engine
            self.txt_engine = txt_engine
            self.dim = dim
            self.device = torch.device(device)

            # Example: add trainable projection heads if needed
            self.visual_projection = nn.Linear(dim, dim)   # replace with nn.Linear(dim, dim) if needed
            self.text_projection   = nn.Linear(dim, dim)   # same as above

            self.preprocess = preprocess
            self.tokenize = lambda s: s
            self.name = "BioViL-T"

        @torch.no_grad()
        def encode_image(self, inputs: Sequence[str]) -> torch.Tensor:
            """
            Accepts a list of file paths (strings).
            Returns: [B, D] image embeddings (L2-normalized).
            """
            embs = []
            for p in inputs:
                z = self.img_engine.get_projected_global_embedding(Path(p))  # path route only
                embs.append(z.to(self.device))
            z = torch.stack(embs, 0)
            return _l2(self.visual_projection(z))

        @torch.no_grad()
        def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
            z = self.txt_engine.get_embeddings_from_prompt(texts, normalize=True)  # [N, D]
            z = z.to(self.device)
            return _l2(self.text_projection(z))

    return BioViLT(img_engine, txt_engine, dim, device)


