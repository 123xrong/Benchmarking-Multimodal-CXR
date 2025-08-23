import torch, torch.nn.functional as F
from pathlib import Path
from typing import Sequence
from PIL import Image
from health_multimodal.image.utils import get_image_inference, ImageModelType
from health_multimodal.text.utils import get_bert_inference, BertEncoderType
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from vl_backbones import VLPack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _l2(x): return F.normalize(x, dim=-1)

def build_biovil_t(device: str = "cuda") -> VLPack:
    img_engine = get_image_inference(ImageModelType.BIOVIL_T)  # returns ImageInferenceEngine
    txt_engine = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

    # infer joint embedding dim from the text side
    with torch.no_grad():
        dim = int(txt_engine.get_embeddings_from_prompt(["probe"], normalize=True).shape[-1])

    # transforms are only needed if you want to pre-process PILs yourself; the engine works from paths.
    transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=480)

    def preprocess(pils_or_list):
        # optional: keep if you also want to use PIL tensors elsewhere
        if isinstance(pils_or_list, (list, tuple)):
            return torch.stack([transform(p) for p in pils_or_list])
        elif isinstance(pils_or_list, Image.Image):
            return transform(pils_or_list).unsqueeze(0)
        elif torch.is_tensor(pils_or_list):
            return pils_or_list
        else:
            raise TypeError("Unsupported input to preprocess")

    class BioViLT(VLPack):
        device = torch.device(device)

        @torch.no_grad()
        def encode_image(self, paths: Sequence[str]) -> torch.Tensor:
            # EXPECTS a list of file paths
            embs = []
            for p in paths:
                z = img_engine.get_projected_global_embedding(Path(p))  # already L2-normalized
                embs.append(z.to(self.device))
            return torch.stack(embs, 0)  # [B, D]

        @torch.no_grad()
        def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
            z = txt_engine.get_embeddings_from_prompt(texts, normalize=True)  # [N, D] L2’d
            return z.to(self.device)

    pack = BioViLT(model={"img_engine": img_engine, "txt_engine": txt_engine},
                   preprocess=preprocess, tokenize=lambda s: s,
                   dim=dim, name="BioViL-T")
    return pack

