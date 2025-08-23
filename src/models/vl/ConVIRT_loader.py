import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoModel, AutoTokenizer

class ConVIRTWrapper(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=128, freeze_backbone=True, device="cpu"):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.device = device

        # ConVIRT usually projects into a shared embedding space
        self.image_proj = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)

        # Normalize embeddings before similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))  # temp=0.07

        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.to(device)

    def encode_image(self, x, normalize=True):
        z = self.image_encoder(x)  # [B, D]
        z = self.image_proj(z)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z

    def encode_text(self, x, normalize=True):
        if isinstance(x, (list, tuple, str)):  
            # tokenize if raw strings
            toks = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            toks = {k: v.to(self.device) for k,v in toks.items()}
            out = self.text_encoder(**toks)
            z = out.pooler_output
        else:
            # already tokenized dict
            out = self.text_encoder(**x)
            z = out.pooler_output

        z = self.text_proj(z)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, images, texts):
        img_z = self.encode_image(images)
        txt_z = self.encode_text(texts)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_z @ txt_z.T
        return logits, img_z, txt_z

def build_convirt(embed_dim=128, freeze_backbone=True, device="cpu"):
    resnet = resnet50(pretrained=True)
    resnet.fc = nn.Identity()  # remove classification layer
    resnet.output_dim = 2048

    # Text backbone
    text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_model.output_dim = text_model.config.hidden_size  # 768
    tokenizer  = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Wrap into ConVIRT
    convrt = ConVIRTWrapper(resnet, text_model, embed_dim=embed_dim, device=device)
    convrt.tokenizer = tokenizer

    return convrt