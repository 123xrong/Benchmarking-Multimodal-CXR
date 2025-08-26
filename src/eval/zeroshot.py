from typing import List, Sequence, Tuple
import numpy as np
import torch

from src.models.vl.pubmedclip_loader import *

NIH14_CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural Thickening","Hernia"
]

T_POS = [
    "a frontal chest x-ray showing {c}.",
    "chest radiograph with {c}.",
    "{c} on chest radiograph."
]
T_NEG = [
    "a frontal chest x-ray without {c}.",
    "no evidence of {c} on chest radiograph."
]

def build_text_centroids(vlp, classes: Sequence[str] = NIH14_CLASSES) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_txt, neg_txt = [], []
    for c in classes:
        cname = c.replace("_", " ").lower()
        pos_txt += [t.format(c=cname) for t in T_POS]
        neg_txt += [t.format(c=cname) for t in T_NEG]
    pos_all = vlp.encode_text(pos_txt)  # [C*len(T_POS), D]
    neg_all = vlp.encode_text(neg_txt)  # [C*len(T_NEG), D]

    C = len(classes); Pp, Pn = len(T_POS), len(T_NEG)
    pos_centroids = torch.stack([pos_all[i*Pp:(i+1)*Pp].mean(0) for i in range(C)], 0)  # [C,D]
    neg_centroids = torch.stack([neg_all[i*Pn:(i+1)*Pn].mean(0) for i in range(C)], 0)  # [C,D]
    return pos_centroids, neg_centroids

def build_tokenized_text_centroids(vlp, tokenizer, classes, device="cpu"):
    pos_txt, neg_txt = [], []
    for c in classes:
        cname = c.replace("_", " ").lower()
        pos_txt += [t.format(c=cname) for t in T_POS]
        neg_txt += [t.format(c=cname) for t in T_NEG]

    # Tokenize and move to device
    pos_tok = tokenizer(pos_txt).to(device)
    neg_tok = tokenizer(neg_txt).to(device)

    # Encode
    pos_all = vlp.encode_text(pos_tok)  # [C*len(T_POS), D]
    neg_all = vlp.encode_text(neg_tok)  # [C*len(T_NEG), D]

    C = len(classes); Pp, Pn = len(T_POS), len(T_NEG)
    pos_centroids = torch.stack([pos_all[i*Pp:(i+1)*Pp].mean(0) for i in range(C)], 0)  # [C,D]
    neg_centroids = torch.stack([neg_all[i*Pn:(i+1)*Pn].mean(0) for i in range(C)], 0)  # [C,D]
    return pos_centroids, neg_centroids


@torch.no_grad()
def zeroshot_on_loader(vlp, loader, classes=NIH14_CLASSES, alpha=0.5, tokenizer=None, device="cpu"):
    # Choose centroid builder based on whether tokenizer is provided
    if tokenizer is not None:
        pos_c, neg_c = build_tokenized_text_centroids(vlp, tokenizer, classes, device=device)
    else:
        pos_c, neg_c = build_text_centroids(vlp, classes)

    all_probs, all_targets = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images, targets = batch[0], batch[1]
        else:
            images, targets = batch["images"], batch["labels"]

        pixel_values = images.to(device)
        img_z = vlp.encode_image(pixel_values)    # [B,D]

        s_pos = img_z @ pos_c.T
        s_neg = img_z @ neg_c.T
        logits = alpha * s_pos + (1 - alpha) * (-s_neg)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_true, y_prob

@torch.no_grad()
def zeroshot_on_biovil(vlp, loader, classes=NIH14_CLASSES, alpha=0.5, tokenizer=None, device="cpu"):
    # Choose centroid builder based on whether tokenizer is provided
    if tokenizer is not None:
        pos_c, neg_c = build_tokenized_text_centroids(vlp, tokenizer, classes, device=device)
    else:
        pos_c, neg_c = build_text_centroids(vlp, classes)

    all_probs, all_targets = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            image_path, targets = batch[2], batch[1]
        else:
            image_path, targets = batch["img_path"], batch["labels"]

        # pixel_values = images.to(device)
        img_z = vlp.encode_image(image_path)    # [B,D]

        s_pos = img_z @ pos_c.T
        s_neg = img_z @ neg_c.T
        logits = alpha * s_pos + (1 - alpha) * (-s_neg)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_true, y_prob