from typing import Dict, List, Sequence
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score, roc_curve, auc

import numpy as np
import pandas as pd
import os, math
import torch
import matplotlib.pyplot as plt

NIH14_CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural Thickening","Hernia"
]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute AUROC and AUPRC (AP) metrics for multilabel classification.
    
    Args:
        y_true: (N, C) binary ground-truth labels
        y_prob: (N, C) predicted probabilities
    
    Returns:
        dict with macro/micro AUROC, macro/micro AUPRC, and per-class metrics
    """
    out = {}
    try:
        C = y_true.shape[1]
        aurocs, aps = [], []

        # ---- Per-class metrics ----
        for c in range(C):
            yt, yp = y_true[:, c], y_prob[:, c]
            if yt.max() > 0 and yt.min() == 0:  # class has both positives and negatives
                aurocs.append(roc_auc_score(yt, yp))
                aps.append(average_precision_score(yt, yp))
            else:
                aurocs.append(np.nan)
                aps.append(np.nan)

        # Macro averages
        out["auroc_macro"] = np.nanmean(aurocs)
        out["map_macro"]   = np.nanmean(aps)

        # Micro averages (treat all (sample,class) pairs as flat binary tasks)
        out["auroc_micro"] = roc_auc_score(y_true.ravel(), y_prob.ravel())
        out["map_micro"]   = average_precision_score(y_true.ravel(), y_prob.ravel())

        # Store per-class results too
        out["auroc_per_class"] = dict(zip(NIH14_CLASSES, aurocs))
        out["map_per_class"]   = dict(zip(NIH14_CLASSES, aps))

    except Exception as e:
        # ---- Fallback: manual AP computation ----
        def simple_ap(y, p):
            order = np.argsort(-p)
            y = y[order]
            cum_tp = np.cumsum(y)
            precision = cum_tp / (np.arange(len(y)) + 1)
            recall = cum_tp / max(1, y.sum())
            ap = 0.0
            for i in range(1, len(y)):
                ap += precision[i] * (recall[i] - recall[i-1])
            return ap

        C = y_true.shape[1]
        aps = []
        for c in range(C):
            yt, yp = y_true[:, c], y_prob[:, c]
            if yt.max() > 0 and yt.min() == 0:
                aps.append(simple_ap(yt, yp))
            else:
                aps.append(np.nan)
        out["map_macro"] = np.nanmean(aps)
        out["map_per_class"] = dict(zip(range(C), aps))

    return out

def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Return per-class thresholds maximizing F_beta on the given (val) set
    using the full precision-recall curve (more accurate than coarse grids).
    """
    C = y_true.shape[1]
    best_th = np.full(C, 0.5, dtype=np.float32)

    for c in range(C):
        yt = y_true[:, c]
        yp = y_prob[:, c]

        # Need both positives and negatives to choose a threshold meaningfully
        if yt.max() == 0 or yt.min() == 1:
            # Fallbacks:
            # - If no positives in val, keep conservative high threshold to avoid false positives
            # - If no negatives in val, use a small threshold to capture most positives
            best_th[c] = 0.90 if yt.max() == 0 else 0.10
            continue

        precision, recall, thresholds = precision_recall_curve(yt, yp)
        # thresholds has length = len(precision) - 1; align arrays
        # F_beta = (1+beta^2) * (P*R) / (beta^2*P + R)
        eps = 1e-8
        denom = (beta * beta) * precision + recall + eps
        f_beta = np.where(denom > 0, (1 + beta * beta) * precision * recall / denom, 0.0)

        # f_beta is same length as precision/recall; map to thresholds (skip the first point)
        if len(thresholds) == 0:
            best_th[c] = 0.5
            continue

        idx = np.argmax(f_beta[1:])  # skip the first point which has no threshold
        best_th[c] = float(thresholds[idx])

    return best_th

def topk_indices(y_prob, y_true, cls_idx, k=12, kind="TP", thr=None):
    thr = 0.5 if thr is None else float(thr)
    p = y_prob[:, cls_idx]; t = y_true[:, cls_idx].astype(bool)
    if kind == "TP":
        idx = np.where((p >= thr) & t)[0]
        idx = idx[np.argsort(-p[idx])]
    elif kind == "FP":
        idx = np.where((p >= thr) & (~t))[0]
        idx = idx[np.argsort(-p[idx])]
    elif kind == "FN":
        idx = np.where((p < thr) & t)[0]
        idx = idx[np.argsort(p[idx])]   # lowest probs first
    else:  # "TopP"
        idx = np.argsort(-p)
    return idx[:k], p

def topk_auroc_classes(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       classes: Sequence[str],
                       k: int = 5) -> List[str]:
    """Return the names of the top-K classes by AUROC (skips classes without both pos/neg)."""
    scores = []
    for i, c in enumerate(classes):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if yt.max() > 0 and yt.min() == 0:  # need both positives & negatives
            try:
                au = roc_auc_score(yt, yp)
            except Exception:
                au = np.nan
        else:
            au = np.nan
        scores.append((c, au))
    # keep only valid, sort desc
    valid = [(c, au) for c, au in scores if not np.isnan(au)]
    valid.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in valid[:k]]

def plot_roc_pr(y_true, y_prob, classes, sel=("Cardiomegaly","Edema","Effusion","Pneumothorax")):
    idx = [classes.index(c) for c in sel]
    # ROC
    plt.figure()
    for j in idx:
        fpr, tpr, _ = roc_curve(y_true[:,j], y_prob[:,j])
        plt.plot(fpr, tpr, label=f"{classes[j]} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Zero-shot ROC (VAL)"); plt.legend(); plt.tight_layout()
    plt.savefig("zs_val_roc.png", dpi=180); plt.close()

    # PR
    plt.figure()
    for j in idx:
        P, R, _ = precision_recall_curve(y_true[:,j], y_prob[:,j])
        plt.plot(R, P, label=classes[j])
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Zero-shot PR (VAL)"); plt.legend(); plt.tight_layout()
    plt.savefig("zs_val_pr.png", dpi=180); plt.close()

def save_gallery(loader, indices, out_png, ncol=6, img_size=224, paths_available=True):
    # iterate once to collect PILs in the original order
    imgs, paths = [], []
    i0 = 0
    for batch in loader:
        B = batch[0].size(0) if isinstance(batch, (list,tuple)) else batch["images"].size(0)
        if isinstance(batch, (list,tuple)):
            imgs_t, _, pth = batch[0], batch[1], (batch[2] if len(batch)>=3 else None)
        else:
            imgs_t, _, pth = batch["images"], batch["labels"], batch.get("paths")
        # de-normalize quick for viewing (ImageNet stats assumed for PubMedCLIP processor)
        x = imgs_t.clone().cpu()
        mean = torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1)
        std  = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
        x = (x*std + mean).clamp(0,1)
        for b in range(B):
            if i0 in indices:
                arr = (x[b].permute(1,2,0).numpy()*255).astype("uint8")
                imgs.append(Image.fromarray(arr))
                paths.append(pth[b] if (pth is not None and paths_available) else f"idx={i0}")
            i0 += 1
    if not imgs:
        print("No images selected for gallery."); return
    n = len(imgs); ncol = min(ncol, n); nrow = math.ceil(n/ncol)
    W = ncol*img_size; H = nrow*img_size
    canvas = Image.new("RGB", (W,H), (255,255,255))
    for i, im in enumerate(imgs):
        r, c = i//ncol, i%ncol
        canvas.paste(im.resize((img_size,img_size)), (c*img_size, r*img_size))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png)
    print("Saved:", out_png)

## Make gallery with text overlay
def per_class_table(y_true, y_prob, classes, thr):
    rows = []
    C = len(classes)
    for i in range(C):
        yt, yp = y_true[:, i], y_prob[:, i]
        yhat = (yp >= thr[i]).astype(int)
        valid = (yt.max() > 0 and yt.min() == 0)
        ap = average_precision_score(yt, yp) if valid else np.nan
        au = roc_auc_score(yt, yp) if valid else np.nan
        pr = precision_score(yt, yhat, zero_division=0)
        rc = recall_score(yt, yhat, zero_division=0)
        rows.append({
            "class": classes[i],
            "prevalence": float(yt.mean()),
            "thresh": float(thr[i]),
            "AP": ap, "AUROC": au,
            "Precision@th": pr, "Recall@th": rc
        })
    df = pd.DataFrame(rows)
    return df
    
def pick_top_classes(df, by="AP", k=4):
    df2 = df.copy()
    # prefer non-NaN; fall back to AUROC if AP is NaN
    if by == "AP":
        df2["_sort"] = df2["AP"].fillna(-1.0)
    elif by == "AUROC":
        df2["_sort"] = df2["AUROC"].fillna(-1.0)
    else:
        raise ValueError("by must be 'AP' or 'AUROC'")
    top = df2.sort_values("_sort", ascending=False).head(k)["class"].tolist()
    return top

def select_indices(y_true, y_prob, cls_idx, thr, k=18, kind="TP"):
    p = y_prob[:, cls_idx]
    t = y_true[:, cls_idx].astype(bool)
    if kind == "TP":
        idx = np.where((p >= thr) & t)[0]
        order = np.argsort(-p[idx])
    elif kind == "FP":
        idx = np.where((p >= thr) & (~t))[0]
        order = np.argsort(-p[idx])
    elif kind == "FN":
        idx = np.where((p < thr) & t)[0]
        order = np.argsort(p[idx])  # lowest probs first
    else:
        raise ValueError("kind must be TP/FP/FN")
    idx = idx[order]
    return idx[:k], p

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

def denorm_clip(batch_t):
    x = batch_t.detach().cpu()
    x = (x * _CLIP_STD + _CLIP_MEAN).clamp(0,1)
    imgs = (x.permute(0,2,3,1).numpy() * 255).astype("uint8")
    return [Image.fromarray(arr) for arr in imgs]

def save_gallery_with_text(loader, select_idx, probs_for_cls, y_true_cls, out_png,
                           ncol=6, img_size=224, title=None):
    """
    Assumes the loader iterates in a fixed, non-shuffled order identical to how y_true/y_prob were computed.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    select_idx = np.asarray(select_idx, dtype=int)
    if select_idx.size == 0:
        print(f"[warn] No samples for {out_png}")
        return

    sel_set = set(select_idx.tolist())
    max_sel = int(select_idx.max())

    images_pil, probs_sel, gts_sel, idxs = [], [], [], []

    # iterate once, picking only the requested global indices
    cursor = 0  # global sample counter
    for batch in loader:
        # supports (images, labels, [paths]) or dict
        if isinstance(batch, (list, tuple)):
            imgs_t, _ = batch[0], batch[1]
        else:
            imgs_t, _ = batch["images"], batch["labels"]

        B = imgs_t.size(0)

        # quickly skip whole batches if they are entirely before/after the selection range
        if cursor > max_sel and len(images_pil) >= len(select_idx):
            break

        # which positions in this batch are selected?
        keep = [b for b in range(B) if (cursor + b) in sel_set]
        if keep:
            # denorm only kept subset for speed
            ims = denorm_clip(imgs_t[keep])  # -> list of PILs in 'keep' order
            for k, b in enumerate(keep):
                g = cursor + b  # global index
                images_pil.append(ims[k])
                probs_sel.append(float(probs_for_cls[g]))
                gts_sel.append(int(y_true_cls[g]))
                idxs.append(g)
                if len(images_pil) == len(select_idx):
                    break

        cursor += B
        if len(images_pil) == len(select_idx):
            break

    if not images_pil:
        print(f"[warn] None of the requested indices were found in loader for {out_png}")
        return

    # ----- make canvas -----
    n = len(images_pil)
    ncol = min(ncol, n)
    nrow = math.ceil(n / ncol)
    W = ncol * img_size
    H = nrow * img_size + 40  # title bar
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # title (ASCII-safe)
    if title:
        title = title.replace("—", "-")
        try:
            tw, th = draw.textsize(title, font=font)
        except Exception:
            tw, th = (len(title) * 6, 12)
        draw.text(((W - tw) // 2, 5), title, fill=(0, 0, 0), font=font)

    # tiles + overlays
    y_off = 40
    for i, im in enumerate(images_pil):
        r, c = divmod(i, ncol)
        x0, y0 = c * img_size, y_off + r * img_size
        canvas.paste(im.resize((img_size, img_size)), (x0, y0))

        p = probs_sel[i]
        gt = gts_sel[i]
        idx = idxs[i]
        txt = f"p={p:.2f}  gt={gt}  idx={idx}"
        # black strip behind text
        strip_h = 16
        draw.rectangle([x0, y0, x0 + img_size, y0 + strip_h], fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 1), txt, fill=(255, 255, 255), font=font)

    canvas.save(out_png)
    print("Saved:", out_png)

# ---------- 6) One-call helper to make TP/FP/FN galleries for top-N classes ----------
def make_demo_galleries(y_true, y_prob, thr, classes, loader, top_by="AP",
                        top_k_classes=4, samples_per_gallery=18, out_dir="demo"):
    df = per_class_table(y_true, y_prob, classes, thr)
    top = pick_top_classes(df, by=top_by, k=top_k_classes)
    print("[demo] Top classes by", top_by, "->", top)

    for cls in top:
        j = classes.index(cls)
        # indices + probs for this class
        idx_tp, p = select_indices(y_true, y_prob, j, thr[j], k=samples_per_gallery, kind="TP")
        idx_fp, _ = select_indices(y_true, y_prob, j, thr[j], k=samples_per_gallery, kind="FP")
        idx_fn, _ = select_indices(y_true, y_prob, j, thr[j], k=samples_per_gallery, kind="FN")

        os.makedirs(out_dir, exist_ok=True)
        save_gallery_with_text(loader, idx_tp, p, y_true[:, j], os.path.join(out_dir, f"{cls}_TP.png"),
                               ncol=6, img_size=224, title=f"{cls} - True Positives")
        save_gallery_with_text(loader, idx_fp, y_prob[:, j], y_true[:, j], os.path.join(out_dir, f"{cls}_FP.png"),
                               ncol=6, img_size=224, title=f"{cls} - False Positives")
        save_gallery_with_text(loader, idx_fn, y_prob[:, j], y_true[:, j], os.path.join(out_dir, f"{cls}_FN.png"),
                               ncol=6, img_size=224, title=f"{cls} - False Negatives")
    return df