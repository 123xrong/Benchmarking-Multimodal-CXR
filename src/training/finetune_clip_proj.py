import torch, numpy as np, time, gc
import torch.nn as nn
import torch.nn.functional as F
import gc
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.eval.eval import compute_metrics, find_optimal_thresholds, topk_indices, save_gallery
from src.models.vl.BiomedCLIP_loader import *
from src.eval.zeroshot import build_tokenized_text_centroids, build_text_centroids

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

# --- Build prompts once ---
def build_prompt_lists(classes):
    pos_txt, neg_txt = [], []
    for c in classes:
        cname = c.replace("_"," ").lower()
        pos_txt += [t.format(c=cname) for t in T_POS]
        neg_txt += [t.format(c=cname) for t in T_NEG]
    return pos_txt, neg_txt

def compute_pos_weights(dataloader, num_classes):
    counts_pos = np.zeros(num_classes, dtype=np.int64)
    counts_neg = np.zeros(num_classes, dtype=np.int64)
    for _, targets, _ in dataloader:   # each batch
        t = targets.numpy().astype(bool)   # (B, C)
        counts_pos += t.sum(axis=0)
        counts_neg += (~t).sum(axis=0)

    pos_weight = counts_neg / np.maximum(counts_pos, 1)  # avoid div/0
    return torch.tensor(pos_weight, dtype=torch.float32)

# --- (No-grad) centroids for eval or when train_text=False ---
@torch.no_grad()
def text_centroids_nograd(pm, classes):
    pos_txt, neg_txt = build_prompt_lists(classes)
    pos_all = pm.encode_text(pos_txt)   # [C*|T_POS|, D]
    neg_all = pm.encode_text(neg_txt)   # [C*|T_NEG|, D]
    C = len(classes); Pp, Pn = len(T_POS), len(T_NEG)
    pos_c = torch.stack([pos_all[i*Pp:(i+1)*Pp].mean(0) for i in range(C)], 0)  # [C,D]
    neg_c = torch.stack([neg_all[i*Pn:(i+1)*Pn].mean(0) for i in range(C)], 0)  # [C,D]
    return pos_c, neg_c

# --- (Grad) centroids for training when train_text=True ---
def text_centroids_grad(model, proc, classes, device):
    pos_txt, neg_txt = build_prompt_lists(classes)
    batch_pos = proc(text=pos_txt, return_tensors="pt", padding=True, truncation=True)
    batch_pos = {k: v.to(device) for k,v in batch_pos.items()}
    zpos = model.get_text_features(**batch_pos)         # [C*|T_POS|, D], grad flows into text_projection
    zpos = F.normalize(zpos, dim=-1)

    batch_neg = proc(text=neg_txt, return_tensors="pt", padding=True, truncation=True)
    batch_neg = {k: v.to(device) for k,v in batch_neg.items()}
    zneg = model.get_text_features(**batch_neg)
    zneg = F.normalize(zneg, dim=-1)

    C = len(classes); Pp, Pn = len(T_POS), len(T_NEG)
    pos_c = torch.stack([zpos[i*Pp:(i+1)*Pp].mean(0) for i in range(C)], 0)     # [C,D]
    neg_c = torch.stack([zneg[i*Pn:(i+1)*Pn].mean(0) for i in range(C)], 0)
    return pos_c, neg_c

# --- Warmup+Cosine (per-step) ---
def build_warmup_cosine(optim, total_steps, warmup_ratio=0.1, eta_min=1e-6):
    warm = max(1, int(total_steps * warmup_ratio))
    cos  = max(1, total_steps - warm)
    return SequentialLR(
        optim,
        schedulers=[LinearLR(optim, start_factor=1e-8, end_factor=1.0, total_iters=warm),
                    CosineAnnealingLR(optim, T_max=cos, eta_min=eta_min)],
        milestones=[warm]
    )

# --- Main trainer ---
def finetune_pubmedclip_projection(
    pm,                    # your PubMedCLIPPack / VLPack instance
    train_loader, val_loader, test_loader=None,
    classes=NIH14_CLASSES,
    train_text=False,      # set True to also unfreeze text_projection
    train_logit_scale=False,
    epochs=3,
    lr_proj=2e-5,          # LR for projections
    lr_alpha=1e-3,         # LR for the temperature scalar
    weight_decay=5e-2,
    warmup_ratio=0.1,
    pos_weight=None,       # optionally pass your estimated class pos_weight tensor (C,)
    save_path=None,
):
    device = next(pm.model.parameters()).device
    model = pm.model  # transformers.CLIPModel

    # 1) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze chosen parts
    train_params = []
    # image side projection
    for p in model.visual_projection.parameters():
        p.requires_grad = True; train_params.append(p)
    # optional text side projection
    if train_text:
        for p in model.text_projection.parameters():
            p.requires_grad = True; train_params.append(p)
    # optional logit_scale
    if train_logit_scale:
        model.logit_scale.requires_grad_(True)
        train_params.append(model.logit_scale)

    # learnable temperature α for our (s_pos - s_neg) logits
    alpha = nn.Parameter(torch.tensor(10.0, device=device))
    train_params.append(alpha)

    # 3) Optimizer / scheduler / loss
    optim = torch.optim.AdamW([
        {"params": [p for p in train_params if p is not alpha], "lr": lr_proj, "weight_decay": weight_decay},
        {"params": [alpha], "lr": lr_alpha, "weight_decay": 0.0},
    ])
    steps_total = max(1, len(train_loader) * epochs)
    scheduler   = build_warmup_cosine(optim, steps_total, warmup_ratio)
    scaler      = GradScaler(enabled=torch.cuda.is_available())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    # 4) Precompute text centroids if text side is frozen
    if not train_text:
        pos_c, neg_c = text_centroids_nograd(pm, classes)
        pos_c = pos_c.to(device); neg_c = neg_c.to(device)

    best_val = -np.inf
    best = None

    for ep in range(1, epochs+1):
        model.train()
        running, seen = 0.0, 0
        tic = time.time()

        for batch in train_loader:
            images, targets = (batch[0], batch[1]) if isinstance(batch, (list,tuple)) else (batch["images"], batch["labels"])
            images  = images.to(device, non_blocking=True)
            targets = targets.to(device).float()

            if train_text:
                # recompute centroids *inside the graph* so grads flow into text_projection
                pos_c, neg_c = text_centroids_grad(model, pm.proc, classes, device)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                img_z = model.get_image_features(pixel_values=images)   # (B,D), grads flow into visual_projection
                img_z = F.normalize(img_z, dim=-1)

                # per-class logits = alpha * (s_pos - s_neg)
                s_pos = img_z @ pos_c.T     # (B,C)
                s_neg = img_z @ neg_c.T
                logits = alpha * (s_pos - s_neg)

                loss = bce(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            scheduler.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)

        # ---- Validation (no grad; recompute centroids with current weights) ----
        model.eval()
        with torch.no_grad():
            pos_c_eval, neg_c_eval = text_centroids_nograd(pm, classes) if not train_text else text_centroids_nograd(pm, classes)
            pos_c_eval = pos_c_eval.to(device); neg_c_eval = neg_c_eval.to(device)

            all_prob, all_true = [], []
            for batch in val_loader:
                images, targets = (batch[0], batch[1]) if isinstance(batch, (list,tuple)) else (batch["images"], batch["labels"])
                images  = images.to(device, non_blocking=True)
                targets = targets.to(device)

                img_z = model.get_image_features(pixel_values=images)
                img_z = F.normalize(img_z, dim=-1)
                s_pos = img_z @ pos_c_eval.T
                s_neg = img_z @ neg_c_eval.T
                probs = torch.sigmoid(alpha * (s_pos - s_neg))   # (B,C)

                all_prob.append(probs.cpu().numpy())
                all_true.append(targets.cpu().numpy())

            y_prob_val = np.concatenate(all_prob, 0)
            y_true_val = np.concatenate(all_true, 0)

        # metrics + best ckpt
        thr = find_optimal_thresholds(y_true_val, y_prob_val, beta=2.0)
        val_pred = (y_prob_val >= thr).astype(np.int32)
        val_metrics = compute_metrics(y_true_val, y_prob_val)
        val_metrics["f1_macro@val_th"] = f1_score(y_true_val, val_pred, average="macro", zero_division=0)
        score = (val_metrics.get("map_macro", 0.0) + val_metrics.get("auroc_macro", 0.0)) / 2.0

        print(f"[proj-ft] epoch {ep}/{epochs}  train_loss={train_loss:.4f}  "
              f"mAP_macro={val_metrics.get('map_macro', float('nan')):.4f}  "
              f"AUROC_macro={val_metrics.get('auroc_macro', float('nan')):.4f}  "
              f"F1_macro@th={val_metrics['f1_macro@val_th']:.4f}")

        if score > best_val:
            best_val = score
            best = {
                "epoch": ep,
                "state_dict": model.state_dict(),
                "alpha": float(alpha.detach().cpu()),
                "thresholds": thr,
                "val_metrics": val_metrics,
                "train_text": train_text,
            }
            if save_path:
                torch.save(best, save_path)

        gc.collect(); torch.cuda.empty_cache()

    # ----- Test (optional) -----
    test_report = {}
    if best is not None and test_loader is not None:
        model.load_state_dict(best["state_dict"])
        pos_c_eval, neg_c_eval = text_centroids_nograd(pm, classes)
        pos_c_eval = pos_c_eval.to(device); neg_c_eval = neg_c_eval.to(device)
        a = torch.tensor(best["alpha"], device=device)

        all_prob, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                images, targets = (batch[0], batch[1]) if isinstance(batch, (list,tuple)) else (batch["images"], batch["labels"])
                images  = images.to(device, non_blocking=True)
                targets = targets.to(device)

                img_z = model.get_image_features(pixel_values=images)
                img_z = F.normalize(img_z, dim=-1)
                s_pos = img_z @ pos_c_eval.T
                s_neg = img_z @ neg_c_eval.T
                probs = torch.sigmoid(a * (s_pos - s_neg))
                all_prob.append(probs.cpu().numpy())
                all_true.append(targets.cpu().numpy())

        y_prob_te = np.concatenate(all_prob, 0)
        y_true_te = np.concatenate(all_true, 0)
        test_report = compute_metrics(y_true_te, y_prob_te)
        y_pred_te = (y_prob_te >= best["thresholds"]).astype(np.int32)
        test_report["f1_macro@val_th"] = f1_score(y_true_te, y_pred_te, average="macro", zero_division=0)

    return best, test_report

def finetune_medclip_projection(
    mc, train_loader, val_loader, test_loader=None,
    classes=NIH14_CLASSES,
    train_text=False,          # <- allow unfreezing text projection too
    train_logit_scale=False,   # <- optional if MedCLIP has logit_scale
    epochs=3,
    lr_proj=2e-5,              # learning rate for projections
    lr_alpha=1e-3,             # learning rate for scalar α
    weight_decay=5e-2,
    pos_weight=None,
    save_path=None,
):
    device = mc.device
    model = mc.model

    # ---- Freeze everything first ----
    for p in model.parameters():
        p.requires_grad = False

    # ---- Unfreeze chosen parts ----
    train_params = []
    if hasattr(model, "visual_projection"):
        for p in model.visual_projection.parameters():
            p.requires_grad = True; train_params.append(p)
            print(sum(p.numel() for p in model.visual_projection.parameters()), "params in visual_projection")
    if train_text and hasattr(model, "text_projection"):
        for p in model.text_projection.parameters():
            p.requires_grad = True; train_params.append(p)
    if train_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad_(True)
        train_params.append(model.logit_scale)

    # ---- Learnable temperature α ----
    alpha = nn.Parameter(torch.tensor(10.0, device=device))
    train_params.append(alpha)

    # ---- Optimizer / Scheduler ----
    optim = torch.optim.AdamW([
        {"params": [p for p in train_params if p is not alpha], "lr": lr_proj, "weight_decay": weight_decay},
        {"params": [alpha], "lr": lr_alpha, "weight_decay": 0.0},
    ])
    scaler = GradScaler(enabled=torch.cuda.is_available())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    best_val, best = -np.inf, None

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running, seen = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device).float()
            optim.zero_grad(set_to_none=True)

            # recompute text centroids inside graph if text side is trainable
            if train_text:
                pos_c, neg_c = text_centroids_grad(model, mc, classes, device)
            else:
                with torch.no_grad():
                    pos_c, neg_c = text_centroids_nograd(mc, classes)
                pos_c, neg_c = pos_c.to(device), neg_c.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                img_z = mc.encode_image(images)   # grads flow if visual_projection is trainable
                s_pos, s_neg = img_z @ pos_c.T, img_z @ neg_c.T
                logits = alpha * (s_pos - s_neg)
                loss = bce(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)

        # ---- Validation ----
        all_prob, all_true = [], []
        model.eval()
        with torch.no_grad():
            pos_c, neg_c = text_centroids_nograd(mc, classes)
            pos_c, neg_c = pos_c.to(device), neg_c.to(device)
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                img_z = mc.encode_image(images)
                probs = torch.sigmoid(alpha * (img_z @ pos_c.T - img_z @ neg_c.T))
                all_prob.append(probs.cpu().numpy())
                all_true.append(targets.cpu().numpy())

        y_prob_val = np.concatenate(all_prob, axis=0)
        y_true_val = np.concatenate(all_true, axis=0)

        val_metrics = compute_metrics(y_true_val, y_prob_val)
        score = (val_metrics.get("map_macro", 0.0) + val_metrics.get("auroc_macro", 0.0)) / 2.0

        print(f"[medclip-ft] epoch {ep}/{epochs} train_loss={train_loss:.4f} "
              f"AUROC_macro={val_metrics['auroc_macro']:.4f} mAP_macro={val_metrics['map_macro']:.4f}")

        if score > best_val:
            best_val = score
            best = {
                "epoch": ep,
                "alpha": float(alpha.detach().cpu()),
                "val_metrics": val_metrics,
                "state_dict": model.state_dict(),
                "train_text": train_text,
            }
            if save_path:
                torch.save(best, save_path)

    return best

def finetune_biomedclip_projection(
    biomedclip,
    tokenizer,             
    train_loader,
    val_loader,
    classes,
    epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pos_weight: torch.Tensor = None,   # handle class imbalance
    save_path: str = None,
) -> Dict:

    model = biomedclip.model
    # ---- Freeze backbone, unfreeze projection head(s)
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = []
    for name, p in model.named_parameters():
        if "image_projection" in name or "text_projection" in name:
            p.requires_grad = True
            trainable_params.append(p)

    # learnable temperature scaling
    logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))  # CLIP init ~ 1/τ
    trainable_params.append(logit_scale)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    best_val, best_state = -np.inf, None

    # ---- Precompute text centroids (these will update since text_projection is trainable!)
    def get_text_centroids():
        from src.eval.zeroshot import build_tokenized_text_centroids  # or use your local version
        return build_tokenized_text_centroids(model, tokenizer, classes, device)

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running, seen = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device).float()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                img_z = F.normalize(model.encode_image(images), dim=-1)
                pos_c, neg_c = build_tokenized_text_centroids(model, tokenizer, classes, device)
                pos_c = F.normalize(pos_c.to(device), dim=-1)
                neg_c = F.normalize(neg_c.to(device), dim=-1)
                s_pos, s_neg = img_z @ pos_c.T, img_z @ neg_c.T
                logits = logit_scale.exp() * (s_pos - s_neg)
                loss = bce(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)

        # ---- Validation ----
        model.eval()
        all_prob, all_true = [], []
        with torch.no_grad():
            pos_c, neg_c = get_text_centroids()
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                img_z = model.encode_image(images)
                probs = torch.sigmoid(logit_scale.exp() * (img_z @ pos_c.T - img_z @ neg_c.T))
                all_prob.append(probs.cpu().numpy())
                all_true.append(targets.cpu().numpy())
        y_prob, y_true = np.concatenate(all_prob), np.concatenate(all_true)

        val_metrics = compute_metrics(y_true, y_prob)
        print(f"[biomedclip-ft] epoch {ep}/{epochs} train_loss={train_loss:.4f} "
              f"AUROC_macro={val_metrics['auroc_macro']:.4f} mAP_macro={val_metrics['map_macro']:.4f}")

        # Save best
        score = (val_metrics.get("map_macro", 0.0) + val_metrics.get("auroc_macro", 0.0)) / 2.0
        if score > best_val:
            best_val = score
            best_state = {
                "epoch": ep,
                "model": model.state_dict(),
                "logit_scale": float(logit_scale.detach().cpu()),
                "val_metrics": val_metrics,
            }
            if save_path:
                torch.save(best_state, save_path)

    return best_state

def finetune_convirt_projection(
    model,
    train_loader,
    val_loader=None,
    classes=NIH14_CLASSES,
    epochs=5,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pos_weight=None,
    save_path=None,
):
    model.to(device)

    # --- Freeze everything ---
    for p in model.parameters():
        p.requires_grad = False
    # --- Unfreeze projection heads + logit_scale ---
    for p in model.image_proj.parameters():
        p.requires_grad = True
    for p in model.text_proj.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad_(True)

    params = list(model.image_proj.parameters()) + \
             list(model.text_proj.parameters()) + \
             [model.logit_scale]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    bce = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )

    best_val, best = -np.inf, None

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running, seen = 0.0, 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device).float()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                img_z = model.encode_image(images)        # [B,D]
                pos_c, neg_c = build_text_centroids(model, classes)  # frozen text side
                pos_c, neg_c = pos_c.to(device), neg_c.to(device)

                s_pos, s_neg = img_z @ pos_c.T, img_z @ neg_c.T
                logits = model.logit_scale.exp() * (s_pos - s_neg)
                loss = bce(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)
        scheduler.step()

        # ---- Validation ----
        val_metrics = {}
        if val_loader is not None:
            model.eval()
            all_prob, all_true = [], []
            with torch.no_grad():
                pos_c, neg_c = build_text_centroids(model, classes)
                pos_c, neg_c = pos_c.to(device), neg_c.to(device)

                for images, targets, _ in val_loader:
                    images, targets = images.to(device), targets.to(device)
                    img_z = model.encode_image(images)
                    s_pos, s_neg = img_z @ pos_c.T, img_z @ neg_c.T
                    probs = torch.sigmoid(model.logit_scale.exp() * (s_pos - s_neg))
                    all_prob.append(probs.cpu().numpy())
                    all_true.append(targets.cpu().numpy())

            y_prob_val = np.concatenate(all_prob, 0)
            y_true_val = np.concatenate(all_true, 0)

            val_metrics = compute_metrics(y_true_val, y_prob_val)
            thr = find_optimal_thresholds(y_true_val, y_prob_val, beta=2.0)
            y_pred_val = (y_prob_val >= thr).astype(np.int32)
            val_metrics["f1_macro@val_th"] = f1_score(
                y_true_val, y_pred_val, average="macro", zero_division=0
            )

            score = (val_metrics.get("map_macro", 0.0) +
                     val_metrics.get("auroc_macro", 0.0)) / 2.0

            print(f"[ConVIRT-FT] epoch {ep}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"mAP_macro={val_metrics.get('map_macro', float('nan')):.4f}  "
                  f"AUROC_macro={val_metrics.get('auroc_macro', float('nan')):.4f}  "
                  f"F1_macro@th={val_metrics['f1_macro@val_th']:.4f}")

            if score > best_val:
                best_val = score
                best = {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "thresholds": thr,
                    "val_metrics": val_metrics,
                }
                if save_path:
                    torch.save(best, save_path)

        gc.collect(); torch.cuda.empty_cache()

    return model, best

def finetune_biovil_projection(
    model, train_loader, val_loader, test_loader=None,
    classes=NIH14_CLASSES,
    train_text=False,          # <- allow unfreezing text projection too
    train_logit_scale=False,   # <- optional if BioViL has logit_scale
    epochs=3,
    lr_proj=1e-4,              # learning rate for projections
    lr_alpha=1e-3,             # learning rate for scalar α
    weight_decay=5e-2,
    pos_weight=None,
    save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Freeze everything first ----
    for p in model.parameters():
        p.requires_grad = False

    # ---- Unfreeze chosen parts ----
    train_params = []
    if hasattr(model, "visual_projection"):
        for p in model.visual_projection.parameters():
            p.requires_grad = True; train_params.append(p)
    if train_text and hasattr(model, "text_projection"):
        for p in model.text_projection.parameters():
            p.requires_grad = True; train_params.append(p)
    if train_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad_(True)
        train_params.append(model.logit_scale)

    # ---- Learnable temperature α ----
    alpha = nn.Parameter(torch.tensor(10.0, device=device))
    train_params.append(alpha)

    # ---- Optimizer / Scheduler ----
    optim = torch.optim.AdamW([
        {"params": [p for p in train_params if p is not alpha], "lr": lr_proj, "weight_decay": weight_decay},
        {"params": [alpha], "lr": lr_alpha, "weight_decay": 0.0},
    ])
    scaler = GradScaler(enabled=torch.cuda.is_available())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    best_val, best = -np.inf, None

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running, seen = 0.0, 0
        for images, targets, paths in train_loader:
            targets = targets.to(device).float()
            optim.zero_grad(set_to_none=True)

            # recompute text centroids inside graph if text side is trainable
            if train_text:
                pos_c, neg_c = text_centroids_grad(model, classes)
            else:
                with torch.no_grad():
                    pos_c, neg_c = text_centroids_nograd(model, classes)
                pos_c, neg_c = pos_c.to(device), neg_c.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                img_z = model.encode_image(paths)   # grads flow if visual_projection is trainable
                s_pos, s_neg = img_z @ pos_c.T, img_z @ neg_c.T
                logits = alpha * (s_pos - s_neg)
                loss = bce(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            running += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running / max(1, seen)

        # ---- Validation ----
        all_prob, all_true = [], []
        model.eval()
        with torch.no_grad():
            pos_c, neg_c = text_centroids_nograd(model, classes)
            pos_c, neg_c = pos_c.to(device), neg_c.to(device)
            for _, targets, paths in val_loader:
                targets = targets.to(device)
                img_z = model.encode_image(paths)
                probs = torch.sigmoid(alpha * (img_z @ pos_c.T - img_z @ neg_c.T))
                all_prob.append(probs.cpu().numpy())
                all_true.append(targets.cpu().numpy())

        y_prob_val = np.concatenate(all_prob, axis=0)
        y_true_val = np.concatenate(all_true, axis=0)

        val_metrics = compute_metrics(y_true_val, y_prob_val)
        score = (val_metrics.get("map_macro", 0.0) + val_metrics.get("auroc_macro", 0.0)) / 2.0

        print(f"[biovil-ft] epoch {ep}/{epochs} train_loss={train_loss:.4f} "
              f"AUROC_macro={val_metrics['auroc_macro']:.4f} mAP_macro={val_metrics['map_macro']:.4f}")

        if score > best_val:
            best_val = score
            best = {
                "epoch": ep,
                "alpha": float(alpha.detach().cpu()),
                "val_metrics": val_metrics,
                "state_dict": model.state_dict(),
                "train_text": train_text,
            }
            if save_path:
                torch.save(best, save_path)

    return best