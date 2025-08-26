import os
import glob
import json
import random
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms

NIH14_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

def build_filename_index(data_root: str,
                         cache_path: Optional[str] = None,
                         exts: Tuple[str,...] = (".png", ".jpg", ".jpeg")) -> dict:
    """
    Scan all shard folders under {data_root}/images_*/images/* and return
    a dict: { "00000001_000.png": "/full/path/images_001/images/00000001_000.png", ... }
    Optionally cache to disk for super-fast re-loads.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    index = {}
    # Shards look like images_001/images, images_002/images, ...
    for shard_dir in glob.glob(os.path.join(data_root, "images_*", "images")):
        for ext in exts:
            for p in glob.glob(os.path.join(shard_dir, f"*{ext}")):
                fname = os.path.basename(p)
                index[fname] = p

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(index, f)

    return index


def labels_to_multihot(labels_str: str,
                       class_list: List[str],
                       drop_no_finding: bool = True) -> torch.Tensor:
    """
    Convert the NIH 'Finding Labels' pipe-separated string to a multi-hot torch vector.
    """
    y = torch.zeros(len(class_list), dtype=torch.float32)
    labels = [s.strip() for s in labels_str.split("|")] if isinstance(labels_str, str) else []

    # Optionally ignore 'No Finding'
    if drop_no_finding:
        labels = [l for l in labels if l != "No Finding"]

    for l in labels:
        if l in class_list:
            y[class_list.index(l)] = 1.0
        # else: label not in 14-class set (rare), ignore

    return y



class NIHChestXrayDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 data_root: str,
                 class_list: List[str] = NIH14_CLASSES,
                 transform=None,
                 cache_index_path: Optional[str] = None,
                 drop_no_finding: bool = True,
                 keep_only_existing: bool = True):
        """
        csv_path: path to Data_Entry_2017.csv
        class_list: order of output labels
        transform: torchvision transform for images
        cache_index_path: json path to cache the filename->fullpath map
        drop_no_finding: drop 'No Finding' label from target (common)
        keep_only_existing: keep only rows whose image is physically found
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.class_list = class_list
        self.transform = transform
        self.drop_no_finding = drop_no_finding

        # Build a fast filename->path map across all shards
        self.filename_index = build_filename_index(
            data_root, cache_path=cache_index_path
        )

        # Keep only rows that exist on disk (prevents file-not-found during training)
        if keep_only_existing:
            self.df = self.df[self.df["Image Index"].isin(self.filename_index.keys())].reset_index(drop=True)

        # Optional: you can also keep only rows with at least one disease (no 'No Finding')
        if self.drop_no_finding:
            # Keep even if there are diseases + 'No Finding' together (rare); we only drop rows that are strictly No Finding, if you prefer:
            strictly_no_finding = self.df["Finding Labels"].eq("No Finding")
            # Comment the next line if you *want* to include strictly 'No Finding' rows with zero vector targets.
            self.df = self.df[~strictly_no_finding].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.filename_index[img_name]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = labels_to_multihot(row["Finding Labels"], self.class_list,
                                    drop_no_finding=self.drop_no_finding)
        return image, target, img_path


def default_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Optional: add normalization if your backbone expects it (e.g., CLIP or ImageNet)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])


def split_indices(n: int, train_ratio=0.8, val_ratio=0.1, seed=42):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train+n_val]
    test_idx = idxs[n_train+n_val:]
    return train_idx, val_idx, test_idx


def make_dataloaders(dataset: Dataset,
                     batch_size=32,
                     num_workers=4,
                     train_ratio=0.8,
                     val_ratio=0.1,
                     seed=42,
                     shuffle_train=True,
                     train_fraction=1.0   # NEW: allows 0.01, 0.1, etc
                     ):
    n = len(dataset)
    train_idx, val_idx, test_idx = split_indices(n, train_ratio, val_ratio, seed)

    # --- Optionally subsample train indices ---
    if train_fraction < 1.0:
        n_sample = int(len(train_idx) * train_fraction)
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(train_idx, n_sample, replace=False)

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
