import torch
import torchvision.transforms.functional as TF
import numpy as np
import os
import pandas as pd

from open_clip import create_model_from_pretrained, get_tokenizer
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from torchvision import transforms

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def create_datalist(csv, image_dir):
    """
    Create a list of dictionaries containing image and text data.
    """
    data_list = []
    for _, row in csv.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        img_dir = f"{image_dir}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        if not os.path.exists(img_dir):
            print(f"Image not found: {img_dir}")
            continue
        try:
            image = Image.open(img_dir).convert("RGB")
        except OSError:
            print(f"[Warning] Corrupt image skipped: {img_dir}")
            continue

        processed = preprocess(image).unsqueeze(0)  # Add batch dimension
        text = row['findings']
        if pd.isna(text):
            print(f"Text not found for subject {subject_id}, study {study_id}, dicom {dicom_id}")
            continue
        data_list.append({'cropped_images': processed, 'text': text})
    return data_list
    
def make_multi_val_test_loaders(
    dataset,
    train_ratio=0.95,
    sizes=(8, 16, 32, 64),
    batch_size=64,
    seed=42,
    num_workers=4,
    pin_memory=True,
):
    """
    Splits `dataset` into a single train set and *multiple* val/test pairs of increasing sizes.
    For each size s in `sizes`, we create:
        - val_s: first s items from a fixed shuffled holdout
        - test_s: next s items from the same permutation (disjoint from val_s for that s)
    Note: val_8 ⊂ val_16 ⊂ val_32 ⊂ ... (nested), same for test; all derived from the same permutation for reproducibility.
    """
    N = len(dataset)
    train_size = int(N * train_ratio)
    holdout_size = N - train_size
    if holdout_size < 2 * min(sizes):
        raise ValueError(f"Holdout ({holdout_size}) too small for smallest paired size 2*{min(sizes)}.")

    # Reproducible split + permutation
    g = torch.Generator()
    g.manual_seed(seed)
    train_set, holdout_set = random_split(dataset, [train_size, holdout_size], generator=g)

    # Shuffle indices of the holdout once; then take prefixes for each size
    perm = torch.randperm(holdout_size, generator=g).tolist()
    holdout_indices = holdout_set.indices  # indices into original dataset

    # Only keep sizes that are <= batch_size and feasible (2*s <= holdout_size)
    filtered_sizes = [s for s in sizes if s <= batch_size and 2 * s <= holdout_size]
    if not filtered_sizes:
        raise ValueError("No feasible sizes after filtering by batch_size and holdout capacity.")

    val_loaders = {}
    test_loaders = {}

    for s in filtered_sizes:
        val_idx_local = perm[:s]
        test_idx_local = perm[s:2*s]
        # Map local holdout positions to original dataset indices
        val_idx = [holdout_indices[i] for i in val_idx_local]
        test_idx = [holdout_indices[i] for i in test_idx_local]

        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        # Use batch_size for all; DataLoader will make a single (smaller) batch if len < batch_size
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        val_loaders[s] = val_loader
        test_loaders[s] = test_loader

    # Also give a standard train loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loaders, test_loaders, filtered_sizes

class BiomedCLIPDataset(Dataset):
    def __init__(self, data_list, tokenizer, train=True):
        self.pairs = []
        self.tokenizer = tokenizer
        self.train = train
        for data in data_list:
            for img, txt in zip(data['cropped_images'], data['text']):
                self.pairs.append((img, txt))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_tensor, text = self.pairs[idx]
        if self.train:
            image_tensor = augment_image(self.pairs[idx][0])
        text_tensor = self.tokenizer(text, context_length=256)

        return image_tensor, text_tensor

def collate_fn(batch, tokenizer=None, max_length=256):
    """
    batch: list of (image_tensor, text_string)
    tokenizer: decoder tokenizer (e.g., BioGPT tokenizer)
    """
    images, texts = zip(*batch)
    images = torch.stack(images)

    # Tokenize the target texts
    tokenized = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    return images, input_ids, attention_mask, texts  # return raw texts for evaluation

# Split into training, validation, and test sets
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)  # For reproducibility          
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def augment_image(image_tensor):
    """
    Apply random augmentations to the image tensor.
    """
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        image_tensor = TF.hflip(image_tensor)

    # Random rotation
    angle = torch.randint(-30, 30, (1,)).item()
    image_tensor = TF.rotate(image_tensor, angle)

    # Random color jitter
    if torch.rand(1).item() > 0.5:
        image_tensor = TF.adjust_brightness(image_tensor, brightness_factor=torch.rand(1).item() * 0.5 + 0.75)
        image_tensor = TF.adjust_contrast(image_tensor, contrast_factor=torch.rand(1).item() * 0.5 + 0.75)

    return image_tensor

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=128, shuffle=False, num_workers=0):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

