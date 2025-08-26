import os
import numpy as np
import pandas as pd
import torch
import torchvision
import argparse

from collections import Counter
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from src.models.vl.pubmedclip_loader import *
from src.models.vl.BioViL_loader import *
from src.models.vl.medclip_loader import *
from src.models.vl.ConVIRT_loader import *
from src.models.vl.BiomedCLIP_loader import *
from src.data.NIHCXRDataLoader import *
from src.data.BiomedDataLoader import *
from src.eval.zeroshot import *
from vl_backbones import *
from src.training.finetune_clip_proj import *
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import f1_score
from src.eval.eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def arg_parser():
    """Argument parser for image prompt classification."""
    parser = argparse.ArgumentParser(description="Image Prompt Classification")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--csv_path", type=str, default="", help="Path to the CSV file with image paths and labels")
    parser.add_argument("--model_type", type=str, choices=["pubmedclip", "medclip", "biovil", "convirt", "biomedclip"], required=True, help="Type of model to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--labeled_data_proportion", type=float, default=1.0, help="Proportion of labeled data to use for training (0 < p <= 1)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    return parser.parse_args()

def load_data(args):
    """Load the dataset."""
    data_root = args.data_root
    csv_path = args.csv_path
    cache_map = os.path.join(data_root, "filename_index.json")
    if args.model_type in ['medclip', 'biovil', 'convirt', 'pubmedclip']:
        dataset = NIHChestXrayDataset(
            csv_path=csv_path,
            data_root=data_root,
            transform=preprocess,
            cache_index_path=cache_map,
            drop_no_finding=True,
            keep_only_existing=True
)
    elif args.model_type == 'biomedclip':
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        dataset = NIHChestXrayDataset(csv_path=csv_path, data_root=args.data_root, transform=tokenizer, cache_map=cache_map, drop_no_finding=True, keep_only_existing=True)
    else:
        raise ValueError("Invalid model type")

    return dataset

def main():
    args = arg_parser()
    dataset = load_data(args)
    ckpt = f'checkpoints/{args.model_type}_proj_only_{args.labeled_data_proportion*100:.0f}percent.pt'

    # --- build loaders for each fraction ---
    train_loader_1, val_loader_1, test_loader_1 = make_dataloaders(dataset, train_fraction=0.01, batch_size=16)
    train_loader_10, val_loader_10, test_loader_10 = make_dataloaders(dataset, train_fraction=0.1, batch_size=32)
    train_loader_100, val_loader_100, test_loader_100 = make_dataloaders(dataset, train_fraction=1.0, batch_size=128)

    # --- map proportion -> dataloaders ---
    loader_map = {
        0.01: (train_loader_1, val_loader_1, test_loader_1),
        0.10: (train_loader_10, val_loader_10, test_loader_10),
        1.00: (train_loader_100, val_loader_100, test_loader_100),
    }

    # pick the loaders based on args.labeled_data_proportion
    try:
        train_loader, val_loader, test_loader = loader_map[args.labeled_data_proportion]
    except KeyError:
        raise ValueError(f"Unsupported labeled_data_proportion={args.labeled_data_proportion}. "
                         "Choose from {0.01, 0.10, 1.0}")

    # --- choose model ---
    if args.model_type == 'pubmedclip':
        model = build_pubmedclip(args.device)
        best, _ = finetune_pubmedclip_projection(
            pm=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=NIH14_CLASSES,
            train_text=False,
            train_logit_scale=False,
            epochs=args.epochs,
            lr_proj=2e-5,
            lr_alpha=1e-2,
            weight_decay=5e-2,
            warmup_ratio=0.1,
            pos_weight=None,
            save_path=ckpt
        )

    elif args.model_type == 'medclip':
        model = build_medclip(args.device)
        best = finetune_medclip_projection(
            mc=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=NIH14_CLASSES,
            epochs=args.epochs,
            lr_proj=1e-3,
            lr_alpha=1e-4,
            weight_decay=5e-2,
            pos_weight=None,
            save_path=ckpt
        )

    elif args.model_type == 'biovil':
        model = build_biovil_t(args.device)
        best = finetune_biovil_projection(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=NIH14_CLASSES,
            epochs=args.epochs,
            lr_proj=1e-3,
            lr_alpha=1e-4,
            weight_decay=5e-2,
            pos_weight=None,
            save_path=ckpt
        )

    elif args.model_type == 'convirt':
        # TODO: add build_convirt
        raise NotImplementedError("ConVIRT not wired yet")
    elif args.model_type == 'biomedclip':
        # TODO: add BiomedCLIPPack
        raise NotImplementedError("BiomedCLIP not wired yet")

    model.train()

if __name__ == "__main__":
    main()

