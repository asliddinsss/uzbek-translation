"""
dataset.py
----------
Dataset loading and preprocessing for Uzbek NMT.

Supported directions:
    uz-en  (Uzbek → English)
    en-uz  (English → Uzbek)
    uz-ru  (Uzbek → Russian)
    ru-uz  (Russian → Uzbek)

Primary data source: OPUS-100 (Helsinki-NLP/opus-100)
  - uz-en config available directly
  - uz-ru constructed via pivot (uz-en + ru-en overlap)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# Direction config
# ─────────────────────────────────────────────

DIRECTION_CONFIG = {
    "uz-en": {"src": "uz", "tgt": "en", "opus_config": "en-uz",  "src_key": "uz", "tgt_key": "en"},
    "en-uz": {"src": "en", "tgt": "uz", "opus_config": "en-uz",  "src_key": "en", "tgt_key": "uz"},
    "uz-ru": {"src": "uz", "tgt": "ru", "opus_config": "uz-ru",  "src_key": "uz", "tgt_key": "ru"},
    "ru-uz": {"src": "ru", "tgt": "uz", "opus_config": "uz-ru",  "src_key": "ru", "tgt_key": "uz"},
}

# mBART language codes
MBART_LANG_CODES = {
    "uz": "uz_UZ",
    "en": "en_XX",
    "ru": "ru_RU",
}


# ─────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────

def clean_text(text: str, max_len: int = 200) -> Optional[str]:
    """Clean and validate a translation sentence."""
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    text = text.strip()
    # Remove lines that are too short or too long
    words = text.split()
    if len(words) < 2 or len(words) > max_len:
        return None
    # Remove lines with too many non-alphabetic characters (URLs, codes)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return None
    return text


def filter_pair(src: str, tgt: str, max_ratio: float = 3.0) -> bool:
    """Filter out misaligned or extremely imbalanced sentence pairs."""
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if src_len == 0 or tgt_len == 0:
        return False
    ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
    return ratio <= max_ratio


# ─────────────────────────────────────────────
# Data loading from HuggingFace
# ─────────────────────────────────────────────

def load_opus100_direction(
    direction: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load OPUS-100 parallel data for a given direction.

    Args:
        direction: one of 'uz-en', 'en-uz', 'uz-ru', 'ru-uz'
        split:     'train' | 'validation' | 'test'
        max_samples: cap for training data (None = use all)

    Returns:
        DataFrame with columns: src, tgt
    """
    from datasets import load_dataset

    cfg = DIRECTION_CONFIG[direction]
    src_key = cfg["src_key"]
    tgt_key = cfg["tgt_key"]

    # Try loading directly; uz-ru may not exist so fall back to pivot
    try:
        opus_cfg = cfg["opus_config"]
        ds = load_dataset("Helsinki-NLP/opus-100", opus_cfg, split=split)
        rows = []
        for ex in ds:
            t = ex["translation"]
            src_text = clean_text(t.get(src_key, ""))
            tgt_text = clean_text(t.get(tgt_key, ""))
            if src_text and tgt_text and filter_pair(src_text, tgt_text):
                rows.append({"src": src_text, "tgt": tgt_text})

        df = pd.DataFrame(rows)
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42).reset_index(drop=True)
        print(f"[Dataset] {direction} ({split}): {len(df):,} pairs loaded")
        return df

    except Exception as e:
        print(f"[Dataset] Could not load {direction}: {e}")
        print(f"[Dataset] Returning empty DataFrame — populate data/raw/ manually")
        return pd.DataFrame(columns=["src", "tgt"])


def load_tatoeba_direction(direction: str) -> pd.DataFrame:
    """Load Tatoeba pairs as supplementary data (especially for uz-ru)."""
    from datasets import load_dataset

    cfg = DIRECTION_CONFIG[direction]
    src_key = cfg["src_key"]
    tgt_key = cfg["tgt_key"]

    try:
        pair = f"{src_key}-{tgt_key}"
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", pair, split="test")
        rows = []
        for ex in ds:
            src_text = clean_text(ex.get("src", ""))
            tgt_text = clean_text(ex.get("tgt", ""))
            if src_text and tgt_text:
                rows.append({"src": src_text, "tgt": tgt_text})
        df = pd.DataFrame(rows)
        print(f"[Tatoeba] {direction}: {len(df):,} pairs")
        return df
    except Exception as e:
        print(f"[Tatoeba] Could not load {direction}: {e}")
        return pd.DataFrame(columns=["src", "tgt"])


def prepare_splits(direction: str, data_dir: str = "data/processed"):
    """
    Download, clean, and save train/val/test CSVs for one direction.
    Call this once before training.
    """
    from sklearn.model_selection import train_test_split

    os.makedirs(data_dir, exist_ok=True)
    dir_path = Path(data_dir) / direction.replace("-", "_")
    dir_path.mkdir(exist_ok=True)

    train_df = load_opus100_direction(direction, "train",      max_samples=500_000)
    val_df   = load_opus100_direction(direction, "validation", max_samples=2_000)
    test_df  = load_opus100_direction(direction, "test",       max_samples=2_000)

    # Supplement with Tatoeba
    tatoeba_df = load_tatoeba_direction(direction)
    if len(tatoeba_df) > 0:
        train_df = pd.concat([train_df, tatoeba_df], ignore_index=True).drop_duplicates()

    train_df.to_csv(dir_path / "train.csv", index=False)
    val_df.to_csv(dir_path   / "val.csv",   index=False)
    test_df.to_csv(dir_path  / "test.csv",  index=False)

    print(f"\n[Saved] {direction}:")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-sequence translation.

    Works for both MarianMT (direction-specific tokenizer)
    and mBART-50 (single tokenizer with language codes).

    Args:
        csv_path:       Path to CSV with columns: src, tgt
        tokenizer:      HuggingFace tokenizer
        src_lang:       Source language code (e.g. 'uz')
        tgt_lang:       Target language code (e.g. 'en')
        max_length:     Max token sequence length
        model_type:     'marian' | 'mbart'
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 128,
        model_type: str = "mbart",
    ):
        df = pd.read_csv(csv_path).dropna()
        self.sources = df["src"].tolist()
        self.targets = df["tgt"].tolist()
        self.tokenizer  = tokenizer
        self.src_lang   = src_lang
        self.tgt_lang   = tgt_lang
        self.max_length = max_length
        self.model_type = model_type

        print(f"[Dataset] {len(self.sources):,} pairs | {src_lang}→{tgt_lang}")

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text = self.sources[idx]
        tgt_text = self.targets[idx]

        if self.model_type == "mbart":
            # mBART needs language tokens set
            self.tokenizer.src_lang = MBART_LANG_CODES[self.src_lang]
            encoding = self.tokenizer(
                src_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
        else:
            # MarianMT — direction already baked into model
            encoding = self.tokenizer(
                src_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = self.tokenizer(
                text_target=tgt_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids

        # Replace padding token id with -100 (ignored in loss)
        labels = labels.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         labels,
        }


def get_dataloaders(
    direction: str,
    tokenizer,
    data_dir: str = "data/processed",
    max_length: int = 128,
    batch_size: int = 16,
    model_type: str = "mbart",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    cfg      = DIRECTION_CONFIG[direction]
    dir_path = Path(data_dir) / direction.replace("-", "_")

    def make_ds(split):
        return TranslationDataset(
            str(dir_path / f"{split}.csv"),
            tokenizer,
            src_lang=cfg["src"],
            tgt_lang=cfg["tgt"],
            max_length=max_length,
            model_type=model_type,
        )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
