"""
model.py
--------
Model loading for MarianMT and mBART-50 translation.

MarianMT: lightweight, direction-specific, fast to train
mBART-50: multilingual, handles all 4 directions, best quality
"""

import os
import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
)

from dataset import MBART_LANG_CODES

# ─────────────────────────────────────────────
# MarianMT direction → HuggingFace model names
# ─────────────────────────────────────────────

MARIAN_MODELS = {
    "uz-en": "Helsinki-NLP/opus-mt-uz-en",
    "en-uz": "Helsinki-NLP/opus-mt-en-uz",
    "uz-ru": "Helsinki-NLP/opus-mt-uz-ru",
    "ru-uz": "Helsinki-NLP/opus-mt-ru-uz",
}

MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


# ─────────────────────────────────────────────
# MarianMT
# ─────────────────────────────────────────────

def build_marian(direction: str):
    """
    Load MarianMT model + tokenizer for a given translation direction.
    Falls back gracefully if a specific direction model isn't available.
    """
    model_name = MARIAN_MODELS.get(direction)
    if model_name is None:
        raise ValueError(f"No MarianMT model defined for direction: {direction}")

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model     = MarianMTModel.from_pretrained(model_name)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[MarianMT] Loaded {model_name} | Params: {total:,}")
        return model, tokenizer
    except Exception as e:
        print(f"[MarianMT] Could not load {model_name}: {e}")
        print(f"  → MarianMT may not have a pre-trained {direction} model.")
        print(f"  → Use mBART for better Uzbek coverage.")
        return None, None


def translate_marian(texts, model, tokenizer, device, num_beams=4, max_length=128):
    """Batch translate with MarianMT."""
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs, num_beams=num_beams, max_length=max_length)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


# ─────────────────────────────────────────────
# mBART-50
# ─────────────────────────────────────────────

def build_mbart():
    """
    Load mBART-large-50 for many-to-many translation.
    One model handles all four directions.
    """
    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL)
    model     = MBartForConditionalGeneration.from_pretrained(MBART_MODEL)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mBART-50] Loaded {MBART_MODEL}")
    print(f"  Total: {total:,} | Trainable: {trainable:,}")
    return model, tokenizer


def translate_mbart(
    texts: list,
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    device,
    num_beams: int = 5,
    max_length: int = 128,
) -> list:
    """Batch translate with mBART-50."""
    model.eval()
    tokenizer.src_lang = MBART_LANG_CODES[src_lang]
    tgt_lang_code      = MBART_LANG_CODES[tgt_lang]

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
            num_beams=num_beams,
            max_length=max_length,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────

def save_model(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[Model] Saved → {save_dir}/")


def load_marian_finetuned(save_dir: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained(save_dir)
    model     = MarianMTModel.from_pretrained(save_dir).to(device)
    return model, tokenizer, device


def load_mbart_finetuned(save_dir: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained(save_dir)
    model     = MBartForConditionalGeneration.from_pretrained(save_dir).to(device)
    return model, tokenizer, device


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_model(model_type: str, direction: str = None):
    if model_type == "marian":
        assert direction is not None, "direction required for MarianMT"
        return build_marian(direction)
    elif model_type == "mbart":
        return build_mbart()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'marian' or 'mbart'.")
