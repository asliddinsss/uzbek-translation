"""
translate.py
------------
CLI for translating text in any supported direction.

Usage:
    python src/translate.py --text "Bugun ob-havo yaxshi" --direction uz-en
    python src/translate.py --text "I want to study AI" --direction en-uz --model mbart
    python src/translate.py --csv data/samples/test.csv --direction uz-ru
"""

import argparse
import os

import torch
import pandas as pd

from dataset import DIRECTION_CONFIG, MBART_LANG_CODES
from model import (
    load_marian_finetuned, load_mbart_finetuned,
    translate_marian, translate_mbart,
)

DIRECTION_EMOJI = {
    "uz-en": "🇺🇿→🇬🇧",
    "en-uz": "🇬🇧→🇺🇿",
    "uz-ru": "🇺🇿→🇷🇺",
    "ru-uz": "🇷🇺→🇺🇿",
}


def parse_args():
    p = argparse.ArgumentParser(description="Uzbek NMT — translate text")
    p.add_argument("--text",      type=str, default=None)
    p.add_argument("--csv",       type=str, default=None)
    p.add_argument("--direction", type=str, required=True,
                   choices=["uz-en","en-uz","uz-ru","ru-uz"])
    p.add_argument("--model",     type=str, default="mbart",
                   choices=["marian","mbart"])
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--output",    type=str, default="results/translations.csv")
    p.add_argument("--num_beams", type=int, default=5)
    return p.parse_args()


def load_model(model_type, direction, model_dir, device):
    if model_dir is None:
        model_dir = f"models/{model_type}/{direction}"

    if model_type == "marian":
        return load_marian_finetuned(model_dir, device)
    else:
        return load_mbart_finetuned(model_dir, device)


def do_translate(texts, model, tokenizer, model_type, direction, device, num_beams):
    cfg = DIRECTION_CONFIG[direction]
    if model_type == "marian":
        return translate_marian(texts, model, tokenizer, device, num_beams=num_beams)
    else:
        return translate_mbart(
            texts, model, tokenizer,
            src_lang=cfg["src"], tgt_lang=cfg["tgt"],
            device=device, num_beams=num_beams,
        )


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Translate] {DIRECTION_EMOJI[args.direction]} | Model: {args.model}")

    model, tokenizer, device = load_model(args.model, args.direction, args.model_dir, device)

    if args.text:
        result = do_translate([args.text], model, tokenizer,
                              args.model, args.direction, device, args.num_beams)
        print(f"\n{'='*50}")
        print(f"  Direction:  {args.direction}")
        print(f"  Input:      {args.text}")
        print(f"  Output:     {result[0]}")
        print(f"{'='*50}\n")

    elif args.csv:
        df   = pd.read_csv(args.csv)
        srcs = df["src"].tolist()

        # Batch translate
        batch_size = 16
        all_preds  = []
        for i in range(0, len(srcs), batch_size):
            batch  = srcs[i:i+batch_size]
            preds  = do_translate(batch, model, tokenizer,
                                  args.model, args.direction, device, args.num_beams)
            all_preds.extend(preds)
            print(f"  Translated {min(i+batch_size, len(srcs))}/{len(srcs)}")

        df["translation"] = all_preds
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n[Saved] {args.output}")

        # Show samples
        print(f"\nSample translations:")
        for _, row in df.head(5).iterrows():
            print(f"  SRC: {row['src'][:70]}")
            print(f"  OUT: {row['translation'][:70]}")
            print()
    else:
        print("[Error] Provide --text or --csv")


if __name__ == "__main__":
    main()
