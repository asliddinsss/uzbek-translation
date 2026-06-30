"""
train.py
--------
Unified fine-tuning script for MarianMT and mBART-50.

Usage:
    python src/train.py --model marian --direction uz-en --epochs 3
    python src/train.py --model mbart  --direction uz-en --epochs 5
"""

import os
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import get_dataloaders, DIRECTION_CONFIG
from model import build_model, save_model, translate_marian, translate_mbart
from evaluate import compute_bleu_chrf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str, required=True, choices=["marian","mbart"])
    p.add_argument("--direction",   type=str, required=True,
                   choices=["uz-en","en-uz","uz-ru","ru-uz"])
    p.add_argument("--data_dir",    type=str, default="data/processed")
    p.add_argument("--save_dir",    type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--max_length",  type=int, default=128)
    p.add_argument("--warmup_ratio",type=float, default=0.1)
    p.add_argument("--patience",    type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults
    args.lr       = args.lr or (5e-5 if args.model == "marian" else 2e-5)
    args.save_dir = args.save_dir or f"models/{args.model}/{args.direction}"

    cfg = DIRECTION_CONFIG[args.direction]

    print(f"\n{'='*55}")
    print(f"  Uzbek NMT — {args.model.upper()} | {args.direction}")
    print(f"  Device: {device} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"{'='*55}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Build model ──────────────────────────────
    model, tokenizer = build_model(args.model, direction=args.direction)
    model = model.to(device)

    # ── Dataloaders ──────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        direction=args.direction,
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        model_type=args.model,
    )

    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )

    log_path   = Path(args.results_dir) / f"{args.model}_{args.direction}_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_bleu", "val_chrf"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_bleu, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──────────────────────────
        model.eval()
        val_loss = 0
        all_preds, all_refs = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch}] Val", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Loss
                outputs   = model(**batch)
                val_loss += outputs.loss.item()

                # Generate translations for BLEU
                if args.model == "marian":
                    gen = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        num_beams=4, max_length=args.max_length,
                    )
                    preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
                else:
                    from dataset import MBART_LANG_CODES
                    gen = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        forced_bos_token_id=tokenizer.lang_code_to_id[MBART_LANG_CODES[cfg["tgt"]]],
                        num_beams=5, max_length=args.max_length,
                    )
                    preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

                labels = batch["labels"].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                refs   = tokenizer.batch_decode(labels, skip_special_tokens=True)

                all_preds.extend(preds)
                all_refs.extend(refs)

        avg_val_loss = val_loss / len(val_loader)
        bleu, chrf   = compute_bleu_chrf(all_refs, all_preds)
        elapsed      = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"BLEU: {bleu:.2f} | chrF: {chrf:.4f} | {elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss":   round(avg_val_loss, 4),
                "val_bleu":   round(bleu, 2),
                "val_chrf":   round(chrf, 4),
            })

        if bleu > best_bleu:
            best_bleu = bleu
            patience_counter = 0
            save_model(model, tokenizer, args.save_dir)
            print(f"  ✓ Best model saved (BLEU: {best_bleu:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs.")
                break

    print(f"\nTraining complete | Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
