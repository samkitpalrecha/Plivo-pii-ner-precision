import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()

    # CHANGED DEFAULT MODEL:
    # Switched from distilbert-base-uncased → microsoft/deberta-v3-xsmall
    # Reason: modern architecture, smaller, faster, more accurate for NER.
    # ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-xsmall")

    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")

    # PRECISION-FOCUSED HYPERPARAMETERS:
    # Larger batch, more epochs, lower LR → reduces false positives.
    ap.add_argument("--batch_size", type=int, default=16)  # was 8
    ap.add_argument("--epochs", type=int, default=7)       # was 3
    ap.add_argument("--lr", type=float, default=2e-5)      # was 5e-5

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# CLASS-WEIGHTING FOR HIGH PRECISION:
# This increases penalty when model predicts false entities.
# - Boost "O" (non-entity) slightly → reduces false alarms.
# - Treat PII labels normally.
# - Reduce penalty for non-PII entities (e.g., CITY), so they don't mislead.
def build_class_weights(device):
    pii_entities = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}
    weights = []

    for lab in LABELS:
        if lab == "O":
            w = 1.3  # Strong penalty → reduces false positives (precision ↑)
        else:
            ent = lab.split("-", 1)[-1]  # remove B-/I-
            if ent in pii_entities:
                w = 1.0  # PII entities normal importance
            else:
                w = 0.8  # Non-PII (e.g., CITY/LOCATION) slightly less important
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float, device=device)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer for selected model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # DATASET LOADING:
    # Same, but ensures label alignment for weighted loss later.
    train_ds = PIIDataset(
        args.train,
        tokenizer,
        LABELS,
        max_length=args.max_length,
        is_train=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,  # Shuffling prevents memorization → improves precision
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # Load model backbone + classifier head for token classification
    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # ADAMW OPTIMIZER (standard + weight decay built-in)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # WARMUP SCHEDULER — smoother start, stabilizes boundary predictions.
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(0.06 * total_steps)  # 6% warmup is sweet spot for NER

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # CUSTOM LOSS FUNCTION WITH WEIGHTS:
    # Penalizes false positives more → ↑ precision.
    class_weights = build_class_weights(args.device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # TRAINING LOOP
    for epoch in range(args.epochs):
        running_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, num_labels)

            # USE CUSTOM WEIGHTED LOSS (not default model loss)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()

            # GRADIENT CLIPPING: prevents unstable large updates.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} avg_loss: {running_loss / max(1, len(train_dl)):.4f}")

    # Save final model + tokenizer
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()