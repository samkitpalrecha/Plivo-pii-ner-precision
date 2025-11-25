import json
import argparse
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, label_is_pii


def bio_to_spans(text, offsets, label_ids):
    """
    Convert token-level BIO labels to character-level spans on the original text.

    Args:
        text: original string.
        offsets: list of (start, end) char offsets per token from tokenizer.
        label_ids: list of predicted label ids per token.

    Returns:
        List of (start, end, ent_type) tuples.
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # tokenizer may produce special tokens with offset (0, 0); skip them
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")

        if label == "O":
            # close any open entity when we hit "O"
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            # starting a new entity; close any previous one
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            # continuation of the same entity type
            if current_label == ent_type:
                current_end = end
            else:
                # inconsistent I- tag; close previous and start a new one
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    # flush any remaining open entity at the end
    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


# ----------------------------
# Precision-oriented validators
# ----------------------------

def _only_digits(text):
    """Return only digit characters from text."""
    return re.sub(r"\D", "", text)


def _passes_luhn(number_str):
    """
    Basic Luhn checksum for credit card style validation.
    Returns True if the number string passes Luhn.
    """
    if not number_str.isdigit():
        return False

    total = 0
    reverse_digits = number_str[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def is_valid_credit_card(span_text):
    """
    Validate CREDIT_CARD span using digits length and Luhn check.

    Strategy:
    - Extract digits from span.
    - Require length between 13 and 19 (typical card lengths).
    - Filter out trivial repeats (all digits same).
    - Pass Luhn checksum.
    """
    digits = _only_digits(span_text)
    if len(digits) < 13 or len(digits) > 19:
        return False
    if len(set(digits)) == 1:
        # e.g. 1111111111111111
        return False
    if not _passes_luhn(digits):
        return False
    return True


def is_valid_phone(span_text):
    """
    Validate PHONE span using digit length and basic pattern.

    Strategy:
    - Extract digits from span.
    - Require length between 7 and 15.
    - Ensure there is at least one non-digit separator in original
      (space, dash, plus) to avoid confusion with credit card strings.
    """
    digits = _only_digits(span_text)
    if len(digits) < 7 or len(digits) > 15:
        return False

    # Simple heuristic to avoid mixing up with credit cards: if 15+ digits
    # with no separators, more likely not a phone number in this context.
    if len(digits) >= 15 and not re.search(r"[ +\-()]", span_text):
        return False

    return True


def is_valid_email(span_text):
    """
    Validate EMAIL span with a conservative regex.

    Strategy:
    - Use a fairly strict email pattern:
      <local>@<domain>.<tld>
    - This favors precision over recall (spoken "at gmail dot com" is likely
      to be rejected here, which is acceptable under precision-first goals).
    """
    text = span_text.strip()
    # basic email regex; intentionally conservative
    email_pattern = re.compile(
        r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
    )
    return bool(email_pattern.match(text))


def is_valid_person_name(span_text):
    """
    Light validation for PERSON_NAME spans.

    Strategy:
    - Require at least one alphabetic character.
    - Trim whitespace and enforce reasonable length bounds.
    - Reject spans that are mostly digits or punctuation.
    """
    text = span_text.strip()
    if len(text) == 0:
        return False
    if len(text) > 80:
        # very long names are unlikely; probably a model error
        return False
    if not re.search(r"[A-Za-z]", text):
        # no alphabetic chars → not a plausible name
        return False
    # Heuristic: if more than half of characters are non-letters, it is suspicious
    letters = len(re.findall(r"[A-Za-z]", text))
    if letters / max(1, len(text)) < 0.4:
        return False
    return True


def filter_span_by_label(text, start, end, label):
    """
    Apply label-specific post-processing filters.

    Returns:
        True if span should be kept, False if it should be discarded.
    """
    span_text = text[start:end]

    if label == "CREDIT_CARD":
        return is_valid_credit_card(span_text)
    elif label == "PHONE":
        return is_valid_phone(span_text)
    elif label == "EMAIL":
        return is_valid_email(span_text)
    elif label == "PERSON_NAME":
        return is_valid_person_name(span_text)
    # For DATE, CITY, LOCATION, etc., no extra filters for now.
    # They are kept as predicted by the model.
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    # Load tokenizer; if model_name is provided, prefer that, else use model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            # Tokenize with offsets. Offsets map each subtoken back to
            # character spans in the original string.
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]  # (seq_len, num_labels)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            # Convert BIO tags to spans at character level
            spans = bio_to_spans(text, offsets, pred_ids)

            # Precision-oriented post-processing:
            # - Apply label-specific validators.
            # - Discard spans that fail validation to reduce false positives.
            ents = []
            for s, e, lab in spans:
                if not filter_span_by_label(text, s, e, lab):
                    continue
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()