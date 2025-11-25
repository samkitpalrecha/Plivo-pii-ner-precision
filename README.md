
(Reference: model configuration change in `train.py`)

### 2.3 Rationale
| DistilBERT | DeBERTa v3 xsmall |
|------------|-------------------|
| Older architecture | Newer architecture using disentangled attention |
| Limited semantic range | Stronger contextual boundary modeling |
| Average performance in noisy transcripts | High-quality token semantics with fewer parameters |

**DeBERTa-v3-xsmall achieves stronger token-level representation even though it is computationally smaller**, making it ideal for low-latency and high-precision tasks.

---

## 3. Precision-Focused Training Strategy

The training design was modified to discourage over-prediction of entities and enforce conservative learning.

### 3.1 Hyperparameter Adjustments

| Parameter | Previous | Updated | Reason |
|-----------|----------|---------|--------|
| Batch size | 8 | 16 | Reduces noisy gradient oscillations |
| Epochs | 3 | 7 | Allows stable convergence |
| Learning rate | 5e-5 | 2e-5 | Prevents overly aggressive learning in noisy data |

These modifications ensure more stable boundary prediction and reduce spurious PII detection.

---

### 3.2 Class-Weighted Loss

Weights were assigned to labels to penalize false positives more aggressively. Implemented using `CrossEntropyLoss` with custom weights.

| Entity Class | Weight | Explanation |
|--------------|--------|-------------|
| O (Non-PII) | **1.3** | Forces the model to think before labeling text as PII |
| PII Labels | **1.0** | Standard influence on learning |
| Non-PII Entities (e.g., CITY, LOCATION) | **0.8** | Reduces their effect on model based decisions |

This modification allows the classifier to remain **cautious**, increasing the precision of PII detection.

---

### 3.3 Learning Stabilization Measures
The following were introduced:

- **6% warmup** of total training steps
- **Gradient clipping (norm = 1.0)**

These prevent unstable jumps when learning token boundaries, particularly in noisy speech data.

---

## 4. Precision-Oriented Post-Processing

### 4.1 Motivation
Neural models may confidently output invalid spans (e.g., random 16-digit numbers treated as credit card numbers). To avoid such errors, logical validation was added after model prediction.

### 4.2 Structural Validation Rules
Applied only for high-risk PII:

| Entity | Validation Strategy |
|--------|---------------------|
| CREDIT_CARD | Luhn checksum + digit length |
| PHONE | digit length + delimiter heuristic |
| EMAIL | strict email regex |
| PERSON_NAME | alphabetic ratio + reasonable length constraints |

These functions were implemented in `predict.py`.

### 4.3 Impact
- Removes unrealistic or hallucinated PII predictions
- Preserves only valid, structurally sound entities
- Boosts precision without retraining

---

## 5. Entity Span Integrity

BIO-to-character span reconstruction remains unchanged to retain evaluation consistency. Validated spans are generated using `bio_to_spans()` before filtering. This maintains compatibility with span-based F1 scoring.

---

## 6. Overall Impact

### Before Modifications
- Over-predicted entity boundaries
- Included structurally invalid PII
- Higher recall, lower precision

### After Modifications
- Stable training behavior
- Reduced false positives due to weighted loss
- Post-processing eliminated invalid predictions
- Achieved precision-oriented behavior essential for compliance

---

## 7. Conclusion

The redesigned pipeline shifts from a standard NER setup to a **compliance-conscious PII identification system**. Through combined improvements in:

- **Model architecture**
- **Training strategy**
- **Loss design**
- **Post-processing validation**

the system becomes substantially safer and more reliable for real-world deployment in auditing, masking, and regulatory enforcement pipelines.

This approach demonstrates that **precision is not just a metric preference but a compliance requirement** in PII tasks.

---