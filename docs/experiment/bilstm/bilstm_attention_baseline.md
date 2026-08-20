Yes. At this point, it is useful to freeze the current state as a **Baseline Experiment / V1** before modifying the training pipeline. That gives you a reproducible record of what was done, what was achieved, and why the next experiment is necessary.

Below is a documentation draft you can put under something like:

```text
docs/
└── experiments/
    └── bilstm_attention_baseline.md
```

It documents the **actual current implementation and results**, including the confusion-matrix findings. It does not claim that the model is successful where the evaluation shows otherwise.

---

# BiLSTM + Temporal Attention — Baseline Training and Evaluation

## 1. Experiment Overview

This experiment establishes the first complete end-to-end baseline for the agricultural multi-hazard forecasting framework using a **BiLSTM with Temporal Attention**.

The purpose of this experiment was to verify that the complete pipeline could successfully perform:

* Multi-branch feature processing
* Adaptive cross-attention fusion
* Temporal sequence modeling
* Multi-task regression
* Drought severity classification
* Heat-stress classification
* Chronological train/validation/test splitting
* Training-only feature scaling
* Model checkpointing
* Held-out test evaluation
* Confusion-matrix analysis

This experiment was conducted **before introducing class-imbalance mitigation and other improvements**.

Therefore, the results reported here should be considered the **baseline performance (V1)** rather than the final model performance.

---

# 2. Dataset

The experiment uses the processed dataset:

```text
data/processed/final_dataset.csv
```

The dataset contains:

```text
Rows    : 29,640
Columns : 18
```

The final feature configuration used in this experiment was:

### Meteorological features

```text
7 features
```

### Vegetation features

```text
2 features
```

### Engineered features

```text
5 features
```

The engineered feature set is:

```text
VPD
Humidex
SPI3
doy_sin
doy_cos
```

The seasonal encoding features were verified in the final dataset:

```text
doy_sin : present
doy_cos : present
```

Both features contain no missing values.

This was an important correction from the earlier dataset configuration because the final model expects **5 engineered features**, not 3.

---

# 3. Seasonal Encoding

The dataset includes cyclical day-of-year encoding:

[
doy_{sin}=\sin\left(\frac{2\pi \cdot doy}{365}\right)
]

[
doy_{cos}=\cos\left(\frac{2\pi \cdot doy}{365}\right)
]

These features allow the model to represent the annual seasonal cycle without treating December 31 and January 1 as distant points.

Example values from the final dataset:

| Date       |  doy_sin |   doy_cos |
| ---------- | -------: | --------: |
| 2010-04-01 | 0.999986 |  0.005376 |
| 2010-04-02 | 0.999930 | -0.011826 |
| 2010-04-03 | 0.999579 | -0.029025 |
| 2010-04-04 | 0.998932 | -0.046215 |
| 2010-04-05 | 0.997989 | -0.063391 |

The seasonal encoding was therefore successfully integrated into the final feature pipeline.

---

# 4. Sequence Construction

The model does not operate on individual daily observations.

Instead, sliding temporal windows are constructed.

### Configuration

```text
Window  : 30 days
Horizon : 7 days
```

Therefore, each training example contains:

```text
Previous 30 days
        ↓
Model
        ↓
Forecast target 7 days ahead
```

The resulting sequence dataset contains:

```text
29,460 sequences
```

with the following shapes:

```text
Meteorological:
(29460, 30, 7)

Vegetation:
(29460, 30, 2)

Engineered:
(29460, 30, 5)

Regression targets:
(29460, 2)

Classification targets:
(29460, 2)
```

---

# 5. Prediction Tasks

The model performs four prediction tasks simultaneously.

## 5.1 Regression

Two continuous variables are predicted:

```text
SPI3
Humidex
```

The regression output therefore has shape:

```text
(B, 2)
```

---

## 5.2 Drought Classification

Drought severity is represented using four classes:

| Class | Meaning         |
| ----: | --------------- |
|     0 | Normal/Wet      |
|     1 | Moderate        |
|     2 | Severe          |
|     3 | Extreme drought |

---

## 5.3 Heat Classification

Heat stress is represented using four classes:

| Class | Meaning                   |
| ----: | ------------------------- |
|     0 | No significant discomfort |
|     1 | Some discomfort           |
|     2 | Great discomfort          |
|     3 | Dangerous                 |

The classification output is therefore:

```text
Drought : 4 classes
Heat    : 4 classes
```

---

# 6. Chronological Data Split

A chronological split was used rather than random splitting.

This is necessary for a forecasting problem because random splitting could allow future observations to influence the training process.

The split was:

```text
Training      : 2010–2022
Validation    : 2023–2024
Testing       : 2025–2026
```

The resulting sample counts were:

| Split      | Period    |    Samples |
| ---------- | --------- | ---------: |
| Training   | 2010–2022 |     23,110 |
| Validation | 2023–2024 |      3,655 |
| Testing    | 2025–2026 |      2,695 |
| **Total**  |           | **29,460** |

The **2025–2026 test set remained held out** during training.

---

# 7. Feature Scaling

Feature scaling was performed separately for the different input branches.

The important methodological decision was:

> **Scalers were fitted using training data only.**

The training data was used to fit the StandardScaler objects.

The same fitted scalers were then applied to:

```text
Training
Validation
Testing
```

This prevents information from the validation or test period from leaking into the preprocessing stage.

The scalers are stored in:

```text
data/processed/scalers/
```

---

# 8. Model Architecture

The baseline model is:

**BiLSTM + Temporal Attention**

The complete pipeline is:

```text
Meteorological Input
        │
        ▼
Meteorological Branch Encoder
        │
        │
Vegetation Input
        │
        ▼
Vegetation Branch Encoder
        │
        │
Engineered Input
        │
        ▼
Engineered Branch Encoder
        │
        ▼
Adaptive Cross-Attention Fusion
        │
        ▼
Dynamic Fusion Gating
        │
        ▼
BiLSTM
        │
        ▼
Temporal Attention
        │
        ▼
Shared Temporal Representation
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Regression Heads   Classification Heads
        │               │
        │          ┌────┴────┐
        │          ▼         ▼
        │       Drought     Heat
        │       4-class    4-class
        │
        ├── SPI3
        └── Humidex
```

---

# 9. Adaptive Multi-Branch Fusion

The three feature groups are processed separately:

```text
Meteorological
Vegetation
Engineered
```

They are subsequently combined through the adaptive fusion block.

The fusion stage produces:

```text
Fused representation
Cross-attention information
Fusion gates
```

This allows the architecture to preserve the distinction between different feature modalities before producing a unified representation for temporal forecasting.

---

# 10. Temporal Modeling

The fused representation is passed to:

```text
BiLSTM + Temporal Attention
```

The BiLSTM processes the 30-day sequence in both temporal directions.

The temporal attention mechanism then produces a weighted temporal representation.

Conceptually:

```text
30-day sequence
      │
      ▼
    BiLSTM
      │
      ▼
Temporal Attention
      │
      ▼
Pooled Representation
```

The pooled representation is passed to the multi-task prediction heads.

---

# 11. Model Configuration

The baseline configuration was:

| Parameter               |             Value |
| ----------------------- | ----------------: |
| Window                  |           30 days |
| Forecast horizon        |            7 days |
| `d_model`               |               128 |
| `d_k`                   |                32 |
| Attention heads         |                 4 |
| Gate hidden dimension   |                64 |
| BiLSTM hidden dimension |               128 |
| BiLSTM layers           |                 1 |
| Batch size              |                64 |
| Maximum epochs          |               100 |
| Initial learning rate   |             0.001 |
| Weight decay            |            0.0001 |
| Gradient clipping       |               1.0 |
| Early stopping patience |                10 |
| LR scheduler            | ReduceLROnPlateau |
| LR reduction factor     |               0.5 |
| Minimum LR              |              1e-6 |

The model contained:

```text
710,607 trainable parameters
```

---

# 12. Multi-Task Learning Objective

The baseline model uses a combined loss:

[
L_{total}
=========

L_{reg}
+
L_{drought}
+
L_{heat}
]

where:

* (L_{reg}) = regression MSE
* (L_{drought}) = drought cross-entropy
* (L_{heat}) = heat cross-entropy

The regression component predicts:

```text
SPI3
Humidex
```

while the classification components predict:

```text
Drought severity
Heat severity
```

At this baseline stage, standard unweighted Cross Entropy was used for both classification tasks.

This becomes important later because the evaluation revealed substantial class imbalance.

---

# 13. Training Procedure

Training was performed using:

```text
Device: CUDA
```

Random seeds were fixed for reproducibility.

The training procedure included:

1. Dataset loading
2. Sequence generation
3. Chronological splitting
4. Training-only scaler fitting
5. Scaling
6. DataLoader construction
7. Model initialization
8. Multi-task loss calculation
9. AdamW optimization
10. Gradient clipping
11. Validation after each epoch
12. ReduceLROnPlateau scheduling
13. Best-checkpoint saving
14. Early stopping

---

# 14. Training Progress

The training initially showed a substantial reduction in training loss.

The important epochs were:

| Epoch | Train Loss | Validation Loss | Val Regression | Val Drought |   Val Heat |
| ----: | ---------: | --------------: | -------------: | ----------: | ---------: |
|     1 |    37.6220 |          5.0822 |         4.0273 |      0.5726 |     0.4824 |
|     2 |     4.9657 |          4.6490 |         3.6539 |      0.5594 |     0.4358 |
|     3 |     4.4720 |          4.9923 |         4.0127 |      0.5412 |     0.4383 |
|     4 |     4.3404 |          5.5878 |         4.6034 |      0.4673 |     0.5172 |
|     5 |     4.1355 |          4.2471 |         3.2914 |      0.5219 |     0.4337 |
| **6** | **4.0238** |      **4.1582** |     **3.2071** |  **0.5347** | **0.4165** |
|     7 |     3.8312 |          4.1746 |         3.2181 |      0.5795 |     0.3770 |
|     8 |     3.5980 |          4.4951 |         3.5447 |      0.5336 |     0.4169 |
|     9 |     3.4282 |          6.1138 |         5.1019 |      0.6159 |     0.3961 |
|    10 |     3.3437 |          4.7216 |         3.7048 |      0.6425 |     0.3744 |
|    11 |     2.8853 |          5.3139 |         4.3290 |      0.6091 |     0.3758 |
|    12 |     2.7144 |          4.7641 |         3.7209 |      0.6653 |     0.3779 |
|    13 |     2.6094 |          4.8088 |         3.6953 |      0.6898 |     0.4238 |
|    14 |     2.4656 |          4.9653 |         3.7727 |      0.7972 |     0.3954 |
|    15 |     2.2496 |          4.9284 |         3.7914 |      0.7299 |     0.4070 |
|    16 |     2.1442 |          5.1234 |         3.9566 |      0.7696 |     0.3971 |

The best validation loss occurred at:

```text
Best epoch : 6
Best val loss : 4.158201
```

Training was stopped at epoch 16 because the validation loss failed to improve for the configured patience period.

---

# 15. Checkpoint

The best model was saved as:

```text
checkpoints/bilstm_attention_best.pt
```

The checkpoint corresponds to:

```text
Epoch      : 6
Val loss   : 4.158201
```

The checkpoint stores the model state and optimizer state.

The training history was saved as:

```text
checkpoints/bilstm_attention_history.csv
```

---

# 16. Test Evaluation

After training, the best validation checkpoint was restored.

The model was then evaluated on the previously unseen:

```text
2025–2026
```

test period.

Test samples:

```text
2,695
```

No test samples were used for model selection.

---

# 17. Regression Results

## 17.1 SPI3

| Metric |       Result |
| ------ | -----------: |
| RMSE   | **0.501239** |
| MAE    | **0.374409** |
| R²     | **0.674520** |
| MAPE   |       73.14% |

The SPI3 model achieved an (R^2) of approximately **0.67**, indicating that the model explains a substantial portion of the observed SPI3 variability.

The relatively high MAPE should be interpreted cautiously because percentage error metrics can become unstable when the true value is near zero.

---

## 17.2 Humidex

| Metric |       Result |
| ------ | -----------: |
| RMSE   | **2.503228** |
| MAE    | **1.930334** |
| R²     | **0.931368** |
| MAPE   |    **6.56%** |

Humidex prediction performed substantially better than SPI3 prediction.

The (R^2) of approximately **0.93** indicates a strong relationship between predicted and observed Humidex values.

---

# 18. Classification Results

## 18.1 Drought Classification

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **93.14%** |
| Macro Precision | **23.30%** |
| Macro Recall    | **24.98%** |
| Macro F1        | **24.11%** |

At first glance, the 93.14% accuracy appears strong.

However, the confusion matrix demonstrates that this accuracy is misleading.

### Drought confusion-matrix interpretation

The test set contains:

```text
Normal/Wet : 2,512
Moderate   : 121
Severe     : 62
Extreme    : 0
```

The model correctly identified:

```text
Normal/Wet : 2,510
Moderate   : 0
Severe     : 0
Extreme    : 0
```

Therefore, almost all predictions were concentrated on:

```text
Normal/Wet
```

The model failed to correctly identify:

```text
Moderate drought
Severe drought
```

and there were no Extreme drought examples in this test period.

### Interpretation

This demonstrates a strong **class-imbalance problem**.

The model can obtain high overall accuracy simply by predicting the dominant Normal/Wet class.

Therefore:

> **Drought accuracy alone is not an appropriate indicator of model quality for this task.**

Macro-F1 and per-class recall provide a much more informative assessment.

The baseline drought Macro-F1 of:

[
0.2411
]

indicates poor minority-class detection.

---

# 19. Heat Classification

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **84.94%** |
| Macro Precision | **63.23%** |
| Macro Recall    | **62.35%** |
| Macro F1        | **62.22%** |

The heat classifier performs considerably better than the drought classifier.

However, the confusion matrix reveals an important weakness.

---

# 20. Heat Confusion-Matrix Analysis

The main results were:

```text
No significant discomfort
Correct : 923

Some discomfort
Correct : 407

Great discomfort
Correct : 959

Dangerous
Correct : 0
```

The most important observation is:

```text
Actual Dangerous : 25
Correctly detected: 0
```

All 25 Dangerous cases were classified as:

```text
Great discomfort
```

Therefore, the model has learned the lower and middle heat-stress categories reasonably well but fails to distinguish the most severe category.

---

# 21. Baseline Findings

The experiment demonstrates that the complete forecasting pipeline is operational.

### Successfully demonstrated

* Final dataset construction
* `doy_sin` / `doy_cos` integration
* Multi-branch input processing
* 30-day temporal sequences
* 7-day forecasting horizon
* Chronological splitting
* Training-only scaling
* Adaptive feature fusion
* BiLSTM temporal modeling
* Temporal attention
* Multi-task regression
* Multi-task classification
* CUDA training
* Checkpointing
* Early stopping
* Held-out test evaluation
* Regression metrics
* Classification metrics
* Confusion matrices

Therefore, the baseline architecture is technically functional.

---

# 22. Main Problems Identified

The baseline experiment revealed two major classification problems.

## Problem 1 — Severe drought class imbalance

The drought classifier overwhelmingly predicts:

```text
Normal/Wet
```

This results in:

```text
Accuracy     : 93.14%
Macro F1     : 24.11%
Macro Recall : 24.98%
```

The large gap between accuracy and Macro-F1 is evidence that the accuracy is dominated by the majority class.

---

## Problem 2 — Failure to detect extreme heat

The heat classifier performs better overall:

```text
Accuracy : 84.94%
Macro F1 : 62.22%
```

but fails completely on:

```text
Dangerous
```

with:

```text
25 actual cases
0 correctly detected
```

This means that the model's current decision boundary does not adequately separate the most severe heat category.

---

# 23. Why This Baseline Is Important

The purpose of this experiment was **not to claim final performance**.

Instead, it establishes a measurable reference point.

Future improvements can now be evaluated against:

```text
BASELINE — BiLSTM + Temporal Attention

SPI3
RMSE       0.501239
MAE        0.374409
R²         0.674520

Humidex
RMSE       2.503228
MAE        1.930334
R²         0.931368

Drought
Accuracy   0.9314
Macro-F1   0.2411

Heat
Accuracy   0.8494
Macro-F1   0.6222
```

This prevents improvements from being judged subjectively.

---

# 24. Hazard Probability vs. Hazard Contribution

An important distinction was identified during development.

The model produces class probabilities through softmax:

[
P(c_i)=softmax(z_i)
]

For drought:

[
P(\text{Drought})
=================

P(\text{Moderate})
+
P(\text{Severe})
+
P(\text{Extreme})
]

Thus, the system can provide an output such as:

> **There is a 65% probability of drought.**

This is a probability derived from the four drought classes.

It is different from the relative hazard contribution.

---

## Relative Hazard Contribution

The existing hazard-percentage mechanism calculates an expected severity:

[
S_D =
\frac{
0P_0+1P_1+2P_2+3P_3
}{3}
]

and similarly:

[
S_H =
\frac{
0P_0+1P_1+2P_2+3P_3
}{3}
]

Then:

[
Drought\ Contribution =
\frac{S_D}{S_D+S_H}\times100
]

and:

[
Heat\ Contribution =
\frac{S_H}{S_D+S_H}\times100
]

Therefore:

```text
Drought probability
```

and

```text
Drought relative contribution
```

are **not the same quantity**.

Both can be useful, but they must not be presented as interchangeable.

---

# 25. Current Evaluation Artifacts

The evaluation pipeline generated:

```text
outputs/
└── Results/
    └── bilstm_attention/
        ├── evalution_bilstm.py
        ├── metrics.json
        ├── predictions_2025_2026.csv
        ├── drought_confusion_matrix.png
        └── heat_confusion_matrix.png
```

These artifacts provide the basis for reproducibility and later comparison.

> Note: the filename `evalution_bilstm.py` is retained here exactly as it currently exists in the project.

---

# 26. Baseline Conclusion

The BiLSTM + Temporal Attention baseline successfully demonstrates the feasibility of the proposed multi-task forecasting architecture.

The regression tasks show promising performance, particularly for Humidex:

[
R^2_{Humidex}=0.9314
]

while SPI3 achieves:

[
R^2_{SPI3}=0.6745
]

However, the classification results reveal substantial limitations.

The drought classifier achieves high overall accuracy but has poor minority-class recognition:

[
Macro\ F1=0.2411
]

indicating that the model predominantly predicts the Normal/Wet class.

Similarly, the heat classifier achieves reasonable overall performance but completely fails to identify the Dangerous category in the test period.

Consequently, the baseline should **not yet be considered the final forecasting model**.

The results instead establish a clear direction for the next experimental stage:

> **Improve minority-class learning while preserving the regression performance and chronological evaluation protocol.**

---

# 27. Next Experimental Stage

The next version should investigate:

### Classification imbalance

* Training-set-derived class weights
* Weighted Cross Entropy
* Per-class metrics
* Minority-class recall
* Minority-class F1

### Multi-task loss balancing

Investigate whether:

[
L_{total}
=========

L_{reg}
+
\lambda_D L_D
+
\lambda_H L_H
]

with configurable (\lambda_D) and (\lambda_H) improves classification without degrading regression.

### Probability outputs

Add explicit:

```text
P(Drought)
P(Heat)
```

in addition to:

```text
Predicted drought severity
Predicted heat severity
Relative drought/heat severity contribution
```

### Evaluation

Continue using:

```text
2010–2022 → Training
2023–2024 → Validation
2025–2026 → Final Test
```

The final test set should remain untouched during model development.

---

## Baseline status

```text
Experiment : BiLSTM + Temporal Attention
Version    : Baseline V1
Status     : Completed
Training   : Completed
Evaluation : Completed
Test set   : 2025–2026
```

**Baseline result:** the full pipeline works, regression is promising, but classification—especially drought and extreme heat—requires improvement before this model can be treated as the final system.
