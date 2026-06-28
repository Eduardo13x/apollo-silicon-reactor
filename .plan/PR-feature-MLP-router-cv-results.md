# PR-feature-MLP-router — Phase 1b CV Results

- **Dataset**: `/tmp/apollo_mlp_dataset.csv` — 5413 windows, 16-d features + 4-class label
- **Window size**: 30s (= 60 cycles @ 2Hz daemon cadence)
- **Architecture**: 16 -> 32 -> 4, ReLU + softmax (sklearn MLPClassifier single hidden layer)
- **Hyperparameters**:
  - `hidden_layer_sizes` = `(32,)`
  - `activation` = `relu`
  - `solver` = `adam`
  - `learning_rate_init` = `0.001`
  - `max_iter` = `200`
  - `early_stopping` = `True`
  - `validation_fraction` = `0.1`
  - `random_state` = `42`
  - `n_iter_no_change` = `10`
- **CV**: stratified 5-fold (random_state=42, shuffle=True)
- **Decision rule**: mean CV accuracy >= 0.55 = PROCEED, else ABORT

## Dataset

- Rows: **5413** (target >1000: PASS)
- NaN cells: **0**
- Label distribution: {"2": 2701, "3": 2003, "1": 709}
- Features with non-zero std: **0 / 16**
- **FEATURE CONSTANCY WARNING**: every row is the same 16-d vector. This collapses the MLP to a constant-output classifier; only the bias terms can adapt.

## Per-fold results

| Fold | n_train | n_val | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | n_iter | val_loss |
|---|---|---|---|---|---|---|---|---|
| 1 | 4330 | 1083 | 0.4986 | 0.1247 | 0.2500 | 0.1664 | 12 | 0.4781 |
| 2 | 4330 | 1083 | 0.4986 | 0.1247 | 0.2500 | 0.1664 | 12 | 0.4873 |
| 3 | 4330 | 1083 | 0.4995 | 0.1249 | 0.2500 | 0.1666 | 12 | 0.5312 |
| 4 | 4331 | 1082 | 0.4991 | 0.1248 | 0.2500 | 0.1665 | 12 | 0.4700 |
| 5 | 4331 | 1082 | 0.4991 | 0.1248 | 0.2500 | 0.1665 | 12 | 0.4793 |

## Summary

- **Mean accuracy**: 0.4990 ± 0.0004
- **Min / max accuracy**: 0.4986 / 0.4995
- **Mean F1 (macro)**: 0.1664 ± 0.0001

- **Aggregate confusion matrix** (rows = true, cols = pred, classes 0..3):

```
       pred0  pred1  pred2  pred3
true0      0      0      0      0
true1      0      0    709      0
true2      0      0   2701      0
true3      0      0   2003      0
```

- **Decision**: **ABORT** (threshold 0.55)

## Paper anchors

- [Barto & Sutton 2018, §9] — function approximation: 16-d feature vector mapped to 4-class softmax via a single 32-unit hidden layer (RELU). Offline-trained on the journal's reward signal.
- [Bishop 2006, §5.3] — MLP template: feed-forward, cross-entropy loss (sklearn's `MLPClassifier` uses log-loss by default), softmax output over 4 classes.

---

_Phase 1c appends: artifact serialization + final ablation notes._
