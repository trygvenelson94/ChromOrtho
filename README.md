# Siamese Regressor — Training from a Single CSV

This document explains how to train a Siamese MLP regressor using a single CSV file that already contains paired features (`*_A`, `*_B`) and a numeric target column.

Script: `siamese_from_single_csv.py`

---

## What this script does

- Parses paired feature columns that end in `_A` and `_B` and pairs them by a common base name.
- Ignores columns containing `Ion_A` or `Ion_B` (case-insensitive), if present.
- Validates and cleans the data (drops non-finite targets and non-finite feature rows).
- Removes any base features that have zero variance in training data.
- Splits the data into Train/Validation sets (row-wise) and fits a scaler on train only (A+B stacked).
- Trains a Siamese MLP that embeds A and B separately, then merges with `[|A−B|, A*B, A+B]`.
- Saves a best checkpoint (if a finite validation metric is observed) and a final checkpoint (always).
- Writes validation predictions and a small run summary JSON.

---

## Expected CSV format

- Columns named as paired features using `_A` and `_B` suffixes. Examples:
  - `pH_A`, `pH_B`
  - `logP_A`, `logP_B`
  - `Hydrophobicity_A`, `Hydrophobicity_B`
- A numeric target column (case-insensitive match to `--target`). For example `Separability`, `Score`, `Y`, etc.
- Any columns whose names contain `Ion_A` or `Ion_B` will be ignored automatically.

Notes:
- The script detects all bases that appear with both `_A` and `_B` and keeps only complete pairs.
- Rows with non-numeric or missing values for any kept feature or for the target are dropped for robustness.

---

## Installation and environment

Recommended environment (Windows example with venv):

```powershell
# from a PowerShell prompt
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch pandas scikit-learn numpy joblib
```

PyTorch installation may vary depending on your CUDA availability; see the official PyTorch site if you need GPU support.

---

## Usage

Basic training command (PowerShell):

```powershell
python .\siamese_from_single_csv.py `
  --csv "C:\\path\\to\\your_data.csv" `
  --target Separability `
  --outdir "C:\\path\\to\\outdir" `
  --epochs 120 `
  --batch_size 128 `
  --lr 1e-3 `
  --val_frac 0.2
```

Arguments:
- `--csv` (required): Path to your single CSV with paired features.
- `--target` (required): Name of the numeric target column (case-insensitive match).
- `--outdir` (required): Output directory where artifacts will be saved.
- `--epochs` (default: 120): Max training epochs.
- `--batch_size` (default: 128)
- `--lr` (default: 1e-3)
- `--patience` (default: 15): Early stopping patience (epochs) based on validation RMSE.
- `--val_frac` (default: 0.2): Fraction of rows used for validation.
- `--emb_dim` (default: 128): Embedding size within the encoder.
- `--hidden` (default: 256): Hidden size of MLP layers.
- `--seed` (default: 42)

During training the script prints lines like:
```
Epoch 001 | train_loss 0.1234 | val_RMSE 0.5678
[info] New best val RMSE: 0.5678
```

---

## Outputs

Saved to `--outdir`:

- `model.pt` and `scaler.joblib`
  - Best checkpoint and scaler if any finite validation metric was observed.
- `model_final.pt` and `scaler_final.joblib`
  - Final model and scaler (always saved).
- `val_predictions.csv`
  - Validation `actual` and `predicted` values (if validation set is non-empty).
- `run_summary.json`
  - Summary meta-data: data paths, target, training/validation sizes, hyperparameters, best validation RMSE, etc.

The checkpoints contain:
- `state_dict`: model weights
- `in_dim`: number of paired base features
- `base_features`: list of base names
- `mapping`: base -> `(colA, colB)` (exact original column names)
- `target_col`: the resolved target column name
- `meta`: embedding and hidden sizes

These metadata fields are intended for use by downstream prediction scripts that must rebuild inputs in the exact training feature order.

---

## Model architecture (summary)

- Two shared encoders (weights shared) map `A` and `B` vectors (length = number of bases) into embeddings of size `emb_dim`.
- Encoded pair is merged as `[|A−B|, A*B, A+B]` and passed to a regression head.
- Loss: Mean Squared Error (MSE) on the target.
- The StandardScaler is fit on train-only data using the union distribution of A and B (stacked) and applied to both A and B.

---

## Tips and best practices

- Ensure every base feature you want to use appears as both `_A` and `_B` columns.
- Make the target numeric. If it’s stored as strings, convert it upstream.
- If you have very few rows, set a smaller `--val_frac` or consider K-fold CV externally.
- The script removes train-only zero-variance bases automatically; consider investigating such features.
- If a warning appears about dropping rows with non-finite values, inspect and clean your CSV.

---

## Troubleshooting

- "No paired *_A/*_B features found":
  - Check your headers; you must have matching `_A` and `_B` suffixes.
- "--target 'X' not found":
  - The script matches case-insensitively, but the name must exist in your CSV.
- Validation doesn’t save a best checkpoint:
  - This occurs if validation RMSE was not finite (e.g., validation set empty). The script still saves a final checkpoint.
- GPU is not used:
  - The script detects CUDA automatically. If `torch.cuda.is_available()` is false, it runs on CPU.

---

## Minimal example CSV (header sketch)

```
Separability, pH_A, pH_B, logP_A, logP_B, Size_A, Size_B
0.42,         5.0,  7.0,  2.1,    1.8,    300,    280
0.35,         6.0,  5.0,  2.0,    2.0,    310,    310
...
```

Run:
```powershell
python .\siamese_from_single_csv.py `
  --csv .\toy.csv `
  --target Separability `
  --outdir .\out
```

---

## License / attribution

This script is provided as-is for internal research and prototyping. If you publish work that uses this code, please describe the Siamese paired-feature setup and cite your data and training configuration appropriately.
