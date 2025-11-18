# ChromOrtho

Predicting chromatographic separability for orthogonal resin pair selection in ion-exchange chromatography

---

## Status: Prototype / Research Stage

**⚠️ This model is currently under development and NOT ready for deployment or predictive use.**

While the architecture is functional and can be trained, the model does not yet achieve sufficient predictive performance for practical applications. This repository serves as a research prototype and documentation of the approach.

---

## Overview

ChromOrtho is a machine learning framework designed to predict separability outcomes when using two-step orthogonal resin combinations in ion-exchange chromatography. The goal is to reduce experimental trial-and-error in resin screening by forecasting which resin pairs will yield the greatest selectivity for protein separations.

### Intended Capabilities (Under Development)

- **Input**: Cheminformatics descriptors for two resins (Resin A and Resin B)
- **Output**: Predicted separability score indicating how well the resin pair can separate target proteins
- **Application**: Guide selection of orthogonal resin combinations for two-dimensional chromatography

### Current Limitations

- Model performance is not yet sufficient for reliable predictions
- Limited training data availability
- Requires further feature engineering and architecture optimization
- Validation metrics do not meet deployment thresholds

---

## Model Architecture

### Siamese Neural Network Approach

The model uses a **Siamese Multi-Layer Perceptron (MLP)** architecture designed to learn relationships between paired resin descriptors:

**Architecture Components:**
1. **Shared Encoder**: Two identical neural networks (shared weights) that independently encode descriptors for Resin A and Resin B
2. **Embedding Layer**: Maps each resin's descriptor vector into a lower-dimensional embedding space
3. **Interaction Features**: Combines the embeddings using:
   - Absolute difference: `|A - B|`
   - Element-wise product: `A × B`
   - Element-wise sum: `A + B`
4. **Regression Head**: Fully connected layers that predict the separability score from the combined features

**Key Design Principles:**
- **Weight sharing** ensures the model learns a consistent representation for both resins
- **Interaction features** capture relative differences and synergies between resin properties
- **End-to-end training** learns optimal feature representations directly from raw descriptors

### Model Hyperparameters

- **Embedding dimension**: 128 (default)
- **Hidden layer size**: 256 (default)
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Learning rate**: 1e-3 (default)
- **Early stopping**: Patience of 15 epochs based on validation RMSE

---

## Data Requirements

### Input Format

The model expects a CSV file with paired descriptor columns:

**Column Naming Convention:**
- Features for Resin A: `<descriptor_name>_A`
- Features for Resin B**: `<descriptor_name>_B`
- Target variable: Numeric separability score

**Example Columns:**
```
pH_A, pH_B
logP_A, logP_B
Hydrophobicity_A, Hydrophobicity_B
Charge_Density_A, Charge_Density_B
Separability  # Target column
```

**Automatic Handling:**
- Only complete pairs (both `_A` and `_B` present) are used
- Columns containing `Ion_A` or `Ion_B` are automatically excluded
- Zero-variance features are removed during training

### Data Preprocessing

1. **Pairing Detection**: Automatically identifies matching `_A` and `_B` columns
2. **Data Cleaning**: Removes rows with non-finite values in features or target
3. **Variance Filtering**: Drops features with zero variance in training set
4. **Standardization**: StandardScaler fitted on training data (A and B stacked together)
5. **Train/Validation Split**: Random split with configurable fraction (default 20% validation)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- pandas
- scikit-learn
- numpy
- joblib

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch pandas scikit-learn numpy joblib
```

**Note**: For GPU support, install PyTorch with CUDA following instructions at [pytorch.org](https://pytorch.org)

---

## Usage

### Training a Model

Basic command (PowerShell):

```powershell
python .\siamese_from_single_csv.py `
  --csv "C:\path\to\your_data.csv" `
  --target Separability `
  --outdir "C:\path\to\outdir" `
  --epochs 120 `
  --batch_size 128 `
  --lr 1e-3 `
  --val_frac 0.2
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | - | Path to CSV with paired features |
| `--target` | Yes | - | Name of target column (case-insensitive) |
| `--outdir` | Yes | - | Output directory for artifacts |
| `--epochs` | No | 120 | Maximum training epochs |
| `--batch_size` | No | 128 | Training batch size |
| `--lr` | No | 1e-3 | Learning rate |
| `--patience` | No | 15 | Early stopping patience (epochs) |
| `--val_frac` | No | 0.2 | Validation set fraction |
| `--emb_dim` | No | 128 | Embedding dimension |
| `--hidden` | No | 256 | Hidden layer size |
| `--seed` | No | 42 | Random seed |

### Training Output

During training, the script prints progress:

```
Epoch 001 | train_loss 0.1234 | val_RMSE 0.5678
[info] New best val RMSE: 0.5678
```

---

## Output Files

All outputs are saved to the specified `--outdir`:

### Model Checkpoints

- **`model.pt`**: Best model checkpoint (lowest validation RMSE)
- **`model_final.pt`**: Final model after all epochs
- **`scaler.joblib`**: Fitted StandardScaler (best checkpoint)
- **`scaler_final.joblib`**: Fitted StandardScaler (final checkpoint)

### Checkpoint Contents

Each `.pt` file contains:
- `state_dict`: Model weights
- `in_dim`: Number of paired base features
- `base_features`: List of base descriptor names
- `mapping`: Dictionary mapping base names to (colA, colB) pairs
- `target_col`: Target column name
- `meta`: Embedding and hidden dimensions

### Predictions and Metadata

- **`val_predictions.csv`**: Validation set actual vs predicted values
- **`run_summary.json`**: Training metadata including:
  - Data paths and target column
  - Training/validation set sizes
  - Hyperparameters
  - Best validation RMSE
  - Feature information

---

## Current Research Directions

### Areas for Improvement

1. **Data Augmentation**
   - Expand training dataset with additional resin pair experiments
   - Synthetic data generation strategies
   - Transfer learning from related chromatography tasks

2. **Feature Engineering**
   - Incorporate domain-specific resin descriptors
   - Interaction terms beyond current set
   - Dimensionality reduction techniques

3. **Architecture Optimization**
   - Hyperparameter tuning (embedding size, hidden layers, dropout)
   - Alternative architectures (attention mechanisms, graph neural networks)
   - Ensemble methods

4. **Regularization**
   - Dropout layers
   - L1/L2 weight penalties
   - Data-specific augmentation

5. **Evaluation Metrics**
   - Cross-validation strategies
   - Applicability domain assessment
   - Uncertainty quantification

---

## Known Issues

- **Insufficient predictive performance**: Current R² and RMSE do not meet deployment standards
- **Limited training data**: Small dataset size limits model generalization
- **Overfitting risk**: Model may memorize training examples rather than learn generalizable patterns
- **Feature selection**: Optimal descriptor set not yet determined

---

## Troubleshooting

### Common Errors

**"No paired _A/_B features found"**
- Check that your CSV has matching `_A` and `_B` column suffixes
- Ensure column names are exactly paired (case-sensitive base names)

**"--target 'X' not found"**
- Verify the target column exists in your CSV
- Target matching is case-insensitive but name must match

**"Validation doesn't save best checkpoint"**
- Occurs if validation RMSE is not finite (e.g., empty validation set)
- Final checkpoint is still saved

**"GPU not detected"**
- Script automatically uses GPU if `torch.cuda.is_available()` returns True
- For CPU-only systems, training will proceed on CPU (slower)

---

## Future Development

This prototype represents initial exploration of machine learning for orthogonal resin selection. Future work will focus on:

- Collecting larger, more diverse training datasets
- Systematic hyperparameter optimization
- Integration with experimental validation workflows
- Development of confidence intervals and prediction uncertainty
- Web interface for easier model interaction

**Contributions and collaborations are welcome** as we work toward a production-ready model.

---

## Citation

If you use ChromOrtho in your research, please cite:

```
[Citation information to be added upon publication]
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions, suggestions, or collaboration inquiries:
- Open an issue on GitHub
- Contact: [Your contact information]

---

## Acknowledgments

This work builds upon established principles in:
- Siamese neural networks for similarity learning
- Cheminformatics descriptor development
- Chromatography separation science

---

## Related Projects

- **PredElute**: Predictive models for protein elution in CEX (production-ready)
- **ProDes**: Protein descriptor calculation tools
- **RDKit**: Cheminformatics toolkit for descriptor generation
