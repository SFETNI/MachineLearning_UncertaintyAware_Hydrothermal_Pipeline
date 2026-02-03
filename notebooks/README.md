# Notebooks Directory

## Overview

This directory contains Jupyter notebooks for the ML-UQ Hydrothermal Pipeline.

## Main Notebook

### `hydrothermal_ml_pipeline.ipynb`

The primary analysis notebook covering:

1. **Data Loading & Exploration**
   - Load HTT dataset
   - Inspect structure and statistics
   - Check for missing values

2. **Feature Engineering**
   - Create interaction features
   - One-hot encode categorical variables
   - Normalize/standardize features

3. **Train/Test Splitting**
   - Random splitting
   - DOI-based splitting (avoid publication bias)
   - Feedstock-based splitting

4. **Model Training**
   - Random Forest regression
   - Cross-validation
   - Hyperparameter tuning

5. **Uncertainty Quantification**
   - Conformal prediction intervals
   - Coverage analysis
   - Interval width assessment

6. **Visualization & Metrics**
   - Parity plots (predicted vs actual)
   - Feature importance charts
   - Residual analysis
   - Performance metrics (R², RMSE, MAE)

## Running the Notebook

### Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```



### Saving Results

Results are automatically saved to `../outputs/`:
- Figures: `../outputs/figures/`
- Tables: `../outputs/tables/`
- Conformal predictions: `../outputs/conformal/`

### Debugging

If you encounter errors:
1. Check data path: `../data/HTT_normalized_data.csv`
2. Verify src modules are accessible
3. Ensure output directories exist
4. Check Python version (3.8+)



### Modify Model Parameters

```python
USE_INTERACTIONS = True      # Enable feature interactions
USE_TIER = True             # Use hierarchical features
ALPHA = 0.10                # 90% confidence intervals
TEST_FRACTION = 0.2         # 20% test set
```

### Select Targets

```python
targets = [
    "biochar_Y_daf",
    "biooil_Y_daf",
    "C_biochar",
    "HHV_biooil"
]
```

## Additional Notebooks

You can create additional notebooks in this directory for:
- Exploratory data analysis
- Model comparison
- Sensitivity analysis
- Specific case studies

## Exporting Results

To export the notebook without outputs:
```bash
jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True hydrothermal_ml_pipeline.ipynb --output hydrothermal_ml_pipeline_clean.ipynb
```

To export as HTML:
```bash
jupyter nbconvert --to html hydrothermal_ml_pipeline.ipynb
```

To export as PDF (requires LaTeX):
```bash
jupyter nbconvert --to pdf hydrothermal_ml_pipeline.ipynb
```
