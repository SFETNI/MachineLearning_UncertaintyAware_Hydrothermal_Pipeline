# Documentation

## Overview

This directory contains additional documentation for the ML-UQ Hydrothermal Pipeline.

## Contents

- **API Reference**: Detailed documentation of functions and classes
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Example workflows and use cases
- **Troubleshooting**: Common issues and solutions

## Key Concepts

### Conformal Prediction

Conformal prediction provides distribution-free prediction intervals with guaranteed coverage. Given a significance level α (e.g., 0.10 for 90% confidence), the method ensures that the true value falls within the predicted interval with probability 1-α.

**Advantages:**
- No distributional assumptions required
- Guaranteed finite-sample coverage
- Adapts to local uncertainty

**Usage:**
```python
from rf_trainers import conformal_predict

y_pred_mean, y_pred_lo, y_pred_hi = conformal_predict(
    model, X_cal, y_cal, X_test, alpha=0.10
)
```

### Feature Engineering

The pipeline supports several feature engineering strategies:

1. **Interactions**: Products of important features (T×IC, T×H/C, etc.)
2. **Categorical Encoding**: One-hot encoding for catalysts, solvents, etc.
3. **Derived Features**: Ratios, normalized values, etc.

### Train/Test Splitting

Three splitting strategies:

1. **Random**: Randomly split rows
2. **By DOI**: Group by publication (avoid data leakage)
3. **By Feedstock**: Group by feedstock type

### Model Persistence

Models are saved using joblib:
- Model object (`.pkl`)
- Best hyperparameters (`.json`)
- Feature names and metadata

## Advanced Topics

### Hyperparameter Tuning

Grid search is performed over:
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples per leaf

### Uncertainty Quantification

Multiple approaches available:
- Conformal prediction (recommended)
- Bootstrap confidence intervals
- Quantile regression forests

