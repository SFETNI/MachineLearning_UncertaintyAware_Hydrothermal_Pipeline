# Source Modules

## Overview

This directory contains the core Python modules for machine learning model training, prediction, and uncertainty quantification.

## Modules

### `rf_trainers.py`

Main module for Random Forest training and prediction.

**Key Functions:**

- `train_rf_models()` - Train RF models with cross-validation
- `conformal_predict()` - Generate prediction intervals using conformal prediction
- `load_best_params()` - Load saved hyperparameters
- `save_model()` - Persist trained models
- `load_model()` - Load saved models

**Example:**
```python
from rf_trainers import train_rf_models, conformal_predict

# Train models
results = train_rf_models(
    X_train, Y_train, X_test, Y_test,
    use_best_params=True,
    cv_folds=5
)

# Get predictions with uncertainty
y_pred_mean, y_pred_lo, y_pred_hi = conformal_predict(
    results['model'], X_cal, y_cal, X_test, alpha=0.10
)
```


## Configuration

### Default Random Forest Parameters

```python
RF_DEFAULT = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}
```

### Allowed Hyperparameter Ranges

```python
RF_ALLOWED = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

## Extending the Code

### Adding New Models

To add a new model type:

1. Create a new trainer function:
```python
def train_new_model(X_train, y_train, **kwargs):
    model = NewModel(**kwargs)
    model.fit(X_train, y_train)
    return model
```

2. Add to model registry:
```python
MODEL_REGISTRY = {
    'rf': train_rf_models,
    'new_model': train_new_model
}
```

### Custom Uncertainty Quantification

To implement custom UQ methods:

```python
def bootstrap_predict(model, X, n_bootstrap=100):
    """Bootstrap confidence intervals"""
    predictions = []
    for i in range(n_bootstrap):
        # Bootstrap sampling and prediction
        ...
    return np.mean(predictions, axis=0), np.percentile(predictions, [5, 95], axis=0)
```

## Testing

To run unit tests (if available):
```bash
pytest tests/
```

## Dependencies

Core dependencies:
- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- joblib >= 1.0.0

Optional dependencies:
- xgboost (for XGBoost models)
- ngboost (for NGBoost models)

## Contributing

When adding new functionality:
1. Follow existing code style
2. Add docstrings to all functions
3. Include type hints where possible
4. Update this README
5. Add unit tests

## Performance Considerations

- Use `n_jobs=-1` to parallelize across CPU cores
- For large datasets, consider subsampling or incremental learning
- Monitor memory usage with `memory_profiler`
- Profile code with `cProfile` to identify bottlenecks
