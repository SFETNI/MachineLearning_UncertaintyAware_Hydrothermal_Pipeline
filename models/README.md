# Models Directory

## Overview

This directory stores trained machine learning models and their associated metadata.

## Contents

After training, you'll find:

### Model Files
- `*.pkl` - Serialized scikit-learn models (Random Forest, etc.)
- `*_best_params.json` - Optimal hyperparameters from grid search
- `*_feature_importance.csv` - Feature importance rankings

### Subdirectories

- `random_forest/` - Random Forest models
- `best_params/` - Hyperparameter configurations
- `checkpoints/` - Training checkpoints

## Loading Saved Models

```python
import joblib
from pathlib import Path

# Load a trained model
model_path = Path("../models/rf_biooil_Y_daf.pkl")
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_new)
```

## Model Naming Convention

Models are saved with descriptive names:
- `rf_{target}_{timestamp}.pkl` - Random Forest for specific target
- `best_params_{mode}.json` - Best hyperparameters for training mode

Example: `rf_biochar_Y_daf_20260120.pkl`

## Storage Considerations

- Model files can be large (10-100 MB for Random Forests)
- Add `*.pkl` to `.gitignore` if models are too large for version control
- Consider using Git LFS for large model files
- Alternatively, share models via cloud storage (Google Drive, Zenodo, etc.)

## Reproducibility

To ensure reproducibility:
1. Save random seeds used during training
2. Store data preprocessing parameters
3. Document scikit-learn version and hyperparameters
4. Include training configuration in metadata

The pipeline automatically saves:
- Model hyperparameters
- Training/test split indices
- Feature names and ordering
- Preprocessing transformations
