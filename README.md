# ML-UQ Hydrothermal Pipeline

## Machine Learning with Uncertainty Quantification for Hydrothermal Processing

This repository provides a complete pipeline for training machine learning models with uncertainty quantification on hydrothermal processing datasets. It includes Random Forest models with conformal prediction for biochar, bio-oil, and hydrochar characterization.

## Features

- **Data Processing**: Feature engineering with interactions, hot encoding, and normalization
- **ML Models**: Random Forest regression with cross-validation
- **Uncertainty Quantification**: Conformal prediction intervals
- **Model Persistence**: Save/load trained models and best parameters
- **Visualization**: Parity plots, feature importance, and performance metrics

## Repository Structure

```
ml_uq_hydrothermal_pipeline/
├── notebooks/              # Jupyter notebooks for analysis
│   └── hydrothermal_ml_pipeline.ipynb
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── rf_trainers.py      # Random Forest training functions
│   ├── rf_trainers_GS.py   # Grid search utilities
│   └── rf_trainers_save.py # Model saving/loading
├── data/                   # Input datasets (place your CSV files here)
│   └── README.md           # Instructions for data placement
├── models/                 # Trained model artifacts
│   └── README.md
├── outputs/                # Results and visualizations
│   ├── conformal/          # Conformal prediction outputs
│   ├── figures/            # Generated plots
│   └── tables/             # CSV results
├── docs/                   # Additional documentation
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
├── .gitignore              # Git ignore patterns
└── LICENSE                 # MIT License

```

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ml_uq_hydrothermal_pipeline
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

## Quick Start

### 1. Prepare your data

Place your hydrothermal processing dataset in the `data/` directory:
- Expected format: CSV file with feedstock composition, process conditions, and product yields
- See `data/README.md` for detailed data format requirements

### 2. Run the notebook

```bash
jupyter notebook notebooks/hydrothermal_ml_pipeline.ipynb
```

### 3. Train models

The notebook walks through:
- Data loading and exploration
- Feature engineering
- Train/test splitting
- Model training with cross-validation
- Uncertainty quantification
- Results visualization

## Key Configuration

Edit these parameters in the notebook:

```python
# Model settings
USE_INTERACTIONS = True      # Enable feature interactions
USE_TIER = True             # Use hierarchical features
ALPHA = 0.10                # Significance level (90% confidence)

# Train/test split
TEST_FRACTION = 0.2         # 20% test set
SPLIT_MODE = "random"       # or "by_doi", "by_feedstock"

# Paths (auto-configured for repository structure)
DATA_DIR = "../data/"
MODELS_DIR = "../models/"
OUTPUTS_DIR = "../outputs/"
```

## Model Training

The pipeline supports multiple training modes:

1. **Baseline RF**: Simple Random Forest
2. **With Best Params**: Load hyperparameters from previous grid search
3. **With Grid Search**: Full hyperparameter optimization

```python
from src.rf_trainers import train_rf_models

results = train_rf_models(
    X_train, Y_train, X_test, Y_test,
    use_best_params=True,
    do_grid_search=False
)
```

## Uncertainty Quantification

Conformal prediction provides distribution-free prediction intervals:

```python
from src.rf_trainers import conformal_predict

y_pred_mean, y_pred_lo, y_pred_hi = conformal_predict(
    rf_model, X_cal, y_cal, X_test, alpha=0.10
)
```

## Output Files

After running the pipeline, you'll find:

- `outputs/tables/`: Performance metrics (R², RMSE), train/test splits
- `outputs/figures/`: Parity plots, feature importance charts
- `outputs/conformal/`: Conformal prediction intervals
- `models/`: Saved model files (.pkl)

## Requirements

- Python 3.8+
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- joblib >= 1.0.0

See `requirements.txt` for complete list.

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```

## Related Work

This pipeline is inspired by:
- Chen et al. (2021) - Machine learning for hydrothermal processing
- [Add your references]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- [Funding sources]
- [Collaborators]
- [Data providers]
# MachineLearning_UncertaintyAware_Hydrothermal_Pipeline
