from __future__ import annotations
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except ImportError:
    NGBRegressor = None
    Normal = None


import os
import ast
import json
import joblib
import numpy as np
import pandas as pd
import warnings

from typing import Dict, Iterable, Optional, Tuple

from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin, clone
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.preprocessing._function_transformer",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.compose._target",
)

SAMPLE_WEIGHT_STRATEGY = None

_POLY = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


RMSE_SCORER = make_scorer(rmse, greater_is_better=False)

COLS_Y_DEFAULT = [
    "B_Y", "C_Y", "A_Y", "G_Y",
    "E_B", "E_H", "C_B", "C_H",
    "HHV_biooil", "C_biooil", "O_biooil", "H_biooil", "N_biooil", "S_biooil",
    "HHV_biochar", "C_biochar", "O_biochar", "H_biochar", "N_biochar", "S_biochar",
    "O/C_biooil", "H/C_biooil", "O/C_biochar", "H/C_biochar",
]

RF_DEFAULT = {
    "n_estimators": 800,
    "max_depth": 40,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "bootstrap": True,
    "max_samples": 0.9,
    "random_state": 42,
    "n_jobs": -1,
}

DT_DEFAULT = {
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "splitter": "best",
    "max_features": None,
    "random_state": 42,
}

RF_ALLOWED = {
    "n_estimators", "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
    "min_weight_fraction_leaf", "max_features", "max_leaf_nodes", "min_impurity_decrease",
    "bootstrap", "oob_score", "n_jobs", "random_state", "ccp_alpha", "max_samples", "warm_start",
}
DT_ALLOWED = {
    "criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf",
    "min_weight_fraction_leaf", "max_features", "random_state", "max_leaf_nodes",
    "min_impurity_decrease", "ccp_alpha",
}

ALLOWED_MODEL_MODES = (
    "rf", "rf_cal", "rf_qr",
    "extratrees", "extratrees_cal",
    "xgb", "xgb_cal",
    "ngb",
)

RF_PARAM_DISTS = {
    "n_estimators": [400, 600, 800, 1200],
    "max_depth": [20, 40, 50, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["sqrt", 0.5, 0.7, 0.9],
    "max_samples": [0.7, 0.85, 1.0],
    "bootstrap": [True],
}

RF_GS_N_ITER = 60

RF_PARAM_DISTS = {
    "n_estimators": [400, 800],          
    "max_depth": [20, None],             
    "min_samples_split": [2, 4],         
    "min_samples_leaf": [1, 2],          
    "max_features": ["sqrt", 0.5],       
    "max_samples": [0.8],                
    "bootstrap": [True],              
}


RF_GS_N_ITER = 6


def _rf_grid_search(
    X_t: pd.DataFrame,
    y_t: pd.Series,
    base_params: dict,
    cv,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    param_dists: Optional[dict] = None,
    n_iter: int = RF_GS_N_ITER,
) -> Tuple[dict, float]:
    """
    Randomized search over RF hyperparameters for a single target.

    Returns:
        best_params (merged with base_params),
        best_score  (CV R² of the best setting).
    """
    if param_dists is None:
        param_dists = RF_PARAM_DISTS

    rf = RandomForestRegressor(**base_params)

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dists,
        n_iter=n_iter,
        scoring="r2",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
    )

    if sample_weight is not None:
        search.fit(X_t, y_t, sample_weight=sample_weight)
    else:
        search.fit(X_t, y_t)

    best_params = base_params.copy()
    best_params.update(search.best_params_)
    best_params = _filter_params(best_params, RF_ALLOWED)

    return best_params, float(search.best_score_)


def save_best_params_csv(
    csv_path: str,
    rf_best: Dict[str, dict],
    dt_best: Dict[str, dict],
    targets: Iterable[str],
) -> None:
    """
    Write per-target RF/DT best params to a CSV, JSON-encoded in two columns.
    """
    rows = []
    for t in targets:
        rows.append(
            {
                "Target": t,
                "rf_best_params": json.dumps(rf_best.get(t, {})),
                "dt_best_params": json.dumps(dt_best.get(t, {})),
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def _ensure_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(float)


def _safe_parse_dict(x) -> dict:
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s or s == "{}":
            return {}
        # Try JSON first
        try:
            return json.loads(s)
        except Exception:
            pass
        # Fallback: Python literal
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}



def _filter_params(params: dict, allowed: set) -> dict:
    return {k: v for k, v in params.items() if k in allowed}


def _sanitize_target_name(target: str) -> str:
    return target.replace("/", "_").replace("\\", "_").replace(":", "_")


def load_best_params(csv_path: str) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    if not (csv_path and os.path.exists(csv_path)):
        return {}, {}
    dfp = pd.read_csv(csv_path)
    rf_col = next((c for c in dfp.columns if "rf_best_params" in c.lower()), None)
    dt_col = next((c for c in dfp.columns if "dt_best_params" in c.lower()), None)
    if rf_col is None and dt_col is None:
        return {}, {}
    rf_best, dt_best = {}, {}
    for _, row in dfp.iterrows():
        tgt = str(row.get("Target", "")).strip()
        if not tgt:
            continue
        if rf_col:
            rf_best[tgt] = _safe_parse_dict(row.get(rf_col))
        if dt_col:
            dt_best[tgt] = _safe_parse_dict(row.get(dt_col))
    return rf_best, dt_best


def _rows_for_target(X: pd.DataFrame, Y: pd.DataFrame, target: str):
    y_full = pd.to_numeric(Y[target], errors="coerce")
    mask = y_full.notna()
    X_t = X.loc[mask].copy()
    y_t = y_full.loc[mask].astype(float)
    nan_mask = X_t.isna().any(axis=1)
    if nan_mask.any():
        X_t = X_t.loc[~nan_mask]
        y_t = y_t.loc[X_t.index]
    return X_t, y_t


def _compute_sample_weight(
    X_t: pd.DataFrame,
    y_t: pd.Series,
    strategy: Optional[str] = None,
) -> np.ndarray:
    """
    Compute per-sample weights for imbalance handling.

    Currently implemented:
      - None:             all ones
      - "balance_temperature": inverse-frequency weighting over T bins
    """
    w = np.ones(len(y_t), dtype=float)
    if strategy is None:
        return w

    if strategy == "balance_temperature":
        if "T" in X_t.columns:
            T_col = X_t["T"].copy()
            if T_col.notna().sum() == 0:
                return w
            T_filled = T_col.fillna(T_col.median())
            bins = [0, 250, 300, 350, 1000]
            T_bins = pd.cut(T_filled, bins=bins, include_lowest=True)
            counts = T_bins.value_counts()
            inv = 1.0 / counts
            w = T_bins.map(inv).astype(float).values
    return w


def _build_lr_design_matrix(X_t: pd.DataFrame) -> pd.DataFrame:
    """
    Build a polynomial interaction-only design matrix for LinearRegression.
    This mimics Kaur-style factorial interactions (T × catalyst, etc.).
    """
    X_arr = _POLY.fit_transform(X_t)
    names = _POLY.get_feature_names_out(X_t.columns)
    return pd.DataFrame(X_arr, columns=names, index=X_t.index)

# ---------- Pickling-safe transforms ----------
EPS = 1e-3


def frac_to_logit(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, EPS, 1 - EPS)
    return np.log(y / (1 - y))


def logit_to_frac(z):
    z = np.asarray(z, dtype=float)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, EPS, 1 - EPS)


def pct_to_logit(y):
    y = np.asarray(y, dtype=float)
    p = np.clip(y / 100.0, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logit_to_pct(z):
    z = np.asarray(z, dtype=float)
    p = 1.0 / (1.0 + np.exp(-z))
    return 100.0 * np.clip(p, EPS, 1 - EPS)


def choose_ttr(target_name: str, base_reg):
    t = target_name.lower()
    if t in ["c_b", "c_h"]:
        return TransformedTargetRegressor(
            regressor=base_reg, func=frac_to_logit, inverse_func=logit_to_frac
        )
    if any(
        x in t
        for x in [
            "c_biooil",
            "c_biochar",
            "o_biooil",
            "o_biochar",
            "h_biooil",
            "h_biochar",
            "n_biooil",
            "n_biochar",
            "s_biooil",
            "s_biochar",
        ]
    ):
        return TransformedTargetRegressor(
            regressor=base_reg, func=pct_to_logit, inverse_func=logit_to_pct
        )
    if "hhv" in t:
        return TransformedTargetRegressor(
            regressor=base_reg,
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        )
    return base_reg


class LinearCalibratedRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps any regressor and applies a 1D linear correction on its predictions.
    """

    def __init__(self, base_regressor, fit_intercept: bool = True):
        self.base_regressor = base_regressor
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        self.base_regressor_ = clone(self.base_regressor)
        if sample_weight is not None:
            self.base_regressor_.fit(X, y, sample_weight=sample_weight)
        else:
            self.base_regressor_.fit(X, y)
        y_pred = self.base_regressor_.predict(X).reshape(-1, 1)
        self.calibrator_ = LinearRegression(fit_intercept=self.fit_intercept)
        if sample_weight is not None:
            self.calibrator_.fit(y_pred, y, sample_weight=sample_weight)
        else:
            self.calibrator_.fit(y_pred, y)
        return self

    def predict(self, X):
        y_pred = self.base_regressor_.predict(X).reshape(-1, 1)
        return self.calibrator_.predict(y_pred)


def _build_rf_like_estimator(
    model_mode: str,
    rf_params: dict,
    target_name: str,
):
    """
    Build estimator backend, wrapped in appropriate target transform.

    model_mode:
      - "rf" / "rf_qr"         : RandomForestRegressor
      - "rf_cal"               : RF + linear calibration
      - "extratrees"           : ExtraTreesRegressor
      - "extratrees_cal"       : ExtraTrees + linear calibration
      - "xgb" / "xgb_cal"      : XGBRegressor (requires xgboost)
      - "ngb"                  : NGBRegressor with Normal dist (requires ngboost)
    """
    mode = model_mode.lower()
    if mode not in ALLOWED_MODEL_MODES:
        raise ValueError(f"Unknown model_mode={model_mode!r}. Allowed: {ALLOWED_MODEL_MODES}")

    # ---------- NGBoost (probabilistic) ----------
    if mode == "ngb":
        if NGBRegressor is None or Normal is None:
            raise ImportError(
                "ngboost is not installed. Install with `pip install ngboost` "
                "or avoid MODEL_MODE='ngb'."
            )
        # For NGBoost we do NOT wrap in TransformedTargetRegressor; it handles
        # distributions internally.
        base = NGBRegressor(
            Dist=Normal,
            n_estimators=600,
            learning_rate=0.03,
            minibatch_frac=1.0,
            verbose=False,
            random_state=42,
        )
        return base

    # ---------- XGBoost, RF, ExtraTrees (all wrapped in our TTR logic) ----------
    if mode.startswith("xgb"):
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install with `pip install xgboost` "
                "or avoid MODEL_MODE='xgb'."
            )
        base = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
    elif mode.startswith("extra"):
        base = ExtraTreesRegressor(**rf_params)
    else:
        # "rf" / "rf_qr" / "rf_cal"
        base = RandomForestRegressor(**rf_params)

    # Apply target transform (logit, Yeo-Johnson, etc.) for RF / ET / XGB
    ttr = choose_ttr(target_name, base)

    # Optional linear calibration on top
    if "cal" in mode:
        return LinearCalibratedRegressor(ttr)
    return ttr



def _train_eval_default_cv(
    X_t: pd.DataFrame,
    y_t: pd.Series,
    rf_params: dict,
    dt_params: dict,
    kf: KFold,
    model_mode: str,
    target_name: str = "",
    verbose: bool = False,
):
    est = _build_rf_like_estimator(model_mode, rf_params, target_name)

    # Sample weights: optional balancing + small-n boost
    sample_weight = _compute_sample_weight(X_t, y_t, strategy=SAMPLE_WEIGHT_STRATEGY)
    if len(y_t) < 600:
        sample_weight *= 1.5
    sample_weight = np.asarray(sample_weight, dtype=float)

    def _cv_scores_weighted(estimator, X, y, cv, w):
        r2_scores = []
        rmse_scores = []
        for tr_idx, te_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = w[tr_idx]

            est_ = clone(estimator)
            est_.fit(X_tr, y_tr, sample_weight=w_tr)
            y_pred = est_.predict(X_te)

            r2_scores.append(r2_score(y_te, y_pred))
            rmse_scores.append(rmse(y_te, y_pred))
        return float(np.mean(r2_scores)), float(np.mean(rmse_scores))

    # RF / ExtraTrees CV (weighted)
    r2_rf, rmse_rf = _cv_scores_weighted(est, X_t, y_t, kf, sample_weight)

    # Fit final RF on all data with weights
    est.fit(X_t, y_t, sample_weight=sample_weight)

    # Decision Tree baseline (weighted CV)
    dt = DecisionTreeRegressor(**dt_params)
    r2_dt, rmse_dt = _cv_scores_weighted(dt, X_t, y_t, kf, sample_weight)
    dt.fit(X_t, y_t, sample_weight=sample_weight)

    # Linear Regression baseline with polynomial interactions (weighted CV)
    X_lr = _build_lr_design_matrix(X_t)
    lr = LinearRegression()
    r2_lr, rmse_lr = _cv_scores_weighted(lr, X_lr, y_t, kf, sample_weight)
    lr.fit(X_lr, y_t, sample_weight=sample_weight)

    if verbose:
        print(f"R²_RF={r2_rf:.3f}, R²_DT={r2_dt:.3f}, R²_LR={r2_lr:.3f}")

    return r2_rf, rmse_rf, r2_dt, rmse_dt, r2_lr, rmse_lr, est, dt, lr



# ---------------- Public API ----------------
def train_auto_from_csv_default(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    save_dir: str,
    best_params_csv: Optional[str] = None,
    cols_y: Optional[Iterable[str]] = None,
    k_splits: int = 10,
    random_state: int = 42,
    use_best_params: bool = False,
    model_mode: str = "rf",
    do_grid_search: bool = False,
) -> pd.DataFrame:
    os.makedirs(save_dir, exist_ok=True)
    cols_y = list(cols_y) if cols_y is not None else COLS_Y_DEFAULT

    Xn = _ensure_numeric_matrix(X)
    Yn = Y.copy()

    # Load existing best params (if any)
    RF_BEST, DT_BEST = ({}, {})
    if best_params_csv and os.path.exists(best_params_csv):
        RF_BEST, DT_BEST = load_best_params(best_params_csv)
        print(
            f"✓ Loaded existing best params: {len(RF_BEST)} RF targets, "
            f"{len(DT_BEST)} DT targets"
        )
    else:
        print("ℹ️ No existing best_params CSV found; will start from RF_DEFAULT.")

    if use_best_params and not do_grid_search:
        print("✓ Using precomputed best params where available.")
    elif do_grid_search:
        print("✓ RF grid search enabled; best params will be written back to CSV.")
    else:
        print("✓ Using RF_DEFAULT for all targets (no best_params, no grid search).")

    print(f"✓ model_mode = {model_mode}")

    rows = []
    for target in cols_y:
        if target not in Yn.columns:
            rows.append(
                {
                    "Target": target,
                    "Samples": 0,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        X_t, y_t = _rows_for_target(Xn, Yn, target)
        n = len(X_t)
        if n < 30:
            rows.append(
                {
                    "Target": target,
                    "Samples": n,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        safe_target = _sanitize_target_name(target)

        n_splits = min(k_splits, max(3, n // 5))
        n_splits = 10
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


        # --- decide RF params for this target ---
        base_rf = RF_DEFAULT.copy()
        if use_best_params and target in RF_BEST:
            base_rf.update(_filter_params(RF_BEST[target], RF_ALLOWED))


        # sample weights (also reused in grid search)
        sample_weight = _compute_sample_weight(X_t, y_t, strategy=SAMPLE_WEIGHT_STRATEGY)
        if len(y_t) < 600:
            sample_weight *= 1.5

        if do_grid_search:
            best_rf, best_score = _rf_grid_search(
                X_t,
                y_t,
                base_params=base_rf,
                cv=kf,
                sample_weight=sample_weight,
                random_state=random_state,
            )
            RF_BEST[target] = best_rf.copy()
            rf_params = best_rf
            print(f"GS {target}: best R²≈{best_score:.3f}")
        else:
            rf_params = base_rf

        dt_params = DT_DEFAULT.copy()
        if use_best_params and target in DT_BEST:
            dt_params |= _filter_params(DT_BEST[target], DT_ALLOWED)

        print(f"Training {target} (n={n}, folds={n_splits})...", end=" ", flush=True)
        (
            r2_rf,
            rmse_rf,
            r2_dt,
            rmse_dt,
            r2_lr,
            rmse_lr,
            rf,
            dt,
            lr,
        ) = _train_eval_default_cv(
            X_t,
            y_t,
            rf_params,
            dt_params,
            kf,
            model_mode=model_mode,
            target_name=target,
            verbose=False,
        )
        print(f"R²={r2_rf:.3f}")

        joblib.dump(rf, os.path.join(save_dir, f"rf_{safe_target}.joblib"))
        joblib.dump(dt, os.path.join(save_dir, f"dt_{safe_target}.joblib"))
        joblib.dump(lr, os.path.join(save_dir, f"lr_{safe_target}.joblib"))

        rows.append(
            {
                "Target": target,
                "Samples": n,
                "R2_RF": round(r2_rf, 3),
                "RMSE_RF": round(rmse_rf, 3),
                "R2_DT": round(r2_dt, 3),
                "RMSE_DT": round(rmse_dt, 3),
                "R2_LR": round(r2_lr, 3),
                "RMSE_LR": round(rmse_lr, 3),
            }
        )

    # persist best params if grid search was used
    if do_grid_search and best_params_csv:
        print(f"\n💾 Saving best params to: {os.path.abspath(best_params_csv)}")
        print(f"   RF_BEST keys: {list(RF_BEST.keys())[:5]} ...")
        print(f"   Example B_Y params at save time: {RF_BEST.get('B_Y', None)}")
        save_best_params_csv(best_params_csv, RF_BEST, DT_BEST, cols_y)

    return pd.DataFrame(rows)



def train_auto_groupkfold(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
    save_dir: str,
    best_params_csv: Optional[str] = None,
    cols_y: Optional[Iterable[str]] = None,
    use_best_params: bool = False,
    model_mode: str = "rf",
    do_grid_search: bool = False,
) -> pd.DataFrame:
    """
    GroupKFold CV using externally provided `groups` (e.g. DOI).
    """
    os.makedirs(save_dir, exist_ok=True)
    cols_y = list(cols_y) if cols_y is not None else COLS_Y_DEFAULT

    Xn = _ensure_numeric_matrix(X)
    Yn = Y.copy()
    groups = groups.loc[Xn.index]

    RF_BEST, DT_BEST = ({}, {})
    if best_params_csv and os.path.exists(best_params_csv):
        RF_BEST, DT_BEST = load_best_params(best_params_csv)
        print(
            f"✓ Loaded existing best params: {len(RF_BEST)} RF targets, "
            f"{len(DT_BEST)} DT targets"
        )

    if use_best_params and not do_grid_search:
        print("✓ Using precomputed best params where available.")
    elif do_grid_search:
        print("✓ RF grid search enabled; best params will be written back to CSV.")
    else:
        print("✓ Using RF_DEFAULT for all targets (no best_params, no grid search).")
    print(f"✓ model_mode = {model_mode}")

    rows = []
    for target in cols_y:
        if target not in Yn.columns:
            rows.append(
                {
                    "Target": target,
                    "Samples": 0,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        X_t, y_t = _rows_for_target(Xn, Yn, target)
        g_t = groups.loc[X_t.index]

        perm = np.random.RandomState(42).permutation(len(X_t))
        X_t = X_t.iloc[perm].reset_index(drop=True)
        y_t = y_t.iloc[perm].reset_index(drop=True)
        g_t = g_t.iloc[perm].reset_index(drop=True)

        n = len(X_t)
        if n < 30:
            rows.append(
                {
                    "Target": target,
                    "Samples": n,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        print(f"Training {target} (n={n})...", end=" ", flush=True)

        safe_target = _sanitize_target_name(target)

        n_splits = min(10, g_t.nunique())
        n_splits = max(n_splits, 3)
        cv = GroupKFold(n_splits=n_splits)

        base_rf = RF_DEFAULT.copy()
        if use_best_params and target in RF_BEST:
            base_rf.update(_filter_params(RF_BEST[target], RF_ALLOWED))


        sample_weight = _compute_sample_weight(X_t, y_t, strategy=SAMPLE_WEIGHT_STRATEGY)
        if len(y_t) < 600:
            sample_weight *= 1.5

        if do_grid_search:
            best_rf, best_score = _rf_grid_search(
                X_t,
                y_t,
                base_params=base_rf,
                cv=cv.split(X_t, y_t, g_t),
                # RandomizedSearchCV expects cv splitter *or* int; wrap via split generator
                sample_weight=sample_weight,
            )
            RF_BEST[target] = best_rf.copy()
            rf_params = best_rf
            print(f"GS {target}: best R²≈{best_score:.3f}")
        else:
            rf_params = base_rf

        est = _build_rf_like_estimator(model_mode, rf_params, target)

        r2_rf = cross_val_score(
            est, X_t, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_rf = -cross_val_score(
            est, X_t, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()
        est.fit(X_t, y_t, sample_weight=sample_weight)
        joblib.dump(est, os.path.join(save_dir, f"rf_{safe_target}.joblib"))

        dt_params = DT_DEFAULT.copy()
        if use_best_params and target in DT_BEST:
            dt_params |= _filter_params(DT_BEST[target], DT_ALLOWED)

        dt = DecisionTreeRegressor(**dt_params)
        r2_dt = cross_val_score(
            dt, X_t, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_dt = -cross_val_score(
            dt, X_t, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()
        dt.fit(X_t, y_t, sample_weight=sample_weight)
        joblib.dump(dt, os.path.join(save_dir, f"dt_{safe_target}.joblib"))

        X_lr = _build_lr_design_matrix(X_t)
        lr = LinearRegression()
        r2_lr = cross_val_score(
            lr, X_lr, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_lr = -cross_val_score(
            lr, X_lr, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()
        lr.fit(X_lr, y_t, sample_weight=sample_weight)
        joblib.dump(lr, os.path.join(save_dir, f"lr_{safe_target}.joblib"))

        print(f"R²={r2_rf:.3f}")

        rows.append(
            {
                "Target": target,
                "Samples": n,
                "R2_RF": round(r2_rf, 3),
                "RMSE_RF": round(rmse_rf, 3),
                "R2_DT": round(r2_dt, 3),
                "RMSE_DT": round(rmse_dt, 3),
                "R2_LR": round(r2_lr, 3),
                "RMSE_LR": round(rmse_lr, 3),
            }
        )

    if do_grid_search and best_params_csv:
        save_best_params_csv(best_params_csv, RF_BEST, DT_BEST, cols_y)

    return pd.DataFrame(rows)


def train_auto_groupkfold_picklesafe(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
    save_dir: str,
    best_params_csv: Optional[str] = None,
    cols_y: Optional[Iterable[str]] = None,
    conformal_dir: Optional[str] = None,
    use_best_params: bool = False,
    model_mode: str = "rf",
) -> pd.DataFrame:
    """
    GroupKFold CV by external `groups` with picklable target transforms and
    conformal 90% residual widths.
    """
    os.makedirs(save_dir, exist_ok=True)
    if conformal_dir:
        os.makedirs(conformal_dir, exist_ok=True)
    cols_y = list(cols_y) if cols_y is not None else COLS_Y_DEFAULT

    Xn = _ensure_numeric_matrix(X)
    Yn = Y.copy()
    groups = groups.loc[Xn.index]

    RF_BEST, DT_BEST = ({}, {})
    if use_best_params and best_params_csv:
        RF_BEST, DT_BEST = load_best_params(best_params_csv)
        print(
            f"✓ Using best params from CSV for {len(RF_BEST)} RF targets, "
            f"{len(DT_BEST)} DT targets"
        )
    else:
        print("✓ Using RF_DEFAULT (n_estimators=600, max_depth=40, max_features=0.5)")
    print(f"✓ model_mode = {model_mode}")

    rows = []
    for target in cols_y:
        if target not in Yn.columns:
            rows.append(
                {
                    "Target": target,
                    "Samples": 0,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        X_t, y_t = _rows_for_target(Xn, Yn, target)
        g_t = groups.loc[X_t.index]

        perm = np.random.RandomState(42).permutation(len(X_t))
        X_t = X_t.iloc[perm].reset_index(drop=True)
        y_t = y_t.iloc[perm].reset_index(drop=True)
        g_t = g_t.iloc[perm].reset_index(drop=True)

        n = len(X_t)
        if n < 30:
            rows.append(
                {
                    "Target": target,
                    "Samples": n,
                    "R2_RF": None,
                    "RMSE_RF": None,
                    "R2_DT": None,
                    "RMSE_DT": None,
                    "R2_LR": None,
                    "RMSE_LR": None,
                }
            )
            continue

        print(f"Training {target} (n={n})...", end=" ", flush=True)

        safe_target = _sanitize_target_name(target)

        n_splits = min(10, g_t.nunique())
        n_splits = max(n_splits, 3)
        cv = GroupKFold(n_splits=n_splits)

        rf_params = RF_DEFAULT | _filter_params(RF_BEST.get(target, {}), RF_ALLOWED)
        est = _build_rf_like_estimator(model_mode, rf_params, target)

        sample_weight = _compute_sample_weight(X_t, y_t, strategy=SAMPLE_WEIGHT_STRATEGY)
        if len(y_t) < 600:
            sample_weight *= 1.5

        r2_rf = cross_val_score(
            est, X_t, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_rf = -cross_val_score(
            est, X_t, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()

        est.fit(X_t, y_t, sample_weight=sample_weight)
        joblib.dump(est, os.path.join(save_dir, f"rf_{safe_target}.joblib"))

        if conformal_dir:
            # --- collect absolute residuals AND their feature vectors (local CP) ---
            abs_resid_blocks = []
            X_cal_blocks = []

            # Use the same feature order we feed into the RF
            model_features = list(X_t.columns)

            for tr, te in cv.split(X_t, y_t, g_t):
                est_tmp = _build_rf_like_estimator(model_mode, rf_params, target)
                est_tmp.fit(X_t.iloc[tr], y_t.iloc[tr])

                X_te = X_t.iloc[te][model_features]
                pred = est_tmp.predict(X_te)
                resid = np.abs(pred - y_t.iloc[te].values)

                abs_resid_blocks.append(resid)
                X_cal_blocks.append(X_te.to_numpy(dtype=float))

            if abs_resid_blocks:
                abs_resid = np.concatenate(abs_resid_blocks, axis=0)
                X_cal = np.vstack(X_cal_blocks)

                q90 = float(np.quantile(abs_resid, 0.90))
                safe_name = safe_target

                base = os.path.join(conformal_dir, f"rf_conf_{safe_name}")

                # 1) JSON summary (backward compatible: global 90% width)
                with open(base + ".json", "w") as f:
                    json.dump(
                        {
                            "target": target,
                            "q90_abs_resid": q90,
                            "n_resid": int(len(abs_resid)),
                        },
                        f,
                    )

                # 2) Global residuals (unchanged from before)
                np.save(base + "_abs_resid.npy", abs_resid)

                # 3) NEW: local calibration arrays for local conformal UQ
                np.savez_compressed(
                    os.path.join(conformal_dir, f"rf_conf_local_{safe_name}.npz"),
                    X_cal=X_cal,
                    resid=abs_resid,
                    feature_names=np.array(model_features),
                )
            else:
                print(f"⚠️ No calibration residuals collected for {target}.")


        dt_params = DT_DEFAULT | _filter_params(DT_BEST.get(target, {}), DT_ALLOWED)
        dt = DecisionTreeRegressor(**dt_params)
        r2_dt = cross_val_score(
            dt, X_t, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_dt = -cross_val_score(
            dt, X_t, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()
        dt.fit(X_t, y_t, sample_weight=sample_weight)
        joblib.dump(dt, os.path.join(save_dir, f"dt_{safe_target}.joblib"))

        X_lr = _build_lr_design_matrix(X_t)
        lr = LinearRegression()
        r2_lr = cross_val_score(
            lr, X_lr, y_t, cv=cv, scoring="r2", n_jobs=-1, groups=g_t
        ).mean()
        rmse_lr = -cross_val_score(
            lr, X_lr, y_t, cv=cv, scoring=RMSE_SCORER, n_jobs=-1, groups=g_t
        ).mean()
        lr.fit(X_lr, y_t, sample_weight=sample_weight)
        joblib.dump(lr, os.path.join(save_dir, f"lr_{safe_target}.joblib"))

        print(f"R²={r2_rf:.3f}")

        rows.append(
            {
                "Target": target,
                "Samples": n,
                "R2_RF": round(r2_rf, 3),
                "RMSE_RF": round(rmse_rf, 3),
                "R2_DT": round(r2_dt, 3),
                "RMSE_DT": round(rmse_dt, 3),
                "R2_LR": round(r2_lr, 3),
                "RMSE_LR": round(rmse_lr, 3),
            }
        )

    return pd.DataFrame(rows)


# ---------- Lightweight RF tuning (OOB, no extra CV) ----------
def tune_rf_defaults_oob(
    X: pd.DataFrame,
    y: pd.Series,
    base_defaults: Optional[dict] = None,
    max_samples: int = 800,
    random_state: int = 42,
) -> dict:
    """
    Very lightweight tuning of RF_DEFAULT using OOB score on a single target.
    """
    if base_defaults is None:
        base_defaults = RF_DEFAULT

    X_t = X.copy()
    y_t = pd.to_numeric(y, errors="coerce").dropna()
    X_t = X_t.loc[y_t.index]
    n = len(X_t)
    if n > max_samples:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(n, size=max_samples, replace=False)
        X_t = X_t.iloc[idx]
        y_t = y_t.iloc[idx]

    print(f"⚙️ Tuning RF defaults on {len(X_t)} samples (OOB-based, single target)")

    candidate_overrides = [
        {},
        {"max_depth": 30, "min_samples_leaf": 1},
        {"max_depth": 60, "min_samples_leaf": 2},
        {"max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    ]

    best_score = -np.inf
    best_params = base_defaults.copy()

    for i, override in enumerate(candidate_overrides):
        params = base_defaults.copy()
        params.update(override)
        params["oob_score"] = True

        rf = RandomForestRegressor(**params)
        rf.fit(X_t, y_t)

        score = getattr(rf, "oob_score_", np.nan)
        print(f"  Candidate {i+1}: override={override} -> OOB R²={score:.3f}")

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_params = params.copy()

    best_params.pop("oob_score", None)
    print("\n✅ Best OOB R²={:.3f}".format(best_score))
    print("   Suggested RF_DEFAULT update:")
    for k in sorted(best_params.keys()):
        print(f"   {k!r}: {best_params[k]!r},")

    return best_params


# ---------- RF ensemble uncertainty helper ----------
def rf_predict_with_uq(model, X, alpha: float = 0.1):
    """
    Predict mean and an approximate (1-alpha) interval using the RF ensemble.

    Works with:
      - RandomForestRegressor / ExtraTreesRegressor
      - TransformedTargetRegressor(RandomForest/ExtraTrees)
      - LinearCalibratedRegressor(...) on top of those

    Returns:
      y_mean, y_lo, y_hi  (all 1D arrays)
    """
    X_arr = np.asarray(X)
    y_mean = np.asarray(model.predict(X_arr), dtype=float).ravel()

    calibrator = None
    inverse = None
    base = model

    # Unwrap linear calibration, if present
    if isinstance(base, LinearCalibratedRegressor) and hasattr(base, "base_regressor_"):
        calibrator = getattr(base, "calibrator_", None)
        base = base.base_regressor_

    # Unwrap TransformedTargetRegressor, tracking inverse transform
    if isinstance(base, TransformedTargetRegressor) and hasattr(base, "regressor_"):
        ttr = base
        if getattr(ttr, "inverse_func", None) is not None:
            inverse = ttr.inverse_func
        elif hasattr(ttr, "transformer_"):
            transformer = ttr.transformer_

            def _inv(z):
                z = np.asarray(z, dtype=float).reshape(-1, 1)
                return transformer.inverse_transform(z).ravel()

            inverse = _inv
        base = ttr.regressor_

    # Now we expect a forest
    if not isinstance(base, (RandomForestRegressor, ExtraTreesRegressor)) or not hasattr(
        base, "estimators_"
    ):
        lo = np.full_like(y_mean, np.nan)
        hi = np.full_like(y_mean, np.nan)
        return y_mean, lo, hi

    all_tree = []
    for tree in base.estimators_:
        pred = tree.predict(X_arr).ravel()
        if inverse is not None:
            pred = inverse(pred)
        all_tree.append(pred)

    if not all_tree:
        lo = np.full_like(y_mean, np.nan)
        hi = np.full_like(y_mean, np.nan)
        return y_mean, lo, hi

    all_tree = np.stack(all_tree, axis=0)

    # to apply linear calibration if present
    if calibrator is not None and hasattr(calibrator, "coef_"):
        a = float(calibrator.coef_[0])
        b = float(calibrator.intercept_)
        all_tree = a * all_tree + b

    lower = np.quantile(all_tree, alpha / 2.0, axis=0)
    upper = np.quantile(all_tree, 1.0 - alpha / 2.0, axis=0)

    return y_mean, lower, upper
