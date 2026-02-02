from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from typing import Iterable, Optional, Tuple, Dict

from sklearn.metrics import r2_score, mean_squared_error
from pandas.api.types import is_numeric_dtype


YIELD_COLS_DEFAULT = ["B_Y", "C_Y", "A_Y", "G_Y"]
DERIVED_ENERGY_CARBON = ["E_B", "E_H", "C_B", "C_H"]

PRIMITIVE_FOR_DERIVED = [
    "B_Y", "C_Y",
    "HHV_biooil", "HHV_biochar",
    "C_biooil", "C_biochar",
]


def _ensure_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(float)


def _sanitize_target_name(target: str) -> str:
    return target.replace("/", "_").replace("\\", "_").replace(":", "_")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def predict_all_targets(
    X: pd.DataFrame,
    save_dir: str,
    cols_y: Iterable[str],
    model_prefix: str = "rf_",
) -> pd.DataFrame:
    """
    Load saved models for each target and produce a DataFrame of raw predictions.
    Models are expected as {save_dir}/{model_prefix}{sanitized_target}.joblib
    """
    Xn = _ensure_numeric_matrix(X)
    cols_y = list(cols_y)

    preds: Dict[str, np.ndarray] = {}
    for tgt in cols_y:
        fname = f"{model_prefix}{_sanitize_target_name(tgt)}.joblib"
        fpath = os.path.join(save_dir, fname)
        if not os.path.exists(fpath):
            preds[tgt] = np.full(len(Xn), np.nan, dtype=float)
            continue

        model = joblib.load(fpath)
        y_hat = model.predict(Xn)
        preds[tgt] = np.asarray(y_hat, dtype=float).ravel()

    return pd.DataFrame(preds, index=Xn.index)


def enforce_yield_closure(
    df_pred: pd.DataFrame,
    yield_cols: Optional[Iterable[str]] = None,
    total: float = 100.0,
) -> pd.DataFrame:
    """
    Project predicted yields onto the simplex:
      - clip negatives to 0
      - renormalize so sum(yield_cols) == total where possible
    """
    if yield_cols is None:
        yield_cols = YIELD_COLS_DEFAULT
    yield_cols = [c for c in yield_cols if c in df_pred.columns]

    if not yield_cols:
        return df_pred

    df = df_pred.copy()
    Y = df[yield_cols].copy()

    Y = Y.clip(lower=0.0)
    s = Y.sum(axis=1)

    mask = s > 0
    Y.loc[mask, :] = Y.loc[mask].div(s[mask], axis=0).mul(total)

    df[yield_cols] = Y
    return df


def compute_derived_energy_carbon(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physics-derived E_B, E_H, C_B, C_H from primitive predictions.
    Uses:
      E_B = (B_Y/100) * HHV_biooil
      E_H = (C_Y/100) * HHV_biochar
      C_B = (B_Y/100) * (C_biooil/100)
      C_H = (C_Y/100) * (C_biochar/100)
    """
    df = df_pred.copy()
    for c in PRIMITIVE_FOR_DERIVED:
        if c not in df.columns:
            df[c] = np.nan

    BY = df["B_Y"].to_numpy(dtype=float)
    CY = df["C_Y"].to_numpy(dtype=float)
    HHV_bo = df["HHV_biooil"].to_numpy(dtype=float)
    HHV_ch = df["HHV_biochar"].to_numpy(dtype=float)
    C_bo = df["C_biooil"].to_numpy(dtype=float)
    C_ch = df["C_biochar"].to_numpy(dtype=float)

    df["E_B_phys"] = (BY / 100.0) * HHV_bo
    df["E_H_phys"] = (CY / 100.0) * HHV_ch
    df["C_B_phys"] = (BY / 100.0) * (C_bo / 100.0)
    df["C_H_phys"] = (CY / 100.0) * (C_ch / 100.0)

    return df


def compute_closure_error(
    df_pred: pd.DataFrame,
    yield_cols: Optional[Iterable[str]] = None,
    total: float = 100.0,
) -> pd.Series:
    """
    Compute closure error: (sum(yields) - total) for each row.
    """
    if yield_cols is None:
        yield_cols = YIELD_COLS_DEFAULT
    yield_cols = [c for c in yield_cols if c in df_pred.columns]
    if not yield_cols:
        return pd.Series(np.nan, index=df_pred.index, name="closure_error")

    s = df_pred[yield_cols].sum(axis=1)
    return (s - total).rename("closure_error")


def evaluate_physics_against_truth(
    Y_true: pd.DataFrame,
    Y_pred_raw: pd.DataFrame,
    Y_pred_phys: pd.DataFrame,
    yield_cols: Optional[Iterable[str]] = None,
    energy_carbon_cols: Optional[Iterable[str]] = None,
    min_samples: int = 15,
) -> pd.DataFrame:
    """
    Build a summary table of R²/RMSE:
      - For all targets in Y_true that exist in Y_pred_raw: R²/RMSE of raw RF.
      - For physics-derived energy/carbon (E_B/E_H/C_B/C_H):
          * R²_phys/RMSE_phys using *_phys columns in Y_pred_phys.
      - For yields: closure error stats, before and after enforcement.
    """
    if yield_cols is None:
        yield_cols = YIELD_COLS_DEFAULT
    if energy_carbon_cols is None:
        energy_carbon_cols = DERIVED_ENERGY_CARBON

    rows = []

    # 1) Generic metrics for all targets with raw predictions
    for tgt in Y_true.columns:
        if tgt not in Y_pred_raw.columns:
            continue

        y_true = pd.to_numeric(Y_true[tgt], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(Y_pred_raw[tgt], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n = int(mask.sum())
        if n < min_samples:
            rows.append(
                {
                    "Target": tgt,
                    "Samples": n,
                    "R2_RF_raw": np.nan,
                    "RMSE_RF_raw": np.nan,
                    "R2_phys": np.nan,
                    "RMSE_phys": np.nan,
                }
            )
            continue

        r2_raw = float(r2_score(y_true[mask], y_pred[mask]))
        rmse_raw = _rmse(y_true[mask], y_pred[mask])

        r2_phys = np.nan
        rmse_phys = np.nan

        if tgt in energy_carbon_cols:
            col_phys = tgt + "_phys"
            if col_phys in Y_pred_phys.columns:
                y_pred_phys = pd.to_numeric(
                    Y_pred_phys[col_phys], errors="coerce"
                ).to_numpy(dtype=float)
                mask2 = mask & np.isfinite(y_pred_phys)
                n2 = int(mask2.sum())
                if n2 >= min_samples:
                    r2_phys = float(r2_score(y_true[mask2], y_pred_phys[mask2]))
                    rmse_phys = _rmse(y_true[mask2], y_pred_phys[mask2])

        rows.append(
            {
                "Target": tgt,
                "Samples": n,
                "R2_RF_raw": round(r2_raw, 3),
                "RMSE_RF_raw": round(rmse_raw, 3),
                "R2_phys": None if np.isnan(r2_phys) else round(r2_phys, 3),
                "RMSE_phys": None if np.isnan(rmse_phys) else round(rmse_phys, 3),
            }
        )

    # 2) Yield closure stats (global)
    closure_raw = compute_closure_error(Y_pred_raw, yield_cols=yield_cols)
    closure_phys = compute_closure_error(Y_pred_phys, yield_cols=yield_cols)

    if closure_raw.notna().any():
        ce = closure_raw.dropna().to_numpy(dtype=float)
        rows.append(
            {
                "Target": "closure_raw",
                "Samples": int(len(ce)),
                "R2_RF_raw": np.nan,
                "RMSE_RF_raw": float(np.sqrt(np.mean(ce**2))),
                "R2_phys": np.nan,
                "RMSE_phys": None,
            }
        )

    if closure_phys.notna().any():
        ce = closure_phys.dropna().to_numpy(dtype=float)
        rows.append(
            {
                "Target": "closure_phys",
                "Samples": int(len(ce)),
                "R2_RF_raw": np.nan,
                "RMSE_RF_raw": float(np.sqrt(np.mean(ce**2))),
                "R2_phys": np.nan,
                "RMSE_phys": None,
            }
        )

    return pd.DataFrame(rows)


def run_full_physics_postprocess(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    save_dir: str,
    cols_y: Iterable[str],
    yield_cols: Optional[Iterable[str]] = None,
    min_samples: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
      1) Predict all targets with saved RF models.
      2) Enforce yield closure + compute derived physics quantities.
      3) Evaluate against truth and return metrics.
    Returns:
      Y_pred_raw, Y_pred_phys, metrics_df
    """
    cols_y = list(cols_y)
    Y_pred_raw = predict_all_targets(X, save_dir, cols_y)

    if yield_cols is None:
        yield_cols = YIELD_COLS_DEFAULT

    Y_pred_closure = enforce_yield_closure(Y_pred_raw.copy(), yield_cols=yield_cols)
    Y_pred_phys = compute_derived_energy_carbon(Y_pred_closure)

    metrics_df = evaluate_physics_against_truth(
        Y_true=Y,
        Y_pred_raw=Y_pred_raw,
        Y_pred_phys=Y_pred_phys,
        yield_cols=yield_cols,
        min_samples=min_samples,
    )
    return Y_pred_raw, Y_pred_phys, metrics_df

import numpy as np
import pandas as pd

# reuse same constants
YIELD_COLS = ["B_Y", "C_Y", "A_Y", "G_Y"]
DERIVED_ENERGY_CARBON = ["E_B", "E_H", "C_B", "C_H"]

def _compute_derived_from_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute E_B, E_H, C_B, C_H from yields + HHV + C%.
    Assumes columns:
      B_Y, C_Y (wt%), HHV_biooil, HHV_biochar (MJ/kg),
      C_biooil, C_biochar (wt%).
    """
    out = df.copy()

    BY = out.get("B_Y")
    CY = out.get("C_Y")
    HHV_bo = out.get("HHV_biooil")
    HHV_ch = out.get("HHV_biochar")
    C_bo = out.get("C_biooil")
    C_ch = out.get("C_biochar")

    # Guard: only compute where all inputs are finite
    def _safe_prod(*cols):
        arrs = [np.asarray(c, float) for c in cols]
        mask = np.isfinite(arrs[0])
        for a in arrs[1:]:
        # ensure matching mask
            mask &= np.isfinite(a)
        res = np.full_like(arrs[0], np.nan, dtype=float)
        res[mask] = 1.0
        for a in arrs:
            res[mask] *= a[mask]
        return res

    if BY is not None and HHV_bo is not None:
        out["E_B_phys"] = (BY.values / 100.0) * HHV_bo.values
    if CY is not None and HHV_ch is not None:
        out["E_H_phys"] = (CY.values / 100.0) * HHV_ch.values
    if BY is not None and C_bo is not None:
        out["C_B_phys"] = (BY.values / 100.0) * (C_bo.values / 100.0)
    if CY is not None and C_ch is not None:
        out["C_H_phys"] = (CY.values / 100.0) * (C_ch.values / 100.0)

    return out


def apply_physics_to_predictions(Y_pred_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Post-hoc physics projection:
      1) Enforce B_Y + C_Y + A_Y + G_Y = 100 (on rows where all 4 are finite).
      2) Recompute E_B, E_H, C_B, C_H from physics using the *corrected* yields.

    Returns:
      DataFrame with:
        - original columns unchanged
        - new columns:
            B_Y_phys, C_Y_phys, A_Y_phys, G_Y_phys
            E_B_phys, E_H_phys, C_B_phys, C_H_phys
    """
    Y_raw = Y_pred_raw.copy()
    Y_phys = Y_raw.copy()

    # ---- 1) YIELD CLOSURE ----
    if all(col in Y_raw.columns for col in YIELD_COLS):
        Y_block = Y_raw[YIELD_COLS].astype(float)
        sums = Y_block.sum(axis=1)

        # rows where closure makes sense: finite and sum>0
        mask = np.isfinite(sums) & (sums > 0)

        # normalized yields
        Y_norm = Y_block.copy()
        Y_norm.loc[mask] = Y_block.loc[mask].div(sums[mask], axis=0) * 100.0

        # store as separate phys columns (do not overwrite raw)
        for c in YIELD_COLS:
            Y_phys[f"{c}_phys"] = Y_norm[c]
    else:
        # if yields missing, just return existing
        return Y_phys

    # ---- 2) DERIVED ENERGY/CARBON FROM PHYS-CORRECTED YIELDS ----
    # Build a temporary df using phys yields where available
    tmp = Y_phys.copy()
    for c in YIELD_COLS:
        phys_col = f"{c}_phys"
        if phys_col in tmp.columns:
            tmp[c] = tmp[phys_col]

    Y_phys = _compute_derived_from_predictions(tmp)

    return Y_phys


def _predict_all_targets_from_dict(
    X: pd.DataFrame,
    rf_models: dict,
    cols_y: Iterable[str],
) -> pd.DataFrame:
    """
    Use a dict of trained RF models {target_name: estimator}
    to predict all targets for a given X.
    Missing models -> NaN column.
    """
    preds = {}
    for target in cols_y:
        model = rf_models.get(target)
        if model is None:
            preds[target] = np.full(len(X), np.nan, dtype=float)
        else:
            preds[target] = model.predict(X)
    return pd.DataFrame(preds, index=X.index)


def predict_with_physics(
    X: pd.DataFrame,
    rf_models: dict,
    cols_y: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
        1) Predict all targets with RF.
        2) Apply physics post-hoc corrections.

    Returns:
        Y_pred_raw  : DataFrame with raw RF outputs
        Y_pred_phys : DataFrame with additional *_phys columns
    """
    Y_pred_raw = _predict_all_targets_from_dict(X, rf_models, cols_y)
    Y_pred_phys = apply_physics_to_predictions(Y_pred_raw)
    return Y_pred_raw, Y_pred_phys
