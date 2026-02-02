from __future__ import annotations
"""
qa_envelopes.py — Envelope-aware QA utilities for HTT/HTL/HTC datasets.

Public API:
- plot_energy_carbon_envelopes(df): histograms for E_B, E_H, C_B, C_H with logical borders
- plot_yield_envelopes(df): histograms for B_Y, C_Y, A_Y, G_Y with logical borders
- run_basic_qc(df): quick consistency checks and small peeks
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()

def _safe_json_load(maybe_json):
    try:
        if pd.isna(maybe_json):
            return {}
    except Exception:
        pass
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, (str, bytes)):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.copy()
    for c in cols:
        if c not in g.columns:
            g[c] = np.nan
    return g

def _first_present(row: pd.Series, keys: list[str], default=None):
    for k in keys:
        if k in row.index:
            v = row.get(k, default)
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            return v
    return default

def _proc_from_row(row: pd.Series) -> str | None:
    v = row.get("Process_type", None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = row.get("process_type", None)
    return v


def _proc_default_envelopes(proc: str | None) -> dict:
    p = _norm_text(proc)
    if ("hydrothermal liquefaction" in p) or (p == "htl") or ("liquefaction" in p):
        return {"E_B_range": (0.0, 25.0), "E_H_range": (0.0, 8.0),
                "C_B_range": (0.0, 0.55), "C_H_range": (0.0, 0.25)}
    if ("hydrothermal carbon" in p) or (p == "htc") or ("carbonization" in p):
        return {"E_B_range": (0.0, 3.0), "E_H_range": (0.0, 22.0),
                "C_B_range": (0.0, 0.10), "C_H_range": (0.0, 0.65)}
    if "pyrolysis" in p:
        return {"E_B_range": (0.0, 22.0), "E_H_range": (0.0, 18.0),
                "C_B_range": (0.0, 0.60), "C_H_range": (0.0, 0.60)}
    return {"E_B_range": (0.0, 25.0), "E_H_range": (0.0, 22.0),
            "C_B_range": (0.0, 0.60), "C_H_range": (0.0, 0.60)}

def _row_caps_from_values(r: pd.Series, eps: float = 0.05) -> dict:
    BY      = pd.to_numeric(r.get("B_Y"), errors="coerce")
    CY      = pd.to_numeric(r.get("C_Y"), errors="coerce")
    HHV_bo  = pd.to_numeric(r.get("HHV_biooil"), errors="coerce")
    HHV_ch  = pd.to_numeric(r.get("HHV_biochar"), errors="coerce")
    C_bo    = pd.to_numeric(r.get("C_biooil"), errors="coerce")
    C_ch    = pd.to_numeric(r.get("C_biochar"), errors="coerce")

    caps = {}
    if pd.notna(BY) and pd.notna(HHV_bo):
        caps["E_B_range"] = (0.0, max(0.0, (BY/100.0) * HHV_bo * (1.0 + eps)))
    if pd.notna(CY) and pd.notna(HHV_ch):
        caps["E_H_range"] = (0.0, max(0.0, (CY/100.0) * HHV_ch * (1.0 + eps)))
    if pd.notna(BY) and pd.notna(C_bo):
        caps["C_B_range"] = (0.0, max(0.0, (BY/100.0) * (C_bo/100.0) * (1.0 + eps)))
    if pd.notna(CY) and pd.notna(C_ch):
        caps["C_H_range"] = (0.0, max(0.0, (CY/100.0) * (C_ch/100.0) * (1.0 + eps)))
    return caps

def _resolve_envelope(row: pd.Series) -> dict:
    ex = _safe_json_load(row.get("extra"))
    explicit = (ex.get("QA") or {}).get("envelope") or {}
    proc = _proc_from_row(row)

    if explicit:
        defaults = _proc_default_envelopes(proc)
        return {
            "E_B_range": tuple(explicit.get("E_B_range", defaults["E_B_range"])),
            "E_H_range": tuple(explicit.get("E_H_range", defaults["E_H_range"])),
            "C_B_range": tuple(explicit.get("C_B_range", defaults["C_B_range"])),
            "C_H_range": tuple(explicit.get("C_H_range", defaults["C_H_range"])),
        }
    caps = _row_caps_from_values(row, eps=0.05)
    if caps:
        defaults = _proc_default_envelopes(proc)
        return {
            "E_B_range": tuple(caps.get("E_B_range", defaults["E_B_range"])),
            "E_H_range": tuple(caps.get("E_H_range", defaults["E_H_range"])),
            "C_B_range": tuple(caps.get("C_B_range", defaults["C_B_range"])),
            "C_H_range": tuple(caps.get("C_H_range", defaults["C_H_range"])),
        }
    return _proc_default_envelopes(proc)

def _global_envelope_bounds(df: pd.DataFrame, range_key: str) -> tuple[float, float]:
    los, his = [], []
    for _, r in df.iterrows():
        env = _resolve_envelope(r)
        lo, hi = env[range_key]
        los.append(lo); his.append(hi)
    return (np.nanmin(los), np.nanmax(his))

def _plot_distribution(df: pd.DataFrame, value_col: str, range_key: str, xlabel: str, title: str):
    vals = pd.to_numeric(df.get(value_col), errors="coerce").dropna().values
    if vals.size == 0:
        print(f"No data to plot for {value_col}."); return
    lo, hi = _global_envelope_bounds(df, range_key)
    plt.figure()
    plt.hist(vals, bins=40)
    plt.axvline(lo, linestyle="--", linewidth=2, label="Envelope low")
    plt.axvline(hi, linestyle="--", linewidth=2, label="Envelope high")
    plt.xlabel(xlabel)
    plt.title(title + f"\nGlobal envelope: [{lo:.3g}, {hi:.3g}]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_carbon_envelopes(df: pd.DataFrame) -> None:
    cols_needed = ["extra","Process_type","process_type","E_B","E_H","C_B","C_H",
                   "B_Y","C_Y","HHV_biooil","HHV_biochar","C_biooil","C_biochar"]
    g = _ensure_cols(df, cols_needed)
    _plot_distribution(g, "E_B", "E_B_range", "E_B (MJ/kg dry biomass)",
                       "Distribution of E_B with logical border")
    _plot_distribution(g, "E_H", "E_H_range", "E_H (MJ/kg dry biomass)",
                       "Distribution of E_H with logical border")
    _plot_distribution(g, "C_B", "C_B_range", "C_B (kg C / kg dry biomass)",
                       "Distribution of C_B with logical border")
    _plot_distribution(g, "C_H", "C_H_range", "C_H (kg C / kg dry biomass)",
                       "Distribution of C_H with logical border")


# --------------------- yield envelopes ---------------------

def _proc_default_yield_envelopes(proc: str | None) -> dict:
    p = _norm_text(proc)
    if ("hydrothermal liq" in p) or ("liquefaction" in p) or (p == "htl"):
        return {"B_Y_range": (0.0, 65.0), "C_Y_range": (0.0, 35.0),
                "A_Y_range": (0.0, 80.0), "G_Y_range": (0.0, 30.0)}
    if ("hydrothermal carbon" in p) or ("carbonization" in p) or (p == "htc"):
        return {"B_Y_range": (0.0, 15.0), "C_Y_range": (20.0, 90.0),
                "A_Y_range": (0.0, 60.0), "G_Y_range": (0.0, 30.0)}
    if "pyrolysis" in p:
        return {"B_Y_range": (0.0, 75.0), "C_Y_range": (0.0, 50.0),
                "A_Y_range": (0.0, 15.0), "G_Y_range": (0.0, 40.0)}
    return {"B_Y_range": (0.0, 80.0), "C_Y_range": (0.0, 90.0),
            "A_Y_range": (0.0, 80.0), "G_Y_range": (0.0, 50.0)}

def _row_caps_from_yields(r: pd.Series, eps: float = 0.02) -> dict:
    BY = pd.to_numeric(r.get("B_Y"), errors="coerce")
    CY = pd.to_numeric(r.get("C_Y"), errors="coerce")
    AY = pd.to_numeric(r.get("A_Y"), errors="coerce")
    GY = pd.to_numeric(r.get("G_Y"), errors="coerce")

    def cap_from_others(others):
        if all(pd.notna(x) for x in others):
            rem = max(0.0, 100.0 - float(np.nansum(others)))
            return (0.0, rem * (1.0 + eps))
        return None

    caps = {}
    c = cap_from_others([CY, AY, GY])
    if c: caps["B_Y_range"] = c
    c = cap_from_others([BY, AY, GY])
    if c: caps["C_Y_range"] = c
    c = cap_from_others([BY, CY, GY])
    if c: caps["A_Y_range"] = c
    c = cap_from_others([BY, CY, AY])
    if c: caps["G_Y_range"] = c
    return caps

def _resolve_yield_envelope(row: pd.Series) -> dict:
    ex = _safe_json_load(row.get("extra"))
    explicit = (ex.get("QA") or {}).get("envelope") or {}
    defaults = _proc_default_yield_envelopes(_proc_from_row(row))
    if explicit:
        return {
            "B_Y_range": tuple(explicit.get("B_Y_range", defaults["B_Y_range"])),
            "C_Y_range": tuple(explicit.get("C_Y_range", defaults["C_Y_range"])),
            "A_Y_range": tuple(explicit.get("A_Y_range", defaults["A_Y_range"])),
            "G_Y_range": tuple(explicit.get("G_Y_range", defaults["G_Y_range"])),
        }
    caps = _row_caps_from_yields(row, eps=0.02)
    return {
        "B_Y_range": tuple(caps.get("B_Y_range", defaults["B_Y_range"])),
        "C_Y_range": tuple(caps.get("C_Y_range", defaults["C_Y_range"])),
        "A_Y_range": tuple(caps.get("A_Y_range", defaults["A_Y_range"])),
        "G_Y_range": tuple(caps.get("G_Y_range", defaults["G_Y_range"])),
    }

def _global_yield_bounds(df: pd.DataFrame, range_key: str) -> tuple[float, float]:
    los, his = [], []
    for _, r in df.iterrows():
        env = _resolve_yield_envelope(r)
        lo, hi = env[range_key]
        los.append(lo); his.append(hi)
    return (np.nanmin(los), np.nanmax(his))

def _plot_yield_distribution(df: pd.DataFrame, yield_col: str, range_key: str, title: str):
    vals = pd.to_numeric(df.get(yield_col), errors="coerce").dropna().values
    if vals.size == 0:
        print(f"No data to plot for {yield_col}."); return
    lo, hi = _global_yield_bounds(df, range_key)
    plt.figure()
    plt.hist(vals, bins=40)
    plt.axvline(lo, linestyle="--", linewidth=2, label="Envelope low")
    plt.axvline(hi, linestyle="--", linewidth=2, label="Envelope high")
    plt.xlabel(f"{yield_col} (wt% of dry feedstock)")
    plt.title(f"Distribution of {yield_col} with logical border\nGlobal envelope: [{lo:.3g}, {hi:.3g}]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_yield_envelopes(df: pd.DataFrame) -> None:
    cols_needed = ["extra","Process_type","process_type","B_Y","C_Y","A_Y","G_Y"]
    g = _ensure_cols(df, cols_needed)

    # 4-product mass balance check ~100% with slight tolerance
    mask_all4 = g[["B_Y","C_Y","A_Y","G_Y"]].notna().all(axis=1)
    if mask_all4.any():
        mb = g.loc[mask_all4, ["B_Y","C_Y","A_Y","G_Y"]].sum(axis=1)
        n_bad = ((mb < 98.0) | (mb > 102.0)).sum()
        print(f"Mass-balance check (B_Y+C_Y+A_Y+G_Y ≈ 100): {n_bad} violations out of {mask_all4.sum()} rows")

    _plot_yield_distribution(g, "B_Y", "B_Y_range", "Biocrude / Liquid yield (B_Y)")
    _plot_yield_distribution(g, "C_Y", "C_Y_range", "Char / Hydrochar yield (C_Y)")
    _plot_yield_distribution(g, "A_Y", "A_Y_range", "Aqueous-organics yield (A_Y)")
    _plot_yield_distribution(g, "G_Y", "G_Y_range", "Gas yield (G_Y)")


# ----------------------- basic QC -----------------------

def run_basic_qc(df: pd.DataFrame) -> None:
    # Ensure key columns exist to avoid KeyErrors/TypeErrors
    need = [
        "process_type","Process_type",
        "E_H","E_B","B_Y","C_Y","C_B","C_H",
        "HHV_input","H/C","O/C",
        "HHV_biooil","HHV_biochar","E_B","E_H","C_B","C_H",
        "T","t","Temp_C","t_min","Ref","Source_Figure"
    ]
    g = _ensure_cols(df, need)

    # Provide friendly aliases T/t from multiple sources (Temp_C/t_min or existing)
    if "T" not in g or g["T"].isna().all():
        g["T"] = g["Temp_C"]
    if "t" not in g or g["t"].isna().all():
        g["t"] = g["t_min"]

    # Normalize process tag for quick filtering/peeks
    g["proc"] = g.get("Process_type").astype(str).str.lower()
    null_mask = g["proc"].isin(["nan", "none"]) | g["proc"].isna()
    g.loc[null_mask, "proc"] = g.get("process_type").astype(str).str.lower()

    to_num = lambda s: pd.to_numeric(s, errors="coerce")

    # Heuristic caps (conservative)
    EH_hi_default = 25.0
    EB_hi_default = 25.0

    bad_EH = g[to_num(g["E_H"]) > EH_hi_default]
    bad_EB = g[to_num(g["E_B"]) > EB_hi_default]

    print(f"E_H > {EH_hi_default} MJ/kg: {len(bad_EH)} rows")
    print(f"E_B > {EB_hi_default} MJ/kg: {len(bad_EB)} rows")

    bad_BY = g[(to_num(g["B_Y"]) < 0) | (to_num(g["B_Y"]) > 100)]
    bad_CY = g[(to_num(g["C_Y"]) < 0) | (to_num(g["C_Y"]) > 100)]
    print(f"B_Y outside [0,100]: {len(bad_BY)} rows")
    print(f"C_Y outside [0,100]: {len(bad_CY)} rows")

    bad_CB = g[to_num(g["C_B"]) > 0.60]
    bad_CH = g[to_num(g["C_H"]) > 0.65]
    print(f"C_B > 0.60 (fraction): {len(bad_CB)} rows")
    print(f"C_H > 0.65 (fraction): {len(bad_CH)} rows")

    bad_HHVf = g[to_num(g["HHV_input"]) < 12]
    bad_HC   = g[(to_num(g["H/C"]) < 0.8) | (to_num(g["H/C"]) > 1.9)]
    bad_OC   = g[(to_num(g["O/C"]) < 0.35) | (to_num(g["O/C"]) > 0.95)]
    print(f"HHV_input < 12 MJ/kg: {len(bad_HHVf)} rows")
    print(f"H/C outside ~[0.8,1.9]: {len(bad_HC)} rows")
    print(f"O/C outside ~[0.35,0.95]: {len(bad_OC)} rows")

    def _peek(df_, cols, n=5, tag=""):
        if len(df_) == 0:
            return
        cols = [c for c in cols if c in g.columns]
        print(f"\n-- {tag} (showing {min(n,len(df_))} of {len(df_)}) --")
        print(df_[cols].head(n))

    show_cols = ["Feedstock","proc","T","t","B_Y","C_Y","A_Y","G_Y",
                 "HHV_biooil","HHV_biochar","E_B","E_H","C_B","C_H",
                 "HHV_input","O/C","H/C","Ref","Source_Figure"]

    _peek(bad_EH, show_cols, tag=f"E_H > {EH_hi_default}")
    _peek(bad_EB, show_cols, tag=f"E_B > {EB_hi_default}")
    _peek(bad_BY, show_cols, tag="B_Y out of [0,100]")
    _peek(bad_CB, show_cols, tag="C_B > 0.60")
    _peek(bad_HHVf, show_cols, tag="HHV_input < 12")
