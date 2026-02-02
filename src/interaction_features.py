import pandas as pd
import numpy as np


INTERACTION_COLS = []


def reset_interactions():
    global INTERACTION_COLS
    INTERACTION_COLS = []


def _add_int(df: pd.DataFrame, new_name: str, col_a: str, col_b: str) -> None:
    """
    Create an interaction feature and an explicit missing flag.

    new_name              = df[col_a] * df[col_b]   (NaNs → 0.0)
    new_name + '_missing' = 1 if col_a or col_b is NaN, else 0
    """
    if col_a not in df.columns or col_b not in df.columns:
        return

    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    miss_flag = (a.isna() | b.isna()).astype(int)
    df[new_name + "_missing"] = miss_flag
    df[new_name] = a.fillna(0.0) * b.fillna(0.0)

    INTERACTION_COLS.append(new_name)
    INTERACTION_COLS.append(new_name + "_missing")


CORE_INTERACTIONS = [
    ("INT_T_IC", "T", "IC"),
    ("INT_T_pressure", "T", "pressure_effective_mpa"),
    ("INT_IC_cat_ratio", "IC", "catalyst_biomass_ratio"),
    ("INT_T_Lignin", "T", "Lignin"),
    ("INT_T_cellulose", "T", "cellulose"),
    ("INT_T_hemicellulose", "T", "hemicellulose"),
    ("INT_T_H_over_C", "T", "H/C"),
    ("INT_T_O_over_C", "T", "O/C"),
    ("INT_T_LRI", "T", "LRI"),
    ("INT_T_cat_ratio", "T", "catalyst_biomass_ratio"),
]

ELEMENTAL_INTERACTIONS = [
    ("INT_T_C", "T", "C"),
    ("INT_T_H", "T", "H"),
    ("INT_T_O", "T", "O"),
    ("INT_T_Ash", "T", "Ash"),
    ("INT_Ash_Lignin", "Ash", "Lignin"),
    ("INT_Ash_LRI", "Ash", "LRI"),
    ("INT_Ash_C", "Ash", "C"),
]


def add_interaction_features(df: pd.DataFrame, use_elemental: bool = True) -> list:
    """
    Add interaction features to dataframe.
    
    Args:
        df: Input dataframe
        use_elemental: Whether to include elemental interactions
    
    Returns:
        List of interaction column names added
    """
    reset_interactions()
    
    interactions = CORE_INTERACTIONS.copy()
    if use_elemental:
        interactions.extend(ELEMENTAL_INTERACTIONS)
    
    for new_name, col_a, col_b in interactions:
        _add_int(df, new_name, col_a, col_b)
    
    return INTERACTION_COLS.copy()
