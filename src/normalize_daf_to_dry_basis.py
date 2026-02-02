YIELD_COLS = ["B_Y", "C_Y", "A_Y", "G_Y", "GW_Y"]

def normalize_daf_to_dry_basis(df, yield_cols=YIELD_COLS):
    df = df.copy()

    ash_effective = df["Ash"].copy()
    if "Ash_imputed" in df.columns:
        ash_effective = ash_effective.fillna(df["Ash_imputed"])

    mask_daf = df["basis_class"] == "daf"
    mask_ash = ash_effective.notna()
    mask = mask_daf & mask_ash

    if not mask.any():
        print("No daf rows with valid Ash found – nothing to normalize.")
        return df

    ash_frac = ash_effective[mask] / 100.0
    factor = (1.0 - ash_frac)

    bad = (factor <= 0) | (factor > 1)
    if bad.any():
        print(f"⚠️ Skipping {bad.sum()} daf rows with non-physical ash fractions.")
        factor = factor[~bad]
        mask.loc[mask] = ~bad.values  # tighten mask to good rows only

    print(f"Normalizing {mask.sum()} daf rows to dry basis...")

    for col in yield_cols:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col] * factor

    df.loc[mask, "basis_class"] = "dry_basis"
    if "Basis_daf" in df.columns:
        df.loc[mask, "Basis_daf"] = 0
    if "Basis_dry_basis" in df.columns:
        df.loc[mask, "Basis_dry_basis"] = 1

    if "Yield_fill_method" in df.columns:
        df.loc[mask, "Yield_fill_method"] = df.loc[mask, "Yield_fill_method"].fillna("") + ";daf_to_dry"

    return df
