"""
Data Preparation Module for HTT ML Pipeline
===========================================

This module handles:
1. Exporting raw data before any drops (for test notebooks)
2. Building engineered features for training
3. Exporting engineered features with metadata

Usage in notebook:
    from data_preparation import export_raw_data, build_training_features
    
    # Export raw data first
    export_raw_data(htt_data, export_dir='Testing')
    
    # Build features for training
    X, Y, df_Xok = build_training_features(
        htt_data, 
        CAND_X_NUM=..., 
        CAND_X_SOLV_NUM=...,
        CAND_X_SOLV_FLAG=...,
        COLS_Y=...,
        export_dir='Testing'
    )
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple


def export_raw_data(
    htt_data: pd.DataFrame, 
    export_dir: str = 'Testing'
) -> None:
    """
    Export raw dataset BEFORE any drops or feature engineering.

    Creates CSV files for test notebooks:
    - metadata_full_before_drop.csv
    - numeric_features_full_before_drop.csv
    - targets_full_before_drop.csv
    - categorical_features_full_before_drop.csv
    """
    os.makedirs(export_dir, exist_ok=True)
    
    print("=" * 80)
    print("EXPORTING RAW DATASET BEFORE DROP (WITH METADATA)")
    print("=" * 80)
    
    paper_title_col = "Paper_Title" if "Paper_Title" in htt_data.columns else "paper_title"
    
    metadata_cols = [
        "DOI",
        "year",
        "Feedstock",
        "Family",
        "Family_std",
        "Tier",
        "T",
        "t",
        "process_type",
        "Process_type",
        "process_subtype",
        "process_subtype_raw",
        paper_title_col,
        "Ref",
    ]
    existing_metadata_cols = [c for c in metadata_cols if c in htt_data.columns]
    
    if existing_metadata_cols:
        metadata_export = htt_data[existing_metadata_cols].copy()
        metadata_path = os.path.join(export_dir, "metadata_full_before_drop.csv")
        metadata_export.to_csv(metadata_path, index=True)
        print(f"\n✅ Metadata exported:")
        print(f"   File: {metadata_path}")
        print(f"   Rows: {len(metadata_export)}")
        print(f"   Columns: {list(metadata_export.columns)}")
    else:
        print("\n⚠️ No metadata columns found!")
    
    numeric_feature_cols = [
        "O/C", "H/C", "Ash", "T", "t", "IC",
        "HHV_feedstock", "LRI", "LRI_imputed",
        "C", "H", "O", "N", "S",
        "Lignin", "cellulose", "hemicellulose",
        "pressure_effective_mpa",
        "pressure_autogenic",
        "catalyst_biomass_ratio",
        "water_biomass_ratio",
        "Moisture_min", "Moisture_max",
    ]
    
    metadata_for_filtering = ["DOI", paper_title_col, "Feedstock", "Ref"]
    all_export_cols = numeric_feature_cols + metadata_for_filtering
    existing_export = [c for c in all_export_cols if c in htt_data.columns]
    
    if existing_export:
        existing_export = list(dict.fromkeys(existing_export))
        numeric_export = htt_data[existing_export].copy()
        numeric_path = os.path.join(export_dir, "numeric_features_full_before_drop.csv")
        numeric_export.to_csv(numeric_path, index=True)
        
        print(f"\n✅ Numeric features + metadata exported:")
        print(f"   File: {numeric_path}")
        print(f"   Rows: {len(numeric_export)}")
        print(f"   Columns: {len(numeric_export.columns)}")
        
        metadata_included = [c for c in metadata_for_filtering if c in existing_export]
        if metadata_included:
            print(f"   📋 Metadata columns: {metadata_included}")
        
        if "LRI_imputed" in numeric_export.columns:
            non_zero = (numeric_export["LRI_imputed"] != 0.0).sum()
            print(f"   📊 LRI_imputed non-zero values: {non_zero}")
    else:
        print("\n⚠️ No numeric features found!")
    
    target_cols = [
        "B_Y", "C_Y", "A_Y", "G_Y", "GW_Y",
        "E_B", "E_H", "C_B", "C_H",
        "HHV_biooil", "C_biooil", "O_biooil", "H_biooil", "N_biooil", "S_biooil",
        "HHV_biochar", "C_biochar", "O_biochar", "H_biochar", "N_biochar", "S_biochar",
        "O/C_biooil", "H/C_biooil", "O/C_biochar", "H/C_biochar",
    ]
    existing_targets = [c for c in target_cols if c in htt_data.columns]
    
    if existing_targets:
        targets_export = htt_data[existing_targets].copy()
        targets_path = os.path.join(export_dir, "targets_full_before_drop.csv")
        targets_export.to_csv(targets_path, index=True)
        print(f"\n✅ Targets exported: {len(targets_export)} rows × {len(targets_export.columns)} cols")
    else:
        print("\n⚠️ No target columns found!")
    
    categorical_cols = [
        "Family", "Family_std", "Tier",
        "process_type", "Process_type",
        "process_subtype", "process_subtype_raw",
        "catalyst",
        "solvent_or_medium", "solvent_family",
        "VK_cluster_k2",
        # basis / QA
        "basis_class",
        "basis_is_carbon", "basis_is_daf", "basis_is_dry",
        "Basis_carbon_basis", "Basis_daf", "Basis_dry_basis",
        "Basis_other", "Basis_unknown",
        "QA_flag",
        # solvent numeric + flags (kept here for convenience)
        "frac_water", "frac_ethanol", "frac_methanol", "frac_isopropanol",
        "frac_acetone", "frac_glycerol",
        "acid_flag", "base_flag", "H_donor_flag",
        "acid_M", "base_M", "phenol_additive_wt_pct",
    ]
    
    cat_cols   = [c for c in htt_data.columns if c.startswith("Cat_")]
    solv_cols  = [c for c in htt_data.columns if c.startswith("Solv_")]
    basis_cols = [c for c in htt_data.columns if c.startswith("Basis_")]
    categorical_cols.extend(cat_cols + solv_cols + basis_cols)
    
    existing_categorical = [c for c in categorical_cols if c in htt_data.columns]
    
    if existing_categorical:
        existing_categorical = list(dict.fromkeys(existing_categorical))
        categorical_export = htt_data[existing_categorical].copy()
        categorical_path = os.path.join(export_dir, "categorical_features_full_before_drop.csv")
        categorical_export.to_csv(categorical_path, index=True)
        print(f"\n✅ Categorical+encoding features exported: "
              f"{len(categorical_export)} rows × {len(categorical_export.columns)} cols")
    else:
        print("\n⚠️ No categorical features found!")
    
    print(f"\n{'=' * 80}")
    print(f"RAW DATA EXPORT COMPLETE: {len(htt_data)} rows")
    print(f"{'=' * 80}\n")


from typing import List, Tuple
import os
import pandas as pd

def build_training_features(
    htt_data: pd.DataFrame,
    CAND_X_NUM: List[str],
    CAND_X_SOLV_NUM: List[str],
    CAND_X_SOLV_FLAG: List[str],
    COLS_Y: List[str],
    USE_TIER: bool = False,
    USE_FAMILY: bool = True,
    USE_BASIS: bool = True,
    export_dir: str = "Testing",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build engineered features for training and export them.
    """
    print("\n" + "=" * 80)
    print("BUILDING TRAINING FEATURES")
    print("=" * 80)

    df_all = htt_data.copy()

    # One-hot columns present in data
    SOLV_DUM_COLS  = sorted([c for c in df_all.columns if c.startswith("Solv_")])
    BASIS_DUM_COLS = sorted([c for c in df_all.columns if c.startswith("Basis_")])

    if SOLV_DUM_COLS:
        print(f"✓ Found {len(SOLV_DUM_COLS)} solvent one-hot columns:")
        print(f"   {SOLV_DUM_COLS}")
    else:
        print("⚠️ No Solv_* columns found in df_all – solvent families will not be encoded.")

    if USE_BASIS and BASIS_DUM_COLS:
        print(f"✓ Found {len(BASIS_DUM_COLS)} Basis_* columns for yield-basis encoding.")

    # ----- Tier (optional) -----
    if USE_TIER and "Tier" in df_all.columns:
        tier_vals = df_all["Tier"]
        tier_cats = pd.Index([v for v in pd.unique(tier_vals) if pd.notna(v)])
        df_all["Tier"] = pd.Categorical(tier_vals, categories=tier_cats, ordered=False)
        tier_dum = pd.get_dummies(df_all["Tier"], prefix="Tier", dummy_na=True)
    else:
        tier_dum = pd.DataFrame(index=df_all.index)

    # ----- process_subtype -----
    if "process_subtype" in df_all.columns:
        sub_vals = df_all["process_subtype"]
        sub_cats = pd.Index([v for v in pd.unique(sub_vals) if pd.notna(v)])
        df_all["process_subtype"] = pd.Categorical(sub_vals, categories=sub_cats, ordered=False)
        subtype_dum = pd.get_dummies(df_all["process_subtype"], prefix="Subtype", dummy_na=True)
    else:
        subtype_dum = pd.DataFrame(index=df_all.index)

    # ----- Family (raw) -----
    if "Family" in df_all.columns:
        fam_vals = df_all["Family"]
        fam_cats = pd.Index([v for v in pd.unique(fam_vals) if pd.notna(v)])
        df_all["Family"] = pd.Categorical(fam_vals, categories=fam_cats, ordered=False)
        family_dum = pd.get_dummies(df_all["Family"], prefix="Family", dummy_na=True)
    else:
        family_dum = pd.DataFrame(index=df_all.index)

    # ----- Family_std (cleaned labels) - use this as the canonical Family encoding -----
    # Use Family_std (standardized) if available, otherwise fallback to raw Family
    # We'll create one-hot columns with "Family_" prefix (not "FamilyStd_") for simplicity
    if "Family_std" in df_all.columns:
        fams_vals = df_all["Family_std"]
        fams_cats = pd.Index([v for v in pd.unique(fams_vals) if pd.notna(v)])
        df_all["Family_std"] = pd.Categorical(fams_vals, categories=fams_cats, ordered=False)
        family_std_dum = pd.get_dummies(df_all["Family_std"], prefix="Family", dummy_na=True)
    else:
        family_std_dum = pd.DataFrame(index=df_all.index)

    # ----- Filter rows with complete core numeric features -----
    num_cols       = [c for c in CAND_X_NUM       if c in df_all.columns]
    solv_num_cols  = [c for c in CAND_X_SOLV_NUM  if c in df_all.columns]
    solv_flag_cols = [c for c in CAND_X_SOLV_FLAG if c in df_all.columns]

    df_Xok = df_all.dropna(subset=num_cols).copy()
    print(f"✓ Rows with complete core numeric features: {len(df_Xok)} / {len(df_all)}")

    # ----- Build X blocks -----
    X_blocks = []

    # Core numeric
    X_blocks.append(df_Xok[num_cols].astype(float))

    # Tier / subtype / family_std (use Family_std only, not raw Family)
    if not tier_dum.empty:
        X_blocks.append(tier_dum.loc[df_Xok.index].astype(float))
    if not subtype_dum.empty:
        X_blocks.append(subtype_dum.loc[df_Xok.index].astype(float))
    
    # Family encoding (optional)
    if USE_FAMILY:
        # Use Family_std (cleaned) instead of raw Family to avoid duplicates
        if not family_std_dum.empty:
            X_blocks.append(family_std_dum.loc[df_Xok.index].astype(float))
        elif not family_dum.empty:
            # Fallback to raw Family if Family_std not available
            X_blocks.append(family_dum.loc[df_Xok.index].astype(float))

    # Solvent one-hots
    if SOLV_DUM_COLS:
        X_blocks.append(
            df_all.loc[df_Xok.index, SOLV_DUM_COLS].fillna(0.0).astype(float)
        )

    # Yield-basis one-hots
    if USE_BASIS and BASIS_DUM_COLS:
        X_blocks.append(
            df_all.loc[df_Xok.index, BASIS_DUM_COLS].fillna(0.0).astype(float)
        )
        print(
            f"✓ Including {len(BASIS_DUM_COLS)} yield-basis one-hot features "
            f"(e.g. {BASIS_DUM_COLS[:3]}...)"
        )

    # Solvent numeric + missing flags
    for c in solv_num_cols:
        col = df_all.loc[df_Xok.index, c]
        X_blocks.append(col.fillna(0.0).astype(float).to_frame(c))
        X_blocks.append(col.isna().astype(int).to_frame(c + "_missing"))

    # Solvent flags
    for c in solv_flag_cols:
        col = df_all.loc[df_Xok.index, c]
        X_blocks.append(col.fillna(False).astype(int).to_frame(c))

    # ----- Catalyst features: all Cat_* and descriptors -----
    present_cat_cols = [c for c in df_all.columns if c.startswith("Cat_")]

    coarse_cat_classes = [
        "Cat_acid", "Cat_additive", "Cat_alkaline", "Cat_metal",
        "Cat_none", "Cat_other", "Cat_oxide", "Cat_phosphide",
        "Cat_salt", "Cat_sulfide", "Cat_support", "Cat_zeolite",
    ]

    cat_descriptor_flags = [
        "Cat_is_base", "Cat_base_strong", "Cat_base_carbonate",
        "Cat_is_acid", "Cat_acid_strong", "Cat_acid_weak",
        "Cat_is_redox", "Cat_has_noble", "Cat_has_base_metal",
        "Cat_bimetallic", "Cat_is_zeolite", "Cat_is_support",
        "Cat_is_hdonor",
    ]

    all_cat_cols = coarse_cat_classes + cat_descriptor_flags
    cat_cols_to_include = [c for c in all_cat_cols if c in present_cat_cols]

    additional_cat = [c for c in present_cat_cols if c not in cat_cols_to_include]
    cat_cols_to_include.extend(additional_cat)

    if cat_cols_to_include:
        X_blocks.append(
            df_all.loc[df_Xok.index, cat_cols_to_include].fillna(0.0).astype(float)
        )
        print(
            f"✓ Including {len(cat_cols_to_include)} catalyst features "
            f"(coarse classes + descriptor flags)"
        )

    # Catalyst ratio log10 (separate numeric, if present)
    if "cat_ratio_log10" in df_all.columns:
        X_blocks.append(
            df_all.loc[df_Xok.index, ["cat_ratio_log10"]].fillna(0.0).astype(float)
        )

    # Interaction features (INT_*)
    interaction_cols = [c for c in df_all.columns if c.startswith("INT_")]
    if interaction_cols:
        X_blocks.append(
            df_all.loc[df_Xok.index, interaction_cols].fillna(0.0).astype(float)
        )
        print(f"✓ Including {len(interaction_cols)} interaction features")

    # Assemble X
    X = pd.concat(X_blocks, axis=1)
    
    # Deduplicate columns (remove any duplicate column names)
    if X.columns.duplicated().any():
        n_before = len(X.columns)
        X = X.loc[:, ~X.columns.duplicated()]
        n_after = len(X.columns)
        print(f"⚠️ Removed {n_before - n_after} duplicate columns")
    
    print(f"✓ Engineered X shape: {X.shape}")

    # ----- Build Y -----
    Y_cols_present = [c for c in COLS_Y if c in df_Xok.columns]
    Y = df_Xok[Y_cols_present].copy()
    print(f"✓ Target Y shape: {Y.shape} (targets only)")

    # Add grouping columns
    if "DOI" in df_Xok.columns:
        Y["DOI"] = df_Xok["DOI"]
    if "Ref" in df_Xok.columns:
        Y["Ref"] = df_Xok["Ref"]
    if "Feedstock" in df_Xok.columns:
        Y["Feedstock"] = df_Xok["Feedstock"]
    if "T" in df_Xok.columns:
        Y["T"] = df_Xok["T"]

    metadata_added = [c for c in ["DOI", "Ref", "Feedstock", "T"] if c in Y.columns]
    print(
        f"✓ Y with grouping columns: {Y.shape} "
        f"(targets + {metadata_added} for grouping/filtering)"
    )

    # ----- Export engineered features -----
    _export_engineered_features(X, Y, df_Xok, export_dir)

    return X, Y, df_Xok



def _export_engineered_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    df_Xok: pd.DataFrame,
    export_dir: str,
) -> None:
    """
    Export engineered features with metadata for test notebooks.
    """
    if export_dir is None:
        print("\n⏭️  Skipping feature export (export_dir=None)")
        return
    
    os.makedirs(export_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("EXPORTING ENGINEERED FEATURES (AFTER FEATURE BUILDING)")
    print("=" * 80)
    
    paper_title_col = 'Paper_Title' if 'Paper_Title' in df_Xok.columns else 'paper_title'
    metadata_cols_export = ['DOI', paper_title_col, 'Feedstock', 'T', 'Ref']
    metadata_available = [c for c in metadata_cols_export if c in df_Xok.columns]
    
    if metadata_available:
        metadata_for_export = df_Xok[metadata_available].copy()
        X_with_metadata = pd.concat([X, metadata_for_export], axis=1)
        
        X_export_path = os.path.join(export_dir, 'X_engineered_with_metadata.csv')
        X_with_metadata.to_csv(X_export_path, index=True)
        
        print(f'\n✅ Engineered features + metadata exported:')
        print(f'   File: {X_export_path}')
        print(f'   Rows: {len(X_with_metadata)}')
        print(f'   Feature columns: {X.shape[1]} (engineered)')
        print(f'   Metadata columns: {len(metadata_available)} ({metadata_available})')
        
        subtype_feats = [c for c in X.columns if c.startswith('Subtype_')]
        family_feats = [c for c in X.columns if c.startswith('Family_')]
        solv_feats = [c for c in X.columns if c.startswith('Solv_')]
        print(f'\n   📊 One-hot features included:')
        print(f'      Subtype: {len(subtype_feats)} dummies')
        print(f'      Family:  {len(family_feats)} dummies')
        print(f'      Solvent: {len(solv_feats)} dummies')
    else:
        X_export_path = os.path.join(export_dir, 'X_engineered_with_metadata.csv')
        X.to_csv(X_export_path, index=True)
        print(f'\n✅ Engineered features exported (no metadata):')
        print(f'   File: {X_export_path}')
        print(f'   Shape: {X.shape}')
    
    Y_export_path = os.path.join(export_dir, 'Y_targets.csv')
    Y.to_csv(Y_export_path, index=True)
    print(f'\n✅ Targets exported:')
    print(f'   File: {Y_export_path}')
    print(f'   Shape: {Y.shape}')
    
    print(f'\n{"=" * 80}')
    print(f'ENGINEERED FEATURES EXPORT COMPLETE: {len(X)} rows')
    print(f'{"=" * 80}\n')
