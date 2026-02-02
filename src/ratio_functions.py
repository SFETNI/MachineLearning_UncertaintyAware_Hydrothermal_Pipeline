"""
Utility functions for computing atomic ratios on-the-fly during RSM plotting.
"""

import sys
from pathlib import Path


testing_dir = Path(__file__).parent
if str(testing_dir) not in sys.path:
    sys.path.insert(0, str(testing_dir))


def compute_atomic_ratio(Y, numerator_elem, denominator_elem, phase):
    """
    Compute atomic ratio (e.g., O/C, H/C) from elemental weight percentages.
    
    Args:
        Y: DataFrame with elemental columns
        numerator_elem: 'O', 'H', etc.
        denominator_elem: 'C', 'N', etc.
        phase: 'biooil' or 'biochar'
    
    Returns:
        Series with atomic ratio values
    """
    # Atomic weights
    atomic_weights = {
        'C': 12.011,
        'H': 1.008,
        'O': 15.999,
        'N': 14.007,
        'S': 32.06
    }
    
    num_col = f"{numerator_elem}_{phase}"
    den_col = f"{denominator_elem}_{phase}"
    
    if num_col not in Y.columns or den_col not in Y.columns:
        raise KeyError(f"Missing columns: {num_col} or {den_col}")
    
   
    # (wt% numerator / AW_numerator) / (wt% denominator / AW_denominator)
    num_wt = Y[num_col]
    den_wt = Y[den_col]
    
    ratio = (num_wt / atomic_weights[numerator_elem]) / (den_wt / atomic_weights[denominator_elem])
    
    return ratio


def run_rsm_block_with_ratios(
    target,
    X,
    Y,
    models,
    run_rsm_block_func,  # Must pass the run_rsm_block function from notebook
    x_col="T",
    y_col="t",
    group_cols=("DOI", "Feedstock"),
    min_points=6,
    min_x_span=10.0,
    min_y_span=10.0,
    k=10,
    degree=2,
    max_plots=20,
    make_residual_plots=True,
    dpi=300,
    elev=30,
    azim=-120,
    surface_cmap="viridis",
    residual_mode="halo",
    residual_cmap="bwr",
    show_residual_colorbar=True,
    candidates_df=None,
    **extra_kwargs,
):
    """
    Enhanced wrapper that can handle atomic ratio targets (O/C, H/C) 
    by computing them on-the-fly from elemental compositions.
    
    If target is a ratio like 'O/C_biooil', it will:
    1. Compute the ratio from O_biooil and C_biooil
    2. Create a synthetic model that predicts the ratio from elemental predictions
    3. Add the ratio column to Y temporarily for plotting
    
    Args:
        run_rsm_block_func: The run_rsm_block function from the notebook
        ... (other args same as run_rsm_block)
    """
  
    ratio_patterns = {
        'O/C_biooil': ('O', 'C', 'biooil'),
        'H/C_biooil': ('H', 'C', 'biooil'),
        'O/C_biochar': ('O', 'C', 'biochar'),
        'H/C_biochar': ('H', 'C', 'biochar'),
    }
    
    if target in ratio_patterns:
        num_elem, den_elem, phase = ratio_patterns[target]
        
        print(f"ℹ️  Target '{target}' is a ratio - computing from {num_elem}_{phase} and {den_elem}_{phase}")
        
       
        if target not in Y.columns:
          
            Y = Y.copy()
            Y[target] = compute_atomic_ratio(Y, num_elem, den_elem, phase)
            print(f"✓ Computed {target} from elemental data")
        
      
        if target not in models:
       
            num_col = f"{num_elem}_{phase}"
            den_col = f"{den_elem}_{phase}"
            
            if num_col not in models or den_col not in models:
                raise KeyError(
                    f"Cannot predict {target}: missing models for {num_col} or {den_col}"
                )
            
            #
            class RatioModel:
                def __init__(self, num_model, den_model, num_elem, den_elem):
                    self.num_model = num_model
                    self.den_model = den_model
                    self.num_elem = num_elem
                    self.den_elem = den_elem
                    
             
                    self.atomic_weights = {
                        'C': 12.011,
                        'H': 1.008,
                        'O': 15.999,
                        'N': 14.007,
                        'S': 32.06
                    }
                
                def predict(self, X):
                    num_pred = self.num_model.predict(X)
                    den_pred = self.den_model.predict(X)
                    
                   
                    ratio = (num_pred / self.atomic_weights[self.num_elem]) / \
                            (den_pred / self.atomic_weights[self.den_elem])
                    
                    return ratio
            
    
            models = models.copy()
            models[target] = RatioModel(
                models[num_col],
                models[den_col],
                num_elem,
                den_elem
            )
            print(f"✓ Created synthetic model for {target}")
    
   
    return run_rsm_block_func(
        target=target,
        X=X,
        Y=Y,
        models=models,
        x_col=x_col,
        y_col=y_col,
        group_cols=group_cols,
        min_points=min_points,
        min_x_span=min_x_span,
        min_y_span=min_y_span,
        k=k,
        degree=degree,
        max_plots=max_plots,
        make_residual_plots=make_residual_plots,
        dpi=dpi,
        elev=elev,
        azim=azim,
        surface_cmap=surface_cmap,
        residual_mode=residual_mode,
        residual_cmap=residual_cmap,
        show_residual_colorbar=show_residual_colorbar,
        candidates_df=candidates_df,
        **extra_kwargs,
    )
