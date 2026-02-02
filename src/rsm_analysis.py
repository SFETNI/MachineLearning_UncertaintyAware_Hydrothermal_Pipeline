"""
Response Surface Methodology (RSM) Analysis Module

Functions for local quadratic response surface reconstruction around
experimental blocks in hydrothermal treatment predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


RSM_TARGET_META = {
    "B_Y": {"units": "% w/w", "pretty_name": "Bio-oil yield"},
    "C_Y": {"units": "% w/w", "pretty_name": "Char yield"},
    "A_Y": {"units": "% w/w", "pretty_name": "Aqueous / other yield"},
    "G_Y": {"units": "% w/w", "pretty_name": "Gas yield"},
    "GW_Y": {"units": "% w/w", "pretty_name": "Gas + water yield"},
    "E_B": {"units": "%", "pretty_name": "Bio-oil energy yield"},
    "E_H": {"units": "%", "pretty_name": "Char energy yield"},
    "C_B": {"units": "%", "pretty_name": "Bio-oil carbon yield"},
    "C_H": {"units": "%", "pretty_name": "Char carbon yield"},
    "HHV_biooil": {"units": "MJ/kg", "pretty_name": "Bio-oil HHV"},
    "HHV_biochar": {"units": "MJ/kg", "pretty_name": "Biochar HHV"},
    "O/C_biooil": {"units": "mol/mol", "pretty_name": "O/C (bio-oil)"},
    "H/C_biooil": {"units": "mol/mol", "pretty_name": "H/C (bio-oil)"},
    "O/C_biochar": {"units": "mol/mol", "pretty_name": "O/C (biochar)"},
    "H/C_biochar": {"units": "mol/mol", "pretty_name": "H/C (biochar)"},
}


def align_X_to_estimator(est, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align DataFrame columns to match estimator's feature_names_in_.
    Adds missing features (as zeros) and drops extra features.
    """
    feat = getattr(est, "feature_names_in_", None)
    if feat is None:
        return X

    feat = list(feat)
    missing = [c for c in feat if c not in X.columns]
    extra = [c for c in X.columns if c not in feat]

    if missing:
        X = X.copy()
        for c in missing:
            X[c] = 0.0

    if extra:
        X = X.drop(columns=extra)

    return X[feat]


def compute_local_rsm_surface(
    target,
    X,
    Y,
    model,
    doi,
    feedstock,
    x_col="T",
    y_col="t",
    degree=2,
    n_grid=80,
    extrapolation_factor=0.05,
    extrapolation_absolute=None,
):
    """
    Build a local quadratic RSM-style surface for one experimental block.
    
    Parameters:
    - target: Target variable name
    - X, Y: Feature and target DataFrames
    - model: Trained estimator
    - doi, feedstock: Block identifiers
    - x_col, y_col: Variables for surface axes (default T, t)
    - degree: Polynomial degree (default 2 = quadratic)
    - n_grid: Grid resolution
    - extrapolation_factor: Fractional extension beyond data range
    - extrapolation_absolute: Dict with absolute extensions
    
    Returns dict with subset, grid (XX, YY, Z), and metadata.
    """
    if target not in Y.columns:
        raise KeyError(f"{target!r} not in Y")

    src_x = Y if x_col in Y.columns else X
    src_y = Y if y_col in Y.columns else X

    mask = (
        (Y["DOI"].astype(str) == str(doi))
        & (Y["Feedstock"].astype(str) == str(feedstock))
    )

    subset = pd.DataFrame(
        {
            x_col: src_x.loc[mask, x_col],
            y_col: src_y.loc[mask, y_col],
            "y_exp": Y.loc[mask, target],
            "Feedstock": Y.loc[mask, "Feedstock"].astype(str),
        }
    ).dropna(subset=[x_col, y_col, "y_exp"])

    if subset.shape[0] < 6:
        raise ValueError(
            f"Too few points for block (DOI={doi}, Feedstock={feedstock}), "
            f"n={subset.shape[0]}"
        )

    X_sub = X.loc[subset.index]
    X_sub_aligned = align_X_to_estimator(model, X_sub)
    y_rf = model.predict(X_sub_aligned)
    subset["y_rf"] = y_rf
    subset["residual"] = subset["y_exp"] - subset["y_rf"]

    X_vals = subset[x_col].values
    Y_vals = subset[y_col].values

    XY = np.column_stack([X_vals, Y_vals])
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(XY)

    reg = LinearRegression()
    reg.fit(Phi, subset["y_rf"].values)

    X_min, X_max = X_vals.min(), X_vals.max()
    Y_min, Y_max = Y_vals.min(), Y_vals.max()

    if extrapolation_absolute is not None:
        X_pad = extrapolation_absolute.get(x_col, 0.05 * (X_max - X_min))
        Y_pad = extrapolation_absolute.get(y_col, 0.05 * (Y_max - Y_min))
    else:
        X_pad = extrapolation_factor * (X_max - X_min)
        Y_pad = extrapolation_factor * (Y_max - Y_min)

    X_grid = np.linspace(X_min - X_pad, X_max + X_pad, n_grid)
    Y_grid = np.linspace(Y_min - Y_pad, Y_max + Y_pad, n_grid)
    XX, YY = np.meshgrid(X_grid, Y_grid)

    grid_xy = np.column_stack([XX.ravel(), YY.ravel()])
    Phi_grid = poly.transform(grid_xy)
    Z = reg.predict(Phi_grid).reshape(XX.shape)

    target_lower = target.lower()
    if '_y' in target_lower or 'yield' in target_lower:
        Z = np.clip(Z, 0, 100)
    elif 'hhv' in target_lower:
        Z = np.clip(Z, 0, 50)
    elif 'o/c' in target_lower or 'h/c' in target_lower:
        Z = np.clip(Z, 0, 3)
    elif any(elem in target_lower for elem in ['_c_', '_h_', '_o_', '_n_', '_s_']):
        Z = np.clip(Z, 0, 100)
    else:
        Z = np.maximum(Z, 0)

    flat_idx = np.argmax(Z)
    ix_opt, iy_opt = np.unravel_index(flat_idx, Z.shape)
    X_opt = XX[ix_opt, iy_opt]
    Y_opt = YY[ix_opt, iy_opt]
    z_opt = Z[ix_opt, iy_opt]

    rmse = float(np.sqrt(mean_squared_error(subset["y_exp"], subset["y_rf"])))
    r2 = float(r2_score(subset["y_exp"], subset["y_rf"]))

    span_x = X_max - X_min if X_max > X_min else 1.0
    span_y = Y_max - Y_min if Y_max > Y_min else 1.0
    dist_edge = min(
        (X_opt - X_min) / span_x,
        (X_max - X_opt) / span_x,
        (Y_opt - Y_min) / span_y,
        (Y_max - Y_opt) / span_y,
    )
    opt_location = "interior" if dist_edge > 0.15 else "edge"

    meta = {
        "doi": str(doi),
        "feedstock": str(feedstock),
        "n_points": int(subset.shape[0]),
        "x_col": x_col,
        "y_col": y_col,
        "X_min": float(X_min),
        "X_max": float(X_max),
        "Y_min": float(Y_min),
        "Y_max": float(Y_max),
        "z_min": float(Z.min()),
        "z_max": float(Z.max()),
        "X_opt": float(X_opt),
        "Y_opt": float(Y_opt),
        "z_opt": float(z_opt),
        "rmse": rmse,
        "r2": r2,
        "opt_location": opt_location,
    }

    return {
        "subset_idx": subset.index,
        "subset": subset,
        "XX": XX,
        "YY": YY,
        "Z": Z,
        "meta": meta,
    }


def find_rsm_candidates(
    X,
    Y,
    target,
    x_col="T",
    y_col="t",
    group_cols=("DOI", "Feedstock"),
    min_points=6,
    min_x_span=10.0,
    min_y_span=10.0,
):
    """
    Scan for experimental blocks with sufficient points and variable spread.
    """
    src_x = X if x_col in X.columns else Y
    src_y = X if y_col in X.columns else Y

    cols_needed = list(group_cols) + [target]
    missing_y = [c for c in cols_needed if c not in Y.columns]
    if missing_y:
        raise KeyError(f"Missing in Y: {missing_y}")

    if x_col not in src_x.columns:
        raise KeyError(f"{x_col!r} not in X or Y")
    if y_col not in src_y.columns:
        raise KeyError(f"{y_col!r} not in X or Y")

    df = pd.DataFrame({
        **{gc: Y[gc] for gc in group_cols},
        x_col: src_x[x_col],
        y_col: src_y[y_col],
        target: Y[target],
    }).dropna()

    def _agg(gr):
        n = gr.shape[0]
        x_min, x_max = gr[x_col].min(), gr[x_col].max()
        y_min, y_max = gr[y_col].min(), gr[y_col].max()
        z_min, z_max = gr[target].min(), gr[target].max()
        return pd.Series({
            "n_points": n,
            f"{x_col}_min": x_min,
            f"{x_col}_max": x_max,
            f"{x_col}_span": x_max - x_min,
            f"{y_col}_min": y_min,
            f"{y_col}_max": y_max,
            f"{y_col}_span": y_max - y_min,
            f"{target}_min": z_min,
            f"{target}_max": z_max,
            f"{target}_span": z_max - z_min,
        })

    summary = df.groupby(list(group_cols)).apply(_agg, include_groups=False).reset_index()

    summary = summary[
        (summary["n_points"] >= min_points)
        & (summary[f"{x_col}_span"] >= min_x_span)
        & (summary[f"{y_col}_span"] >= min_y_span)
    ].copy()

    summary = summary.sort_values(
        by=[f"{target}_span", "n_points"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary


def get_rsm_meta(target):
    """
    Return units and pretty name for target.
    """
    if target in RSM_TARGET_META:
        return RSM_TARGET_META[target]

    elems = {"C": "Carbon", "H": "Hydrogen", "O": "Oxygen",
             "N": "Nitrogen", "S": "Sulfur"}

    if "_" in target:
        head, phase = target.split("_", 1)

        if head in elems and phase in ("biooil", "biochar"):
            phase_label = "bio-oil" if phase == "biooil" else "biochar"
            return {
                "units": "wt %",
                "pretty_name": f"{elems[head]} in {phase_label} (wt%)",
            }

        if "/" in head and phase in ("biooil", "biochar"):
            phase_label = "bio-oil" if phase == "biooil" else "biochar"
            return {
                "units": "mol/mol",
                "pretty_name": f"{head} ({phase_label})",
            }

    return {"units": "", "pretty_name": target}


def _axis_label(name):
    """Format axis label."""
    if name == "T":
        return "Temperature (°C)"
    if name == "t":
        return "Time (min)"
    if name == "IC":
        return "Initial concentration (wt%)"
    return name


class HandlerColormapCircle(HandlerBase):
    """Legend handler for filled disc with radial colormap gradient."""
    def __init__(self, cmap, n_steps=32, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        self.n_steps = n_steps

    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        artists = []
        r_max = 0.45 * min(width, height)
        cx = xdescent + 0.5 * width
        cy = ydescent + 0.5 * height

        for i in range(self.n_steps, 0, -1):
            frac = i / self.n_steps
            r = r_max * frac
            circle = mpatches.Circle(
                (cx, cy),
                radius=r,
                transform=trans,
                facecolor=self.cmap(frac),
                edgecolor="none",
            )
            artists.append(circle)

        outline = mpatches.Circle(
            (cx, cy),
            radius=r_max,
            transform=trans,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
        )
        artists.append(outline)

        return artists


class HandlerColormapHalo(HandlerBase):
    """Legend handler for hollow ring with colormap gradient."""
    def __init__(self, cmap, n_steps=32, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        self.n_steps = n_steps

    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        artists = []
        r_outer = 0.85 * min(width, height)
        r_inner_bg = 0.55 * r_outer
        r_inner_grad = 0.65 * r_outer

        cx = xdescent + 0.65 * width
        cy = ydescent + 0.65 * height

        bg = legend.get_frame().get_facecolor()
        inner = mpatches.Circle(
            (cx, cy),
            radius=r_inner_bg,
            transform=trans,
            facecolor=bg,
            edgecolor="none",
        )
        artists.append(inner)

        for i in range(self.n_steps):
            frac = (i + 0.5) / self.n_steps
            r = r_inner_grad + (r_outer - r_inner_grad) * frac
            circle = mpatches.Circle(
                (cx, cy),
                radius=r,
                transform=trans,
                facecolor="none",
                edgecolor=self.cmap(frac),
                linewidth=0.8,
            )
            artists.append(circle)

        outline = mpatches.Circle(
            (cx, cy),
            radius=r_outer,
            transform=trans,
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
        )
        artists.append(outline)

        return artists


def plot_rsm_surface(
    rsm_data,
    target,
    units="",
    pretty_name=None,
    x_col=None,
    y_col=None,
    cmap="viridis",
    elev=25,
    azim=-55,
    figsize=(6.5, 5.5),
    dpi=150,
    residual_mode="none",
    residual_cmap="coolwarm",
    show_residual_colorbar=False,
):
    """Plot 3D response surface with experimental data projection."""
    from matplotlib.lines import Line2D

    subset = rsm_data["subset"]
    XX = rsm_data["XX"]
    YY = rsm_data["YY"]
    Z = rsm_data["Z"]
    meta = rsm_data["meta"]

    x_col = x_col or meta.get("x_col", "T")
    y_col = y_col or meta.get("y_col", "t")

    label = pretty_name or target
    zlab = label + (f" ({units})" if units else "")

    X_opt = meta["X_opt"]
    Y_opt = meta["Y_opt"]
    z_opt = meta["z_opt"]
    fs_label = meta.get("feedstock", "Test subset")

    plt.rcParams.update({
        "figure.dpi": dpi,
        "axes.grid": True,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.06, right=0.80, bottom=0.16, top=0.88)

    surf = ax.plot_surface(
        XX, YY, Z,
        cmap=cmap,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        zorder=3,
    )

    z_floor = 0.0
    z_span = float(Z.max() - Z.min()) if np.isfinite(Z).all() else 1.0
    z_span = max(z_span, 1e-6)
    z_offset = 0.02 * z_span

    X_min, X_max = float(XX.min()), float(XX.max())
    Y_min, Y_max = float(YY.min()), float(YY.max())
    XX_plane, YY_plane = np.meshgrid(
        np.linspace(X_min, X_max, 2),
        np.linspace(Y_min, Y_max, 2),
    )
    ZZ_plane = np.full_like(XX_plane, z_floor)

    ax.plot_surface(
        XX_plane, YY_plane, ZZ_plane,
        color="0.9",
        alpha=0.5,
        linewidth=0,
        zorder=0,
    )

    ax.contour(
        XX, YY, Z,
        zdir="z",
        offset=z_floor + z_offset,
        cmap=cmap,
        levels=12,
        linewidths=1.5,
        alpha=0.8,
        zorder=1,
    )

    for _, row in subset.iterrows():
        ax.plot(
            [row[x_col], row[x_col]],
            [row[y_col], row[y_col]],
            [row["y_exp"], z_floor + z_offset],
            color="gray",
            linewidth=1.2,
            alpha=0.4,
            zorder=2,
        )

    scat_yield = ax.scatter(
        subset[x_col],
        subset[y_col],
        np.full(len(subset), z_floor + z_offset),
        c=subset["y_exp"],
        cmap=cmap,
        s=70,
        alpha=1.0,
        edgecolor="k",
        linewidth=0.8,
        zorder=3,
    )

    res_norm = None
    res_vals = subset["residual"].values

    if residual_mode == "halo":
        abs_max_res = float(np.abs(res_vals).max())
        if abs_max_res <= 0:
            abs_max_res = 1e-6

        res_norm = colors.Normalize(vmin=-abs_max_res, vmax=abs_max_res)
        res_cmap_obj = plt.get_cmap(residual_cmap)
        halo_colors = res_cmap_obj(res_norm(res_vals))

        ax.scatter(
            subset[x_col],
            subset[y_col],
            np.full(len(subset), z_floor + z_offset),
            s=180,
            facecolors="none",
            edgecolors=halo_colors,
            linewidths=4.0,
            zorder=4,
        )

    opt_scatter = ax.scatter(
        X_opt, Y_opt, z_opt,
        c="white",
        edgecolor="cyan",
        s=100,
        marker="*",
        zorder=5,
    )

    ax.set_xlabel(_axis_label(x_col), labelpad=8)
    ax.set_ylabel(_axis_label(y_col), labelpad=8)
    ax.set_zlabel(zlab, labelpad=8)
    ax.set_title(f"{zlab}\nFeedstock ~ {fs_label}", fontsize=8)
    ax.view_init(elev=elev, azim=azim)

    surface_cmap_obj = plt.get_cmap(cmap)

    if residual_mode == "halo":
        residual_cmap_obj = plt.get_cmap(residual_cmap)
        exp_handle = object()
        res_handle = object()
        opt_handle = Line2D(
            [], [], linestyle="",
            marker="*", markersize=10,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=1.5,
        )

        legend = ax.legend(
            [exp_handle, res_handle, opt_handle],
            ["Experiments (projected)", "Residuals (exp - RF)", "RF local optimum"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.),
            fontsize=7,
            frameon=True,
            handler_map={
                exp_handle: HandlerColormapCircle(surface_cmap_obj),
                res_handle: HandlerColormapHalo(residual_cmap_obj),
            },
        )
    else:
        exp_handle = object()
        opt_handle = Line2D(
            [], [], linestyle="",
            marker="*", markersize=10,
            markerfacecolor="none",
            markeredgecolor="cyan",
            markeredgewidth=1.5,
        )

        legend = ax.legend(
            [exp_handle, opt_handle],
            ["Experiments (projected)", "RF local maximum"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.03),
            frameon=True,
            fontsize=8,
            handler_map={
                exp_handle: HandlerColormapCircle(surface_cmap_obj),
            },
        )

    legend.get_frame().set_alpha(1)

    n_points = meta.get("n_points", len(subset))
    rmse = meta.get("rmse", float(np.sqrt(np.mean(subset["residual"] ** 2))))
    r2 = meta.get("r2", float(r2_score(subset["y_exp"], subset["y_rf"])))
    mean_res = float(subset["residual"].mean())

    metrics_text = (
        f"Model fit:\n"
        f"n = {n_points}\n"
        f"RMSE = {rmse:.2f} {units}\n"
        f"R² = {r2:.3f}\n"
        f"|Mean res.| = {abs(mean_res):.2f} {units}"
    )

    ax.text2D(
        0.95, 0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="k"),
        zorder=10,
    )

    box = ax.get_position()
    cax_yield = fig.add_axes([
        box.x1 + 0.01,
        box.y0 + 0.10 * box.height,
        0.02,
        0.80 * box.height,
    ])
    cbar_exp = fig.colorbar(scat_yield, cax=cax_yield)
    cbar_exp.set_label(zlab)
    cbar_exp.ax.tick_params(labelsize=7)

    if residual_mode == "halo" and show_residual_colorbar and res_norm is not None:
        mappable_res = cm.ScalarMappable(norm=res_norm, cmap=residual_cmap)
        mappable_res.set_array(res_vals)

        cax_res = fig.add_axes([
            box.x0 + 0.15 * box.width,
            box.y0 - 0.08,
            0.70 * box.width,
            0.03,
        ])
        cbar_res = fig.colorbar(
            mappable_res,
            cax=cax_res,
            orientation="horizontal",
        )
        cbar_res.set_label(
            f"Residual (exp - RF) ({units})" if units else "Residual (exp - RF)"
        )
        cbar_res.ax.tick_params(labelsize=7)

    ax.grid(False)
    plt.show()


def plot_residual_map(
    rsm_data,
    target,
    units="",
    pretty_name=None,
    x_col=None,
    y_col=None,
    cmap="coolwarm",
    figsize=(5.2, 4.2),
    dpi=150,
):
    """Plot 2D residual map."""
    subset = rsm_data["subset"]
    meta = rsm_data["meta"]

    x_col = x_col or meta.get("x_col", "T")
    y_col = y_col or meta.get("y_col", "t")

    label = pretty_name or target
    rlab = f"Residual (exp - RF) ({units})" if units else "Residual (exp - RF)"

    X_opt = meta["X_opt"]
    Y_opt = meta["Y_opt"]

    n_points = meta.get("n_points", int(subset.shape[0]))
    rmse = meta.get(
        "rmse",
        float(np.sqrt(np.mean((subset["y_exp"] - subset["y_rf"]) ** 2))),
    )
    ss_tot = float(np.var(subset["y_exp"], ddof=1) * (len(subset) - 1))
    ss_res = float(np.sum((subset["y_exp"] - subset["y_rf"]) ** 2))
    r2 = meta.get("r2", 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    fs_label = meta.get("feedstock", "Test subset")

    plt.rcParams.update({
        "figure.dpi": dpi,
        "axes.grid": True,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        subset[x_col],
        subset[y_col],
        c=subset["residual"],
        cmap=cmap,
        s=55,
        edgecolor="k",
    )

    ax.scatter(
        X_opt, Y_opt,
        c="white",
        s=80,
        marker="*",
        edgecolor="black",
        label="RF local optimum",
        zorder=5,
    )

    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.set_title(f"{label} residuals\nFeedstock ~ {fs_label}", fontsize=11)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(rlab)

    txt = f"n={n_points}, RMSE={rmse:.2f}, R²={r2:.2f}"
    ax.text(
        0.02, 0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()


def auto_rsm_dashboard(
    target,
    X,
    Y,
    models,
    candidates_df,
    k=5,
    x_col="T",
    y_col="t",
    degree=2,
    units="",
    pretty_name=None,
    max_plots=None,
    make_residual_plots=True,
    output_dir=None,
    dpi=150,
    elev=25,
    azim=-55,
    surface_cmap="viridis",
    residual_mode="none",
    residual_cmap="coolwarm",
    show_residual_colorbar=False,
    extrapolation_factor=0.05,
    extrapolation_absolute=None,
):
    """Generate RSM dashboard with multiple plots for top candidates."""
    if target not in models:
        raise KeyError(f"No model for target {target!r}")

    model = models[target]
    label = pretty_name or target

    if max_plots is None:
        max_plots = k

    candidates_df = candidates_df.copy()
    sort_cols = ["n_points"]
    if f"{target}_span" in candidates_df.columns:
        sort_cols.append(f"{target}_span")

    candidates_df = candidates_df.sort_values(
        by=sort_cols,
        ascending=False,
    )

    print(f"\n=== Auto 3D RSM plots for {label} ({target}) ===")
    print(f"Will try up to {max_plots} blocks")

    top_candidates = candidates_df.head(min(k, max_plots))
    if len(top_candidates) > 0:
        print(f"\nFeedstocks to plot ({len(top_candidates)} blocks):")
        for i, row in enumerate(top_candidates.iterrows(), 1):
            _, r = row
            print(f"   {i}. {r['Feedstock']} (n={r['n_points']}, DOI={r['DOI']})")
        print()

    plots = []
    n_plots = 0

    for _, row in candidates_df.head(k).iterrows():
        if n_plots >= max_plots:
            break

        doi = row["DOI"]
        feedstock = row["Feedstock"]

        try:
            rsm_data = compute_local_rsm_surface(
                target=target,
                X=X,
                Y=Y,
                model=model,
                doi=doi,
                feedstock=feedstock,
                x_col=x_col,
                y_col=y_col,
                degree=degree,
                extrapolation_factor=extrapolation_factor,
                extrapolation_absolute=extrapolation_absolute,
            )
        except Exception as e:
            print(
                f"[{n_plots+1}] Block: n={row['n_points']}, DOI={doi}, Feedstock={feedstock}\n"
                f"  Skipped due to error: {e}"
            )
            continue

        meta = rsm_data["meta"]
        subset_data = rsm_data["subset"]
        subset_idx = subset_data.index

        cat_cols = [c for c in X.columns if c.startswith('Cat_')]
        catalyst_info = "none"
        if cat_cols:
            for cat_col in cat_cols:
                if cat_col in X.columns:
                    cat_vals = X.loc[subset_idx, cat_col].values
                    if cat_vals.max() > 0:
                        catalyst_info = cat_col.replace('Cat_', '')
                        break

        solv_cols = [c for c in X.columns if c.startswith('Solv_')]
        solvent_info = "water"
        if solv_cols:
            for solv_col in solv_cols:
                if solv_col in X.columns:
                    solv_vals = X.loc[subset_idx, solv_col].values
                    if solv_vals.max() > 0:
                        solvent_info = solv_col.replace('Solv_', '')
                        break

        subprocess_info = ""
        subtype_cols = [c for c in X.columns if c.startswith('Subtype_')]
        if subtype_cols:
            for sub_col in subtype_cols:
                if sub_col in X.columns:
                    sub_vals = X.loc[subset_idx, sub_col].values
                    if sub_vals.max() > 0:
                        subprocess_info = f", {sub_col.replace('Subtype_', '')}"
                        break

        print(
            f"[{n_plots+1}] DOI={doi}, Feedstock={feedstock}, "
            f"n={meta['n_points']}, "
            f"{x_col}={meta['X_min']:.1f}-{meta['X_max']:.1f}, "
            f"{y_col}={meta['Y_min']:.1f}-{meta['Y_max']:.1f}, "
            f"catalyst={catalyst_info}, solvent={solvent_info}{subprocess_info}, "
            f"opt-> {x_col}={meta['X_opt']:.1f}, {y_col}={meta['Y_opt']:.1f}, "
            f"z={meta['z_opt']:.2f} ({meta['opt_location']})"
        )

        plot_rsm_surface(
            rsm_data,
            target=target,
            units=units,
            pretty_name=pretty_name,
            x_col=x_col,
            y_col=y_col,
            cmap=surface_cmap,
            elev=elev,
            azim=azim,
            dpi=dpi,
            residual_mode=residual_mode,
            residual_cmap=residual_cmap,
            show_residual_colorbar=show_residual_colorbar,
        )

        if make_residual_plots:
            plot_residual_map(
                rsm_data,
                target=target,
                units=units,
                pretty_name=pretty_name,
                x_col=x_col,
                y_col=y_col,
                dpi=dpi,
            )

        plots.append(rsm_data)
        n_plots += 1

    if n_plots == 0:
        print("No plots produced (all blocks failed).")
    else:
        print(f"\nProduced {n_plots} RSM blocks for {label}")

    return plots


def run_rsm_block(
    target,
    X,
    Y,
    models,
    x_col="T",
    y_col="t",
    group_cols=("DOI", "Feedstock"),
    min_points=8,
    min_x_span=10.0,
    min_y_span=10.0,
    k=10,
    degree=2,
    extrapolation_factor=0.05,
    extrapolation_absolute=None,
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
    Convenience wrapper for RSM analysis workflow.
    """
    if target not in models:
        raise KeyError(f"No model for target {target!r}")

    meta = get_rsm_meta(target)
    units = meta["units"]
    pretty_name = meta["pretty_name"]

    if candidates_df is None:
        candidates_df = find_rsm_candidates(
            X, Y,
            target=target,
            x_col=x_col,
            y_col=y_col,
            group_cols=group_cols,
            min_points=min_points,
            min_x_span=min_x_span,
            min_y_span=min_y_span,
        )

    print(f"\n>>> RSM dashboard for {pretty_name} ({target})")

    summary = auto_rsm_dashboard(
        target=target,
        X=X,
        Y=Y,
        models=models,
        candidates_df=candidates_df,
        k=k,
        x_col=x_col,
        y_col=y_col,
        degree=degree,
        units=units,
        pretty_name=pretty_name,
        max_plots=max_plots,
        make_residual_plots=make_residual_plots,
        dpi=dpi,
        elev=elev,
        azim=azim,
        surface_cmap=surface_cmap,
        residual_mode=residual_mode,
        residual_cmap=residual_cmap,
        show_residual_colorbar=show_residual_colorbar,
        extrapolation_factor=extrapolation_factor,
        extrapolation_absolute=extrapolation_absolute,
        **extra_kwargs,
    )

    return summary
