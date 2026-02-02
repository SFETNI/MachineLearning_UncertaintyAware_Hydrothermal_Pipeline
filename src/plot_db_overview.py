import numpy as np
import matplotlib.pyplot as plt

def _clean_series(s, clip_quantiles=None):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if clip_quantiles is not None and not s.empty:
        q_lo, q_hi = clip_quantiles
        lo, hi = s.quantile(q_lo), s.quantile(q_hi)
        s = s[(s >= lo) & (s <= hi)]
    return s


def _group_stats(df, cols, clip_quantiles=None):
    stats = {}
    for c in cols:
        s = _clean_series(df[c], clip_quantiles=clip_quantiles)
        if s.empty:
            stats[c] = dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
        else:
            stats[c] = dict(
                n=int(s.size),
                mean=s.mean(),
                std=s.std(),
                min=s.min(),
                max=s.max(),
            )
    return stats


def _add_box_group(
    ax,
    df,
    cols,
    labels,
    ylabel,
    clip_quantiles=None,
    log_y=False,
    showfliers=False,
    ylim=None,
    annotate_means=True,
    annotate_minmax=True,
    whis=(0, 100),
    scale_factors=None,   # dict: {column_name: factor}
):
    """
    Draw a group of boxplots with:
      - white square = mean
      - horizontal bar = median
      - vertical bar = mean ± 1.5 SD
      - optional mean, min, max labels
      - optional per-column scaling factor (for plotting only)
    """
    if scale_factors is None:
        scale_factors = {}

    stats = _group_stats(df, cols, clip_quantiles=clip_quantiles)

    data = []
    for c in cols:
        f = float(scale_factors.get(c, 1.0))
        s = _clean_series(df[c], clip_quantiles=clip_quantiles)
        data.append(s.values * f)

    bp = ax.boxplot(
        data,
        positions=np.arange(1, len(cols) + 1),
        widths=0.6,
        patch_artist=True,
        showfliers=showfliers,
        whis=whis,
    )

    colors = ["#fdbf6f", "#a6cee3", "#b2df8a",
              "#cab2d6", "#fb9a99", "#ffff99", "#e31a1c"]
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors * 3), start=1):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        
        if i <= len(cols):
            c = cols[i - 1]
            f = float(scale_factors.get(c, 1.0)) if scale_factors else 1.0
            if f != 1.0:
                box_width = 0.6
                box_pad = 0.15  # padding around the box
                y_data = data[i - 1]
                if len(y_data) > 0:
                    from matplotlib.patches import Rectangle
                    q1, q3 = np.percentile(y_data, [25, 75])
                    y_min_box, y_max_box = y_data.min(), y_data.max()
                    
                    rect = Rectangle(
                        (i - box_width/2 - box_pad, y_min_box - 0.02 * (y_max_box - y_min_box)),
                        box_width + 2*box_pad,
                        (y_max_box - y_min_box) * 1.04,
                        linewidth=1.2,
                        linestyle='--',
                        edgecolor='gray',
                        facecolor='none',
                        alpha=0.6,
                        zorder=1
                    )
                    ax.add_patch(rect)
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    # For positioning min/max labels
    if ylim is not None:
        y_min, y_max = ylim
    else:
        all_vals = np.concatenate([d for d in data if len(d) > 0])
        if all_vals.size > 0:
            y_min, y_max = float(all_vals.min()), float(all_vals.max())
        else:
            y_min, y_max = 0.0, 1.0
    dy = 0.015 * (y_max - y_min)

    # Means and mean ±1.5 SD
    xs = np.arange(1, len(cols) + 1)
    for i, c in enumerate(cols, start=1):
        s = stats[c]
        if s["n"] == 0 or np.isnan(s["std"]):
            continue

        f = float(scale_factors.get(c, 1.0))
        m = s["mean"]
        sd = 1.5 * s["std"]

        # mean square - positioned more to the right
        ax.scatter(
            i + 0.15,
            m * f,
            marker="s",
            s=30,
            facecolor="white",
            edgecolor="black",
            zorder=3,
        )

        # mean ± 1.5 SD bar
        ax.errorbar(
            i,
            m * f,
            yerr=sd * f,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
        )

        # text for mean – slightly to the right of the square
        if annotate_means:
            ax.text(
                i + 0.33,
                m * f,
                f"{m:.1f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        # min / max values at whisker ends
        if annotate_minmax:
            vmin = s["min"]
            vmax = s["max"]
            ax.text(
                i,
                vmin * f - dy,
                f"{vmin:.2f}",
                ha="center",
                va="top",
                fontsize=7,
            )
            ax.text(
                i,
                vmax * f + dy,
                f"{vmax:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")

    if log_y:
        ax.set_yscale("log")

    if ylim is not None:
        ax.set_ylim(*ylim)

    return stats


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_db_overview(htt_data):
    # ===================================================================
    # CONFIGURABLE SCALE FACTORS (modify these to adjust visual scaling)
    # ===================================================================
    scale=5
    SCALE_FEEDSTOCK = {"N": scale, "S": scale, "H": scale}  # Panel A
    SCALE_CONDITIONS = {"IC": 3.0, "pressure_effective_mpa": 3.0}  # Panel B - IC scaled by 2
    SCALE_BIOOIL = {"N_biooil": scale, "S_biooil": scale, "H_biooil": scale}  # Panel D
    SCALE_BIOCHAR = {"H_biochar": scale, "N_biochar": scale, "S_biochar": scale}  # Panel E
    
    # 3 rows × 4 columns; panels A-D span 2 cols each, E spans 2 cols centered
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.9])

    axA = fig.add_subplot(gs[0, 0:2])  # left half
    axB = fig.add_subplot(gs[0, 2:4])  # right half
    axC = fig.add_subplot(gs[1, 0:2])  # left half
    axD = fig.add_subplot(gs[1, 2:4])  # right half
    
    # Panel E: centered in bottom row, spanning middle 2 columns
    axE = fig.add_subplot(gs[2, 1:3])  # centered, same width as others

    # ---- helper to build labels with "×factor" when scaled ----
    def _labels_with_scale(cols, base_labels, scale_factors):
        labels = []
        for c in cols:
            base = base_labels.get(c, c)
            f = float(scale_factors.get(c, 1.0)) if scale_factors else 1.0
            if f != 1.0:
                labels.append(f"{base} ×{f:g}")
            else:
                labels.append(base)
        return labels

    # ------------------------------------------------------------------
    # A) Feedstock composition – add cellulose & hemicellulose
    # ------------------------------------------------------------------
    cols_A = ["C", "H", "N", "O", "S", "Ash", "Lignin", "cellulose", "hemicellulose"]
    base_labels_A = {
        "C": "C (wt%)",
        "H": "H (wt%)",
        "N": "N (wt%)",
        "O": "O (wt%)",
        "S": "S (wt%)",
        "Ash": "Ash (wt%)",
        "Lignin": "Lignin (wt%)",
        "cellulose": "Cellulose (wt%)",
        "hemicellulose": "Hemicellulose (wt%)",
    }
    # Use configurable scale factors
    labels_A = _labels_with_scale(cols_A, base_labels_A, SCALE_FEEDSTOCK)

    stats_A = _add_box_group(
        axA, htt_data, cols_A, labels_A,
        ylabel="Range",
        clip_quantiles=(0.01, 0.99),
        ylim=(0, 100),
        whis=(0, 100),
        scale_factors=SCALE_FEEDSTOCK,
    )
    axA.set_title("A  Feedstock composition", pad=12)

    # ------------------------------------------------------------------
    # B) Operating conditions – moderate clipping, no scaling
    # ------------------------------------------------------------------
    cols_B = ["IC", "T", "t", "pressure_effective_mpa"]
    base_labels_B = {
        "IC": "IC (wt% slurry)",
        "T": "Temperature (°C)",
        "t": "Residence time (min)",
        "pressure_effective_mpa": "Pressure (MPa)",
    }
    labels_B = _labels_with_scale(cols_B, base_labels_B, SCALE_CONDITIONS)
    stats_B = _add_box_group(
        axB, htt_data, cols_B, labels_B,
        ylabel="Range",
        clip_quantiles=(0.05, 0.95),
        whis=(0, 100),
        scale_factors=SCALE_CONDITIONS,
    )
    axB.set_title("B  Operating conditions")

    # ------------------------------------------------------------------
    # C) Product yields – full 0–100 %, no clipping
    # ------------------------------------------------------------------
    cols_C = ["B_Y", "A_Y", "G_Y", "C_Y"]
    base_labels_C = {
        "B_Y": "Bio-oil yield (%)",
        "A_Y": "Aqueous yield (%)",
        "G_Y": "Gas yield (%)",
        "C_Y": "Char yield (%)",
    }
    labels_C = _labels_with_scale(cols_C, base_labels_C, {})
    stats_C = _add_box_group(
        axC, htt_data, cols_C, labels_C,
        ylabel="Range",
        clip_quantiles=None,
        ylim=(0, 100),
        whis=(0, 100),
        scale_factors={},
    )
    axC.set_title("C  Product yields")

    # ------------------------------------------------------------------
    # D) Bio-oil properties – light clipping, scale N,S by 2
    # ------------------------------------------------------------------
    cols_D = ["C_biooil", "H_biooil", "N_biooil",
              "O_biooil", "S_biooil", "HHV_biooil"]
    base_labels_D = {
        "C_biooil": "C (wt%)",
        "H_biooil": "H (wt%)",
        "N_biooil": "N (wt%)",
        "O_biooil": "O (wt%)",
        "S_biooil": "S (wt%)",
        "HHV_biooil": "HHV (MJ/kg)",
    }
    labels_D = _labels_with_scale(cols_D, base_labels_D, SCALE_BIOOIL)

    stats_D = _add_box_group(
        axD, htt_data, cols_D, labels_D,
        ylabel="Range",
        clip_quantiles=(0.01, 0.99),
        whis=(0, 100),
        scale_factors=SCALE_BIOOIL,
    )
    axD.set_title("D  Bio-oil properties")

    # ------------------------------------------------------------------
    # E) Bio-char properties – new panel, bottom row spanning both columns
    # ------------------------------------------------------------------
    cols_E = ["C_biochar", "H_biochar", "N_biochar",
              "O_biochar", "S_biochar", "HHV_biochar"]
    base_labels_E = {
        "C_biochar": "C (wt%)",
        "H_biochar": "H (wt%)",
        "N_biochar": "N (wt%)",
        "O_biochar": "O (wt%)",
        "S_biochar": "S (wt%)",
        "HHV_biochar": "HHV (MJ/kg)",
    }
    labels_E = _labels_with_scale(cols_E, base_labels_E, SCALE_BIOCHAR)

    stats_E = _add_box_group(
        axE, htt_data, cols_E, labels_E,
        ylabel="Range",
        clip_quantiles=(0.01, 0.99),
        whis=(0, 100),
        scale_factors=SCALE_BIOCHAR,
    )
    axE.set_title("E  Bio-char properties")

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    handles = [
        plt.Line2D([], [], color="black", marker="s", linestyle="none",
                   markersize=6, markerfacecolor="white", label="Mean"),
        plt.Line2D([], [], color="black", linewidth=1.4, label="Median line"),
        plt.Line2D([], [], color="black", linewidth=1.0, marker="_",
                   markersize=10, label="Mean ± 1.5 SD"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    plt.show()


import pandas as pd

import pandas as pd

def db_overview_stats(htt_data, return_dfs=False):
    """
    Compute key stats (n, mean, std, min, max) for the overview panels,
    using the SAME columns and quantile clipping as in plot_db_overview.
    """

    panels = {
        "A_feedstock": {
            "title": "A  Feedstock composition",
            # now includes cellulose and hemicellulose
            "cols": ["C", "H", "N", "O", "S", "Ash", "Lignin",
                     "cellulose", "hemicellulose"],
            "clip": (0.01, 0.99),
        },
        "B_conditions": {
            "title": "B  Operating conditions",
            "cols": ["IC", "T", "t", "pressure_effective_mpa"],
            "clip": (0.05, 0.95),
        },
        "C_yields": {
            "title": "C  Product yields",
            "cols": ["B_Y", "A_Y", "G_Y", "C_Y"],
            "clip": None,   # full 0–100 range
        },
        "D_biooil": {
            "title": "D  Bio-oil properties",
            "cols": ["C_biooil", "H_biooil", "N_biooil",
                     "O_biooil", "S_biooil", "HHV_biooil"],
            "clip": (0.01, 0.99),
        },
        "E_biochar": {
            "title": "E  Bio-char properties",
            "cols": ["C_biochar", "H_biochar", "N_biochar",
                     "O_biochar", "S_biochar", "HHV_biochar"],
            "clip": (0.01, 0.99),
        },
    }

    stats_all = {}
    dfs = {}

    for key, cfg in panels.items():
        title = cfg["title"]
        cols = cfg["cols"]
        clip = cfg["clip"]

        stats = _group_stats(htt_data, cols, clip_quantiles=clip)
        stats_all[key] = stats

        print("\n" + "=" * 70)
        print(title)
        if clip is not None:
            print(f"(stats after clipping to {clip[0]*100:.1f}–{clip[1]*100:.1f} percentiles)")
        else:
            print("(no quantile clipping)")

        df_panel = (
            pd.DataFrame(stats)
            .T[["n", "mean", "std", "min", "max"]]
            .rename_axis("variable")
        )

        with pd.option_context(
            "display.float_format", lambda x: f"{x:7.3f}",
            "display.max_rows", None,
            "display.max_columns", None,
        ):
            print(df_panel)

        dfs[key] = df_panel

    if return_dfs:
        return stats_all, dfs
    return stats_all
