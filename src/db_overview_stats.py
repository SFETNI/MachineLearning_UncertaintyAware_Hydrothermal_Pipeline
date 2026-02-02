import pandas as pd
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


def db_overview_stats(htt_data, return_dfs=False):
    """
    Compute key stats (n, mean, std, min, max) for the four overview panels,
    using the SAME columns and quantile clipping as in plot_db_overview.

    Parameters
    ----------
    htt_data : DataFrame
        Master dataset.
    return_dfs : bool
        If True, also return dict of DataFrames (one per panel).

    Returns
    -------
    stats_all : dict
        Nested dict: {panel_key -> {column -> stats_dict}}.
    dfs : dict (optional)
        {panel_key -> DataFrame of stats} if return_dfs=True.
    """

    # Panel definitions (mirrors plot_db_overview)
    panels = {
        "A_feedstock": {
            "title": "A  Feedstock composition",
            "cols": ["C", "H", "N", "O", "S", "Ash", "Lignin"],
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
