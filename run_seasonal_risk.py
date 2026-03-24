#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run seasonal wind/load risk calculations.

This script serves as the driver for the ``risk_tools`` package. It loads
the required input datasets, prepares them for analysis, splits them into
seasonal subsets, runs the bootstrap-based risk calculation for each
season, and assembles the final summary table.

Inputs
------
The script expects the following pickle files to be located in the same
directory as the script:

- ``df_normalized_p_v_char.pkl``
    DataFrame containing paired time series of normalized load and wind
    speed with columns ``"p_load"`` and ``"v_char"``.
- ``v_char_series.pkl``
    Series or one-column DataFrame containing a longer wind-speed time
    series indexed by timestamp.
- ``df_plimits.pkl``
    DataFrame indexed by wind-speed bin with columns ``"danger"`` and
    ``"limit"``.

Outputs
-------
The script prints a seasonal summary table whose rows correspond to
seasons and whose columns are a two-level column index:

- top level: ``"Low wind"``, ``"High wind"``
- lower level: ``"Danger"``, ``"Limit"``

Each cell contains the bootstrap mean and standard deviation formatted as
``mean ± std``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_tools import (
    coerce_v_char_input,
    format_mean_pm_std,
    risk_calculation,
    split_by_season,
)
from risk_tools.utils import SEASON_ORDER


def main() -> None:
    """
    Load data, run seasonal risk calculations, and print a summary table.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function prints the seasonal summary table to standard output.

    Notes
    -----
    The seasonal analysis uses the same risk-calculation settings for
    each season:

    - overall wind-speed range: 0 to 32 m/s
    - low-wind range: 0 to 15 m/s
    - high-wind range: 15 to 32 m/s
    - upper normalized load integration limit: 1.0
    """
    base_path = Path(__file__).resolve().parent

    fp_path = base_path / "df_normalized_p_v_char.pkl"
    v_path = base_path / "v_char_series.pkl"
    plimits_path = base_path / "df_plimits.pkl"

    if not fp_path.exists():
        raise FileNotFoundError(f"Could not find file: {fp_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"Could not find file: {v_path}")
    if not plimits_path.exists():
        raise FileNotFoundError(f"Could not find file: {plimits_path}")

    df_fp = pd.read_pickle(fp_path)
    v_obj = pd.read_pickle(v_path)
    df_v = coerce_v_char_input(v_obj)
    df_plimits = pd.read_pickle(plimits_path)

    seasonal_fp = split_by_season(df_fp)
    seasonal_fv = split_by_season(df_v)

    n_boot = 10
    random_state = 42

    cols = pd.MultiIndex.from_product(
        [["Low wind", "High wind"], ["Danger", "Limit"]],
        names=["Wind regime", "Threshold"],
    )
    df_summary = pd.DataFrame(index=SEASON_ORDER, columns=cols, dtype=object)

    for season in SEASON_ORDER:
        out = risk_calculation(
            df_fp=seasonal_fp[season],
            df_v=seasonal_fv[season],
            df_plimits=df_plimits,
            n_boot=n_boot,
            random_state=random_state,
            overall_v_lower=0.0,
            overall_v_upper=32.0,
            low_v_lower=0.0,
            low_v_upper=15.0,
            high_v_lower=15.0,
            high_v_upper=32.0,
            p_upper=1.0,
            bin_width=1.0,
            fp_bin_min=0.0,
            fp_bin_max=32.0,
        )

        stats = out["stats_by_regime"]

        df_summary.loc[season, ("Low wind", "Danger")] = format_mean_pm_std(
            stats.loc["Low wind / Danger", "mean"],
            stats.loc["Low wind / Danger", "std"],
        )
        df_summary.loc[season, ("Low wind", "Limit")] = format_mean_pm_std(
            stats.loc["Low wind / Limit", "mean"],
            stats.loc["Low wind / Limit", "std"],
        )
        df_summary.loc[season, ("High wind", "Danger")] = format_mean_pm_std(
            stats.loc["High wind / Danger", "mean"],
            stats.loc["High wind / Danger", "std"],
        )
        df_summary.loc[season, ("High wind", "Limit")] = format_mean_pm_std(
            stats.loc["High wind / Limit", "mean"],
            stats.loc["High wind / Limit", "std"],
        )

    print("\nSeasonal risk summary (mean ± std):")
    print(df_summary)


if __name__ == "__main__":
    main()
