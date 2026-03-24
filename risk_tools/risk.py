# -*- coding: utf-8 -*-

"""
Risk integration functions.

This module contains the main risk-calculation logic. It combines
Weibull fits for wind speed and lognormal fits for load conditioned on
wind speed to approximate exceedance risk above externally defined
operating-envelope boundaries.

The module calculates:

- overall risk,
- low-wind risk,
- high-wind risk,
- bootstrap realizations of each, and
- summary statistics of the bootstrap results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min

from .fp import fp_params
from .fv import fv_params


def risk_calculation(
    df_fp: pd.DataFrame,
    df_v: pd.DataFrame,
    df_plimits: pd.DataFrame,
    n_boot: int,
    random_state: int | None = None,
    overall_v_lower: float = 0.0,
    overall_v_upper: float = 32.0,
    low_v_lower: float = 0.0,
    low_v_upper: float = 15.0,
    high_v_lower: float = 15.0,
    high_v_upper: float = 32.0,
    p_upper: float = 1.0,
    bin_width: float = 1.0,
    fp_bin_min: float = 0.0,
    fp_bin_max: float = 32.0,
) -> dict[str, pd.DataFrame]:
    """
    Calculate original and bootstrap risk metrics.

    Parameters
    ----------
    df_fp : pandas.DataFrame
        DataFrame containing columns ``"p_load"`` and ``"v_char"`` for
        the paired load-wind dataset.
    df_v : pandas.DataFrame
        DataFrame containing a column named ``"v_char"`` for the wind
        time series used to fit the wind-speed distribution.
    df_plimits : pandas.DataFrame
        DataFrame indexed by wind-speed bin with columns ``"danger"``
        and ``"limit"``. These columns define the lower integration
        limits of the risk calculation.
    n_boot : int
        Number of bootstrap samples to generate.
    random_state : int or None, optional
        Seed for reproducible bootstrap resampling. The default is
        ``None``.
    overall_v_lower, overall_v_upper : float, optional
        Wind-speed bounds for the overall risk calculation.
    low_v_lower, low_v_upper : float, optional
        Wind-speed bounds for the low-wind regime.
    high_v_lower, high_v_upper : float, optional
        Wind-speed bounds for the high-wind regime.
    p_upper : float, optional
        Upper integration limit in normalized load space. The default is
        ``1.0``.
    bin_width : float, optional
        Width of the wind-speed bins. The default is ``1.0``.
    fp_bin_min, fp_bin_max : float, optional
        Wind-speed binning range used when fitting the lognormal
        conditional load distributions.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary containing the following DataFrames:

        - ``original_overall`` : overall risk for original data
        - ``original_by_regime`` : low- and high-wind risks for original data
        - ``bootstrap_overall`` : overall bootstrap risks
        - ``bootstrap_by_regime`` : low- and high-wind bootstrap risks
        - ``stats_overall`` : mean and standard deviation of overall bootstrap risks
        - ``stats_by_regime`` : mean and standard deviation of regime bootstrap risks

    Raises
    ------
    ValueError
        If required input columns are missing.

    Notes
    -----
    The risk calculation approximates

    .. math::

        \\int_{v_{lower}}^{v_{upper}} f_v(v)
        \\int_{p_{lower}(v)}^{p_{upper}} f_p(p \\mid v) \\, dp \\, dv

    using integer wind-speed bins and fitted distribution functions.

    For each wind-speed bin, the contribution is the product of:

    - the Weibull probability mass of the wind-speed bin, and
    - the lognormal probability mass between the boundary and
      ``p_upper``.

    Bins whose lower integration limit is greater than or equal to
    ``p_upper`` contribute zero risk.

    Examples
    --------
    >>> # Example omitted because realistic use requires prepared DataFrames.
    """
    if "v_char" not in df_v.columns:
        raise ValueError("df_v must contain a column named 'v_char'.")
    if not {"p_load", "v_char"}.issubset(df_fp.columns):
        raise ValueError("df_fp must contain columns 'p_load' and 'v_char'.")
    if not {"danger", "limit"}.issubset(df_plimits.columns):
        raise ValueError("df_plimits must contain columns 'danger' and 'limit'.")

    fv = fv_params(df_v=df_v, n_boot=n_boot, random_state=random_state)
    fp = fp_params(
        df_fp=df_fp,
        n_boot=n_boot,
        random_state=random_state,
        bin_width=bin_width,
        bin_min=fp_bin_min,
        bin_max=fp_bin_max,
    )

    def _risk_for_sample(
        sample_name: str,
        threshold_col: str,
        v_lower: float,
        v_upper: float,
    ) -> float:
        """
        Calculate risk for a single sample and one threshold curve.

        Parameters
        ----------
        sample_name : str
            Sample label, such as ``"original"`` or ``"boot_0"``.
        threshold_col : {"danger", "limit"}
            Column in ``df_plimits`` defining the lower integration
            limit.
        v_lower, v_upper : float
            Wind-speed bounds for the calculation.

        Returns
        -------
        float
            Calculated risk value for the specified sample and wind-speed
            range.
        """
        fv_shape = float(fv.loc[sample_name, "shape"])
        fv_scale = float(fv.loc[sample_name, "scale"])

        if not np.isfinite(fv_shape) or not np.isfinite(fv_scale):
            return np.nan

        try:
            fp_sample = fp.xs(sample_name, level="sample")
        except KeyError:
            return np.nan

        total = 0.0

        for k in range(int(v_lower), int(v_upper)):
            if k not in df_plimits.index or k not in fp_sample.index:
                continue

            p_lower = float(df_plimits.loc[k, threshold_col])
            if p_lower >= p_upper:
                continue

            fp_shape = float(fp_sample.loc[k, "shape"])
            fp_scale = float(fp_sample.loc[k, "scale"])

            if not np.isfinite(fp_shape) or not np.isfinite(fp_scale):
                continue

            w = weibull_min.cdf(
                k + bin_width, c=fv_shape, scale=fv_scale, loc=0
            ) - weibull_min.cdf(k, c=fv_shape, scale=fv_scale, loc=0)

            if w <= 0:
                continue

            tail = lognorm.cdf(
                p_upper, s=fp_shape, scale=fp_scale, loc=0
            ) - lognorm.cdf(p_lower, s=fp_shape, scale=fp_scale, loc=0)

            total += w * max(tail, 0.0)

        return total

    original_overall = pd.DataFrame(
        {
            "Risk": [
                _risk_for_sample(
                    "original", "danger", overall_v_lower, overall_v_upper
                ),
                _risk_for_sample("original", "limit", overall_v_lower, overall_v_upper),
            ]
        },
        index=["Danger", "Limit"],
    )

    original_by_regime = pd.DataFrame(
        {
            "Low wind": [
                _risk_for_sample("original", "danger", low_v_lower, low_v_upper),
                _risk_for_sample("original", "limit", low_v_lower, low_v_upper),
            ],
            "High wind": [
                _risk_for_sample("original", "danger", high_v_lower, high_v_upper),
                _risk_for_sample("original", "limit", high_v_lower, high_v_upper),
            ],
        },
        index=["Danger", "Limit"],
    )

    boot_overall_rows = []
    boot_regime_rows = []

    for i in range(n_boot):
        sample_name = f"boot_{i}"

        boot_overall_rows.append(
            {
                "sample": sample_name,
                "Danger": _risk_for_sample(
                    sample_name, "danger", overall_v_lower, overall_v_upper
                ),
                "Limit": _risk_for_sample(
                    sample_name, "limit", overall_v_lower, overall_v_upper
                ),
            }
        )

        boot_regime_rows.append(
            {
                "sample": sample_name,
                "Low wind / Danger": _risk_for_sample(
                    sample_name, "danger", low_v_lower, low_v_upper
                ),
                "Low wind / Limit": _risk_for_sample(
                    sample_name, "limit", low_v_lower, low_v_upper
                ),
                "High wind / Danger": _risk_for_sample(
                    sample_name, "danger", high_v_lower, high_v_upper
                ),
                "High wind / Limit": _risk_for_sample(
                    sample_name, "limit", high_v_lower, high_v_upper
                ),
            }
        )

    bootstrap_overall = pd.DataFrame(boot_overall_rows).set_index("sample")
    bootstrap_by_regime = pd.DataFrame(boot_regime_rows).set_index("sample")

    stats_overall = pd.DataFrame(
        {
            "mean": bootstrap_overall.mean(axis=0),
            "std": bootstrap_overall.std(axis=0),
        }
    )

    stats_by_regime = pd.DataFrame(
        {
            "mean": bootstrap_by_regime.mean(axis=0),
            "std": bootstrap_by_regime.std(axis=0),
        }
    )

    return {
        "original_overall": original_overall,
        "original_by_regime": original_by_regime,
        "bootstrap_overall": bootstrap_overall,
        "bootstrap_by_regime": bootstrap_by_regime,
        "stats_overall": stats_overall,
        "stats_by_regime": stats_by_regime,
    }
