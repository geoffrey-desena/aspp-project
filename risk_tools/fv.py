# -*- coding: utf-8 -*-

"""
Wind-speed distribution fitting.

This module contains functions for fitting Weibull distributions to
wind-speed time series. The main public function, ``fv_params``, fits a
two-parameter Weibull distribution to the original dataset and to a set
of bootstrap resamples.

The output is intended to be used later in the risk calculation, where
the fitted Weibull distribution describes the probability of wind speed
falling within each integer wind-speed bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import weibull_min

from .bootstrap import bootstrap


def fv_params(
    df_v: pd.DataFrame,
    n_boot: int,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Fit Weibull distributions to original and bootstrapped wind-speed data.

    Parameters
    ----------
    df_v : pandas.DataFrame
        DataFrame containing a column named ``"v_char"``. The values are
        assumed to represent wind speed. The index is typically a
        timestamp, but only the data column is used in the fit.
    n_boot : int
        Number of bootstrap samples to generate and fit.
    random_state : int or None, optional
        Seed for reproducible bootstrap resampling. The default is
        ``None``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by sample name. The first row corresponds to
        the original dataset and is labeled ``"original"``. Subsequent
        rows correspond to bootstrap samples labeled ``"boot_0"``,
        ``"boot_1"``, and so on. Columns are:

        - ``shape`` : fitted Weibull shape parameter
        - ``scale`` : fitted Weibull scale parameter

    Raises
    ------
    ValueError
        If the input DataFrame does not contain a column named
        ``"v_char"``.
    ValueError
        If the wind-speed data contain no finite positive values.

    Notes
    -----
    The fit is a two-parameter Weibull fit with location fixed at zero
    by passing ``floc=0`` to ``scipy.stats.weibull_min.fit``.

    Only finite positive values are retained before fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from risk_tools.fv import fv_params
    >>> df = pd.DataFrame({"v_char": [5.0, 6.0, 7.0, 8.0]})
    >>> out = fv_params(df, n_boot=2, random_state=1)
    >>> "original" in out.index
    True
    """
    if "v_char" not in df_v.columns:
        raise ValueError("Input DataFrame must contain a column named 'v_char'.")

    v = df_v["v_char"].to_numpy()
    v = v[np.isfinite(v)]
    v = v[v > 0]

    if len(v) == 0:
        raise ValueError("Input wind-speed data contains no finite positive values.")

    results: list[dict[str, float | str]] = []

    shape, _, scale = weibull_min.fit(v, floc=0)
    results.append({"sample": "original", "shape": shape, "scale": scale})

    boot_idx = bootstrap(df_v, n_boot=n_boot, random_state=random_state)

    for i, idx in enumerate(boot_idx):
        v_boot = df_v["v_char"].iloc[idx].to_numpy()
        v_boot = v_boot[np.isfinite(v_boot)]
        v_boot = v_boot[v_boot > 0]

        if len(v_boot) == 0:
            shape = np.nan
            scale = np.nan
        else:
            try:
                shape, _, scale = weibull_min.fit(v_boot, floc=0)
            except Exception:
                shape = np.nan
                scale = np.nan

        results.append({"sample": f"boot_{i}", "shape": shape, "scale": scale})

    return pd.DataFrame(results).set_index("sample")
