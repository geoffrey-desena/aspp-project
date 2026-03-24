# -*- coding: utf-8 -*-

"""
Conditional load-distribution fitting.

This module contains functions for fitting lognormal distributions to
normalized load data conditioned on wind-speed bins. The main public
function, ``fp_params``, bins the paired ``p_load`` and ``v_char`` data
into integer wind-speed bins and fits a lognormal distribution to the
load values in each bin.

The output includes the original dataset and a set of bootstrap
resamples. The fitted parameters are intended for later use in the risk
integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .bootstrap import bootstrap


def fp_params(
    df_fp: pd.DataFrame,
    n_boot: int,
    random_state: int | None = None,
    bin_width: float = 1.0,
    bin_min: float = 0.0,
    bin_max: float = 32.0,
) -> pd.DataFrame:
    """
    Fit lognormal distributions to load data within wind-speed bins.

    Parameters
    ----------
    df_fp : pandas.DataFrame
        DataFrame containing columns ``"p_load"`` and ``"v_char"``.
        ``p_load`` is the normalized load variable and ``v_char`` is the
        associated wind speed.
    n_boot : int
        Number of bootstrap samples to generate and fit.
    random_state : int or None, optional
        Seed for reproducible bootstrap resampling. The default is
        ``None``.
    bin_width : float, optional
        Width of the wind-speed bins. The default is ``1.0``.
    bin_min : float, optional
        Lower bound of the wind-speed binning range. The default is
        ``0.0``.
    bin_max : float, optional
        Upper bound of the wind-speed binning range. The default is
        ``32.0``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``["sample", "bin"]`` with columns:

        - ``shape`` : fitted lognormal shape parameter
        - ``scale`` : fitted lognormal scale parameter
        - ``n`` : number of valid observations used in the fit

        The ``sample`` level contains ``"original"`` plus the bootstrap
        samples ``"boot_0"``, ``"boot_1"``, and so on.

    Raises
    ------
    ValueError
        If the input DataFrame does not contain both ``"p_load"`` and
        ``"v_char"``.
    ValueError
        If ``bin_width`` is not positive.
    ValueError
        If ``bin_max`` is not greater than ``bin_min``.

    Notes
    -----
    The lognormal fit is performed with location fixed at zero by
    passing ``floc=0`` to ``scipy.stats.lognorm.fit``.

    Only finite positive ``p_load`` values are retained before fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from risk_tools.fp import fp_params
    >>> df = pd.DataFrame(
    ...     {"p_load": [0.4, 0.5, 0.6], "v_char": [8.2, 8.7, 9.1]}
    ... )
    >>> out = fp_params(df, n_boot=1, random_state=1)
    >>> "original" in out.index.get_level_values("sample")
    True
    """
    required_cols = {"p_load", "v_char"}
    if not required_cols.issubset(df_fp.columns):
        raise ValueError("Input DataFrame must contain columns 'p_load' and 'v_char'.")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    if bin_max <= bin_min:
        raise ValueError("bin_max must be greater than bin_min.")

    bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)
    bin_labels = np.arange(len(bin_edges) - 1)

    def _fit_one_sample(
        p_load: pd.Series,
        v_char: pd.Series,
        sample_name: str,
    ) -> list[dict[str, float | int | str]]:
        """
        Fit one original or bootstrapped sample.

        Parameters
        ----------
        p_load : pandas.Series
            Load values for one sample.
        v_char : pandas.Series
            Wind-speed values for one sample.
        sample_name : str
            Name used to label the sample in the output.

        Returns
        -------
        list of dict
            List of row dictionaries to be assembled into a DataFrame.
        """
        v_bins = pd.cut(
            v_char,
            bins=bin_edges,
            right=False,
            labels=bin_labels,
            include_lowest=True,
        )

        rows: list[dict[str, float | int | str]] = []

        for bin_id in bin_labels:
            mask = v_bins == bin_id
            p_bin = p_load.loc[mask].to_numpy()

            p_bin = p_bin[np.isfinite(p_bin)]
            p_bin = p_bin[p_bin > 0]

            n = len(p_bin)

            if n == 0:
                shape = np.nan
                scale = np.nan
            else:
                try:
                    shape, _, scale = lognorm.fit(p_bin, floc=0)
                except Exception:
                    shape = np.nan
                    scale = np.nan

            rows.append(
                {
                    "sample": sample_name,
                    "bin": int(bin_id),
                    "shape": shape,
                    "scale": scale,
                    "n": n,
                }
            )

        return rows

    results: list[dict[str, float | int | str]] = []

    results.extend(
        _fit_one_sample(
            p_load=df_fp["p_load"],
            v_char=df_fp["v_char"],
            sample_name="original",
        )
    )

    boot_idx = bootstrap(df_fp, n_boot=n_boot, random_state=random_state)

    for i, idx in enumerate(boot_idx):
        df_boot = df_fp.iloc[idx]
        results.extend(
            _fit_one_sample(
                p_load=df_boot["p_load"],
                v_char=df_boot["v_char"],
                sample_name=f"boot_{i}",
            )
        )

    return pd.DataFrame(results).set_index(["sample", "bin"]).sort_index()
