# -*- coding: utf-8 -*-

"""
Utility functions for the risk_tools package.

This module contains helper functions for:

- normalizing wind-speed input loaded from pickle files,
- splitting timestamp-indexed data into seasonal subsets, and
- formatting summary statistics for display.
"""

from __future__ import annotations

import pandas as pd


SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

SEASON_ORDER = ["spring", "summer", "fall", "winter"]


def coerce_v_char_input(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Normalize wind-speed input to a one-column DataFrame named ``"v_char"``.

    Parameters
    ----------
    obj : pandas.Series or pandas.DataFrame
        Wind-speed data loaded from file. It may already be a one-column
        DataFrame or it may be a Series without a column name.

    Returns
    -------
    pandas.DataFrame
        One-column DataFrame with the column name ``"v_char"``.

    Raises
    ------
    TypeError
        If the input is neither a pandas Series nor a one-column
        DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from risk_tools.utils import coerce_v_char_input
    >>> s = pd.Series([1.0, 2.0, 3.0])
    >>> df = coerce_v_char_input(s)
    >>> list(df.columns)
    ['v_char']
    """
    if isinstance(obj, pd.Series):
        return obj.to_frame(name="v_char")

    if isinstance(obj, pd.DataFrame):
        if "v_char" in obj.columns:
            return obj[["v_char"]].copy()
        if len(obj.columns) == 1:
            df_v = obj.copy()
            df_v.columns = ["v_char"]
            return df_v

    raise TypeError("Wind-speed input must be a pandas Series or a one-column DataFrame.")


def split_by_season(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split a timestamp-indexed DataFrame into seasonal subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a ``DatetimeIndex``.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary with keys ``"spring"``, ``"summer"``, ``"fall"``,
        and ``"winter"``. Each value is the subset of the input DataFrame
        corresponding to that season.

    Raises
    ------
    TypeError
        If the input DataFrame does not have a ``DatetimeIndex``.

    Notes
    -----
    Seasons are defined by calendar month as follows:

    - winter: December, January, February
    - spring: March, April, May
    - summer: June, July, August
    - fall: September, October, November
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")

    df_tmp = df.copy()
    df_tmp["season"] = df_tmp.index.month.map(SEASON_MAP)

    return {
        season: df_tmp.loc[df_tmp["season"] == season].drop(columns="season")
        for season in SEASON_ORDER
    }


def format_mean_pm_std(mean_val: float, std_val: float, decimals: int = 6) -> str:
    """
    Format a mean and standard deviation as ``mean ± std``.

    Parameters
    ----------
    mean_val : float
        Mean value.
    std_val : float
        Standard deviation.
    decimals : int, optional
        Number of decimal places to display. The default is ``6``.

    Returns
    -------
    str
        Formatted string of the form ``"0.123456 ± 0.012345"``. If
        either value is missing, returns ``"NaN"``.
    """
    if pd.isna(mean_val) or pd.isna(std_val):
        return "NaN"
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"