# -*- coding: utf-8 -*-

"""
Bootstrap tools for resampling indexed data.

This module contains the bootstrap function used throughout the
risk-calculation workflow. The function generates bootstrap index arrays
that can be applied to pandas objects or NumPy arrays without copying
the full datasets in advance.

The design keeps the resampling step separate from the fitting steps so
that bootstrap logic can be tested independently and reused across the
different parts of the workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap(
    data: int | pd.Series | pd.DataFrame | np.ndarray,
    n_boot: int,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """
    Generate bootstrap index arrays for an input dataset.

    Parameters
    ----------
    data : int or pandas.Series or pandas.DataFrame or numpy.ndarray
        Input data or number of observations. If an integer is supplied,
        it is interpreted directly as the number of observations. If a
        pandas or NumPy object is supplied, the number of observations
        is inferred using ``len(data)``.
    n_boot : int
        Number of bootstrap samples to generate.
    random_state : int or None, optional
        Seed for the random number generator. If provided, the bootstrap
        samples are reproducible. The default is ``None``.

    Returns
    -------
    list of numpy.ndarray
        List of bootstrap index arrays. Each array has length equal to
        the number of observations in the original dataset and contains
        indices sampled with replacement.

    Raises
    ------
    ValueError
        If the input dataset is empty.
    ValueError
        If ``n_boot`` is negative.

    Notes
    -----
    Bootstrap resampling is performed by drawing indices with
    replacement from the range ``0`` to ``n_obs - 1``. The returned
    indices can then be applied to the relevant pandas objects using
    ``.iloc`` or to NumPy arrays using standard indexing.

    Examples
    --------
    >>> import pandas as pd
    >>> from risk_tools.bootstrap import bootstrap
    >>> df = pd.DataFrame({"x": [1, 2, 3, 4]})
    >>> idx = bootstrap(df, n_boot=2, random_state=42)
    >>> len(idx)
    2
    >>> len(idx[0])
    4
    """
    if isinstance(data, int):
        n_obs = data
    else:
        n_obs = len(data)

    if n_obs <= 0:
        raise ValueError("Input data must contain at least one observation.")
    if n_boot < 0:
        raise ValueError("n_boot must be zero or a positive integer.")

    rng = np.random.default_rng(random_state)
    return [rng.choice(n_obs, size=n_obs, replace=True) for _ in range(n_boot)]
