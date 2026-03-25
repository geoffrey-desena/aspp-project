# -*- coding: utf-8 -*-

"""
risk_tools
==========

Package for bootstrapped wind/load risk calculations.

This package provides functions to:

- generate bootstrap resampling indices,
- fit Weibull distributions to wind-speed data,
- fit lognormal distributions to normalized load data conditioned on
  wind-speed bins,
- calculate exceedance risk for operating-envelope boundaries, and
- split timestamp-indexed data into seasonal subsets.

Modules
-------
bootstrap
    Bootstrap index generation.

fv
    Weibull fitting for wind-speed data.

fp
    Lognormal fitting for load conditioned on wind speed.

risk
    Risk calculation from fitted wind-speed and load distributions.

utils
    Utility functions for data preparation and formatting.
"""

from .bootstrap import bootstrap
from .fv import fv_params
from .fp import fp_params
from .risk import risk_calculation
from .utils import coerce_v_char_input, split_by_season, format_mean_pm_std
from .profiling import timed

__all__ = [
    "bootstrap",
    "fv_params",
    "fp_params",
    "risk_calculation",
    "coerce_v_char_input",
    "split_by_season",
    "format_mean_pm_std",
    "timed",
]