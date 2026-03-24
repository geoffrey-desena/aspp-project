# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from risk_tools.fv import fv_params


def test_fv_params_returns_original_plus_bootstraps():
    rng = np.random.default_rng(42)
    df_v = pd.DataFrame({"v_char": rng.weibull(a=2.0, size=200) * 10})

    out = fv_params(df_v, n_boot=5, random_state=42)

    assert "original" in out.index
    assert len(out) == 6


def test_fv_params_has_expected_columns():
    rng = np.random.default_rng(42)
    df_v = pd.DataFrame({"v_char": rng.weibull(a=2.0, size=200) * 10})

    out = fv_params(df_v, n_boot=3, random_state=42)

    assert list(out.columns) == ["shape", "scale"]


def test_fv_params_returns_finite_positive_parameters_for_reasonable_input():
    rng = np.random.default_rng(42)
    df_v = pd.DataFrame({"v_char": rng.weibull(a=2.5, size=300) * 8})

    out = fv_params(df_v, n_boot=3, random_state=42)

    assert np.isfinite(out["shape"]).all()
    assert np.isfinite(out["scale"]).all()
    assert (out["shape"] > 0).all()
    assert (out["scale"] > 0).all()
