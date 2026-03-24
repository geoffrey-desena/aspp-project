# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from risk_tools.fp import fp_params


def make_fp_input(n=400, seed=42):
    rng = np.random.default_rng(seed)
    v_char = rng.uniform(0, 32, size=n)
    p_load = rng.lognormal(mean=-0.7, sigma=0.35, size=n)
    p_load = np.clip(p_load, 0, 1.2)

    return pd.DataFrame({"p_load": p_load, "v_char": v_char})


def test_fp_params_returns_original_plus_bootstraps():
    df_fp = make_fp_input()

    out = fp_params(df_fp, n_boot=4, random_state=42)

    samples = out.index.get_level_values("sample").unique()
    assert "original" in samples
    assert len(samples) == 5


def test_fp_params_has_expected_columns():
    df_fp = make_fp_input()

    out = fp_params(df_fp, n_boot=2, random_state=42)

    assert list(out.columns) == ["shape", "scale", "n"]


def test_fp_params_contains_expected_bins():
    df_fp = make_fp_input()

    out = fp_params(
        df_fp,
        n_boot=1,
        random_state=42,
        bin_width=1.0,
        bin_min=0.0,
        bin_max=32.0,
    )

    bins = out.index.get_level_values("bin").unique()
    assert bins.min() == 0
    assert bins.max() == 31
    assert len(bins) == 32


def test_fp_params_counts_are_nonnegative():
    df_fp = make_fp_input()

    out = fp_params(df_fp, n_boot=1, random_state=42)

    assert (out["n"] >= 0).all()
