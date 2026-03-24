# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from risk_tools.risk import risk_calculation


def make_inputs(seed=42):
    rng = np.random.default_rng(seed)

    n_fp = 500
    n_v = 1000

    idx_fp = pd.date_range("2020-01-01", periods=n_fp, freq="H")
    idx_v = pd.date_range("2019-01-01", periods=n_v, freq="H")

    df_fp = pd.DataFrame(
        {
            "p_load": np.clip(rng.lognormal(mean=-0.8, sigma=0.3, size=n_fp), 0, 1.1),
            "v_char": rng.uniform(0, 32, size=n_fp),
        },
        index=idx_fp,
    )

    df_v = pd.DataFrame(
        {
            "v_char": rng.weibull(a=2.2, size=n_v) * 10,
        },
        index=idx_v,
    )

    df_plimits = pd.DataFrame(
        {
            "danger": np.full(32, 0.55),
            "limit": np.full(32, 0.70),
        },
        index=range(32),
    )

    return df_fp, df_v, df_plimits


def test_risk_calculation_returns_expected_keys():
    df_fp, df_v, df_plimits = make_inputs()

    out = risk_calculation(
        df_fp=df_fp,
        df_v=df_v,
        df_plimits=df_plimits,
        n_boot=3,
        random_state=42,
    )

    expected_keys = {
        "original_overall",
        "original_by_regime",
        "bootstrap_overall",
        "bootstrap_by_regime",
        "stats_overall",
        "stats_by_regime",
    }

    assert expected_keys.issubset(out.keys())


def test_risk_calculation_overall_tables_have_expected_shape():
    df_fp, df_v, df_plimits = make_inputs()

    out = risk_calculation(
        df_fp=df_fp,
        df_v=df_v,
        df_plimits=df_plimits,
        n_boot=4,
        random_state=42,
    )

    assert out["original_overall"].shape == (2, 1)
    assert out["original_by_regime"].shape == (2, 2)
    assert out["bootstrap_overall"].shape == (4, 2)


def test_risk_calculation_stats_are_finite_for_reasonable_input():
    df_fp, df_v, df_plimits = make_inputs()

    out = risk_calculation(
        df_fp=df_fp,
        df_v=df_v,
        df_plimits=df_plimits,
        n_boot=5,
        random_state=42,
    )

    stats = out["stats_overall"]

    assert np.isfinite(stats["mean"]).all()
    assert np.isfinite(stats["std"]).all()


def test_risk_danger_exceeds_limit_when_limit_is_stricter():
    df_fp, df_v, df_plimits = make_inputs()

    out = risk_calculation(
        df_fp=df_fp,
        df_v=df_v,
        df_plimits=df_plimits,
        n_boot=5,
        random_state=42,
    )

    stats = out["stats_overall"]

    assert stats.loc["Danger", "mean"] >= stats.loc["Limit", "mean"]
