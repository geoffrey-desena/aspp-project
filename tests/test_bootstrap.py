# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from risk_tools.bootstrap import bootstrap


def test_bootstrap_returns_correct_number_of_samples():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = bootstrap(df, n_boot=3, random_state=42)

    assert isinstance(out, list)
    assert len(out) == 3


def test_bootstrap_each_sample_has_input_length():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = bootstrap(df, n_boot=4, random_state=42)

    assert all(len(idx) == len(df) for idx in out)


def test_bootstrap_indices_are_in_valid_range():
    df = pd.DataFrame({"x": [10, 20, 30, 40]})
    out = bootstrap(df, n_boot=2, random_state=42)

    for idx in out:
        assert np.all(idx >= 0)
        assert np.all(idx < len(df))


def test_bootstrap_is_reproducible_with_seed():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    out1 = bootstrap(df, n_boot=3, random_state=123)
    out2 = bootstrap(df, n_boot=3, random_state=123)

    for a, b in zip(out1, out2):
        assert np.array_equal(a, b)


def test_bootstrap_accepts_integer_length():
    out = bootstrap(5, n_boot=2, random_state=1)

    assert len(out) == 2
    assert all(len(idx) == 5 for idx in out)
