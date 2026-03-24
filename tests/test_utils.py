# -*- coding: utf-8 -*-

import pandas as pd

from risk_tools.utils import coerce_v_char_input, format_mean_pm_std, split_by_season


def test_coerce_v_char_input_from_series():
    s = pd.Series([1.0, 2.0, 3.0])
    df = coerce_v_char_input(s)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["v_char"]
    assert len(df) == 3


def test_coerce_v_char_input_from_one_column_dataframe():
    df_in = pd.DataFrame({"wind": [1.0, 2.0, 3.0]})
    df_out = coerce_v_char_input(df_in)

    assert list(df_out.columns) == ["v_char"]
    assert len(df_out) == 3


def test_format_mean_pm_std():
    out = format_mean_pm_std(0.1234567, 0.0009876, decimals=4)
    assert out == "0.1235 ± 0.0010"


def test_split_by_season_assigns_rows_correctly():
    idx = pd.to_datetime(
        [
            "2020-01-15",  # winter
            "2020-04-15",  # spring
            "2020-07-15",  # summer
            "2020-10-15",  # fall
        ]
    )
    df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=idx)

    out = split_by_season(df)

    assert len(out["winter"]) == 1
    assert len(out["spring"]) == 1
    assert len(out["summer"]) == 1
    assert len(out["fall"]) == 1

    assert out["winter"]["value"].iloc[0] == 1
    assert out["spring"]["value"].iloc[0] == 2
    assert out["summer"]["value"].iloc[0] == 3
    assert out["fall"]["value"].iloc[0] == 4
