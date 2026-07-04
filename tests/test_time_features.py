from __future__ import annotations

import pandas as pd

from src.time_features import add_time_features


def test_adds_expected_columns():
    df = pd.DataFrame({"timestamp": ["2025-01-04 15:30:00", "2025-01-06 09:00:00"]})
    out = add_time_features(df)
    assert {"hour_of_day", "day_of_week", "is_weekend"} <= set(out.columns)
    assert out.loc[0, "hour_of_day"] == 15
    # 2025-01-04 is a Saturday -> weekend; 2025-01-06 is Monday -> not weekend
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[1, "is_weekend"] == 0
    assert out.loc[1, "day_of_week"] == 0


def test_missing_timestamp_returns_copy_unchanged():
    df = pd.DataFrame({"a": [1, 2]})
    out = add_time_features(df)
    assert list(out.columns) == ["a"]
    assert out is not df  # a copy, not the same object
