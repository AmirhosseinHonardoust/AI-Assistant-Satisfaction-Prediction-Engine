"""Shared time-feature engineering.

A single source of truth for deriving calendar features from the raw
``timestamp`` column, used by data prep, batch scoring and the dashboard so the
three stay in sync.
"""

from __future__ import annotations

import pandas as pd

from .config import TIME_COL


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``hour_of_day``, ``day_of_week`` and
    ``is_weekend`` derived from :data:`~src.config.TIME_COL`.

    If the timestamp column is absent the frame is returned unchanged (a copy),
    which lets the scoring paths accept CSVs that already carry the engineered
    columns.
    """
    if TIME_COL not in df.columns:
        return df.copy()

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["hour_of_day"] = df[TIME_COL].dt.hour
    df["day_of_week"] = df[TIME_COL].dt.dayofweek  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df
