import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TIME_COL,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_raw(path: str | None = None) -> pd.DataFrame:
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day, day_of_week, is_weekend derived from timestamp."""
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    df["hour_of_day"] = df[TIME_COL].dt.hour
    df["day_of_week"] = df[TIME_COL].dt.dayofweek  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def split_train_test(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from data.")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_df, test_df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_path = PROCESSED_DATA_DIR / "sessions_train.csv"
    test_path = PROCESSED_DATA_DIR / "sessions_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to:  {test_path}")


def main() -> None:
    df = load_raw()
    print(f"Loaded raw data: {df.shape}")

    df_feat = engineer_time_features(df)
    print("Engineered time features:")
    print(df_feat[[TIME_COL, "hour_of_day", "day_of_week", "is_weekend"]].head())

    train_df, test_df = split_train_test(df_feat)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    print("Train satisfaction distribution:")
    print(train_df[TARGET_COL].value_counts(normalize=True).rename("train_ratio"))

    print("Test satisfaction distribution:")
    print(test_df[TARGET_COL].value_counts(normalize=True).rename("test_ratio"))

    save_processed(train_df, test_df)


if __name__ == "__main__":
    main()
