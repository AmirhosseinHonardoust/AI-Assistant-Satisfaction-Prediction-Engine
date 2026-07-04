"""Synthetic dataset generator for AI-assistant satisfaction.

The dataset is **synthetic by design**. Satisfaction is drawn from a documented
latent scoring process so the modelling pipeline has a real (but noisy) signal
to recover. Because the data-generating process is known, the SHAP drivers and
the behavioural "insights" in the README are verifiable rather than anecdotal:
a good model should recover the effect sizes defined below.

Latent satisfaction score (before binning to a 1-5 rating)::

    latent = device_effect
           + usage_effect
           + model_effect
           + 0.06 * (session_length_minutes - mean)
           + 0.5  * is_weekend
           + Normal(0, NOISE_SD)

The continuous score is binned into five roughly balanced classes by quintile,
so chance accuracy is ~0.20 and the majority baseline is ~0.20 too.

Run with::

    python -m src.generate_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, RAW_DATA_PATH

N_SESSIONS = 1500
NOISE_SD = 0.85

DEVICES = ["Desktop", "Mobile", "Tablet", "Smart Speaker"]
DEVICE_P = [0.40, 0.35, 0.15, 0.10]
DEVICE_EFFECT = {"Smart Speaker": 1.1, "Desktop": 0.7, "Tablet": 0.1, "Mobile": -0.9}

USAGE = [
    "Coding",
    "Productivity",
    "Research",
    "Writing",
    "Education",
    "Daily Tasks",
    "Entertainment",
]
USAGE_EFFECT = {
    "Coding": 1.0,
    "Research": 0.6,
    "Productivity": 0.4,
    "Education": 0.2,
    "Writing": 0.0,
    "Daily Tasks": -0.3,
    "Entertainment": -0.6,
}

MODELS = ["GPT-4o", "GPT-5", "GPT-5.1", "Mini", "o1"]
MODEL_P = [0.25, 0.20, 0.20, 0.20, 0.15]
MODEL_EFFECT = {"GPT-5.1": 1.1, "GPT-5": 0.7, "GPT-4o": 0.2, "o1": 0.0, "Mini": -0.8}


def generate(n: int = N_SESSIONS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate ``n`` synthetic sessions with a designed satisfaction signal."""
    rng = np.random.default_rng(seed)

    device = rng.choice(DEVICES, n, p=DEVICE_P)
    usage_category = rng.choice(USAGE, n)
    model = rng.choice(MODELS, n, p=MODEL_P)

    prompt_length = rng.integers(5, 200, n)
    session_length = np.round(rng.gamma(2.0, 4.0, n) + 0.5, 2)
    tokens_used = rng.integers(20, 1500, n)

    start = pd.Timestamp("2025-01-01")
    minutes = rng.integers(0, 90 * 24 * 60, n)
    ts = start + pd.to_timedelta(minutes, unit="m")
    is_weekend = np.isin(ts.dayofweek, [5, 6]).astype(int)

    device_effect = np.vectorize(DEVICE_EFFECT.get)(device)
    usage_effect = np.vectorize(USAGE_EFFECT.get)(usage_category)
    model_effect = np.vectorize(MODEL_EFFECT.get)(model)
    session_effect = 0.06 * (session_length - session_length.mean())
    weekend_effect = 0.5 * is_weekend
    noise = rng.normal(0.0, NOISE_SD, n)

    latent = device_effect + usage_effect + model_effect + session_effect + weekend_effect + noise

    quintiles = np.quantile(latent, [0.2, 0.4, 0.6, 0.8])
    rating = np.digitize(latent, quintiles) + 1

    return pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "usage_category": usage_category,
            "prompt_length": prompt_length,
            "session_length_minutes": session_length,
            "satisfaction_rating": rating.astype(int),
            "assistant_model": model,
            "tokens_used": tokens_used,
        }
    )


def main() -> None:
    df = generate()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(df)} sessions -> {RAW_DATA_PATH}")
    print("Class balance:")
    print(df["satisfaction_rating"].value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    main()
