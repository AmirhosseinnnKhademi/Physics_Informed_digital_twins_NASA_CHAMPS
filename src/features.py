"""Shared rolling feature engineering utilities."""
import numpy as np
import pandas as pd


def add_rolling_features(all_df, stat_cols, window):
    """Add per-engine rolling slope and delta for every stat column.

    For each column two features are added:
      {col}_slope : linear trend (units/cycle) over the last `window` cycles
      {col}_delta : value change from `window` cycles ago to now

    Computed strictly within each engine — no cross-engine leakage.
    Partial windows at the start use however many cycles are available.
    """
    frames = []
    for _, unit_df in all_df.groupby("unit", sort=True):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        n = len(unit_df)

        extra = {}
        for col in stat_cols:
            vals   = unit_df[col].values.astype(np.float64)
            slopes = np.zeros(n, dtype=np.float32)
            deltas = np.zeros(n, dtype=np.float32)

            for i in range(n):
                start = max(0, i - window + 1)
                y = vals[start: i + 1]
                w = len(y)

                if w >= 2:
                    x    = np.arange(w, dtype=np.float64)
                    xm   = x.mean();  ym = y.mean()
                    denom = ((x - xm) ** 2).sum()
                    slopes[i] = float(((x - xm) * (y - ym)).sum() / denom) if denom > 0 else 0.0

                ref_idx  = i - window + 1 if i >= window - 1 else 0
                deltas[i] = float(vals[i] - vals[ref_idx])

            extra[f"{col}_slope"] = slopes
            extra[f"{col}_delta"] = deltas

        frames.append(pd.concat([unit_df, pd.DataFrame(extra, index=unit_df.index)], axis=1))

    return pd.concat(frames, ignore_index=True)
