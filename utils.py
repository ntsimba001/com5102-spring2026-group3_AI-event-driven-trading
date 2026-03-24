from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_recent_trading_days(count: int = 5) -> pd.DatetimeIndex:
    end = pd.Timestamp.now().normalize()
    days = pd.bdate_range(end=end, periods=count)
    return days


def build_intraday_index(days: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    timestamps: list[pd.Timestamp] = []
    for day in days:
        start = day + pd.Timedelta(hours=9, minutes=30)
        end = day + pd.Timedelta(hours=15, minutes=59)
        timestamps.extend(pd.date_range(start=start, end=end, freq="1min"))
    return pd.DatetimeIndex(timestamps, name="Timestamp")


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_metrics_csv(path: str | Path, payload: dict) -> None:
    frame = pd.DataFrame([payload])
    frame.to_csv(path, index=False)


def clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def rolling_average_volume(volume: pd.Series, window: int = 30) -> pd.Series:
    rolling = volume.rolling(window=window, min_periods=1).mean()
    return rolling.replace(0, np.nan).bfill().fillna(0)
