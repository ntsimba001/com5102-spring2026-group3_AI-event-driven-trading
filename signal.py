from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    positive_threshold: float = 0.55
    negative_threshold: float = -0.55
    volume_multiplier: float = 1.3
    fast_ma_window: int = 5
    slow_ma_window: int = 20
    momentum_window: int = 3
    reversal_buffer: float = 0.001


def prepare_signal_features(market_data: pd.DataFrame, config: SignalConfig | None = None) -> pd.DataFrame:
    config = config or SignalConfig()
    frame = market_data.copy()

    frame["FastMA"] = frame["Close"].rolling(window=config.fast_ma_window, min_periods=1).mean()
    frame["SlowMA"] = frame["Close"].rolling(window=config.slow_ma_window, min_periods=1).mean()
    frame["Momentum"] = frame["Close"].pct_change(periods=config.momentum_window).fillna(0.0)
    frame["VolumeRatio"] = (frame["Volume"] / frame["AverageVolume"].replace(0, pd.NA)).fillna(0.0)
    frame["TrendGap"] = (frame["FastMA"] / frame["SlowMA"].replace(0, pd.NA) - 1).fillna(0.0)
    return frame


def generate_signal(
    sentiment_score: float,
    current_volume: float,
    average_volume: float,
    momentum: float = 0.0,
    trend_gap: float = 0.0,
    config: SignalConfig | None = None,
) -> str:
    config = config or SignalConfig()
    volume_threshold = average_volume * config.volume_multiplier
    bullish_trend = trend_gap > config.reversal_buffer
    bearish_trend = trend_gap < -config.reversal_buffer
    positive_momentum = momentum > 0
    negative_momentum = momentum < 0

    if (
        sentiment_score > config.positive_threshold
        and current_volume > volume_threshold
        and bullish_trend
        and positive_momentum
    ):
        return "BUY"
    if (
        sentiment_score < config.negative_threshold
        and current_volume > volume_threshold
        and bearish_trend
        and negative_momentum
    ):
        return "SELL"
    if bearish_trend and negative_momentum and sentiment_score < 0:
        return "EXIT"
    return "HOLD"
