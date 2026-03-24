from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import build_intraday_index, get_recent_trading_days, rolling_average_volume

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None


@dataclass
class MarketDataLoader:
    period: str = "5d"
    interval: str = "1m"
    seed: int = 42

    def load_intraday_data(self, ticker: str) -> pd.DataFrame:
        frame = self._download_market_data(ticker)
        if frame.empty:
            frame = self._generate_synthetic_data(ticker)

        frame = frame.copy()
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"]).dt.tz_localize(None)
        frame = frame.sort_values("Timestamp").reset_index(drop=True)
        frame["AverageVolume"] = rolling_average_volume(frame["Volume"])
        return frame

    def _download_market_data(self, ticker: str) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()

        try:
            history = yf.download(
                tickers=ticker,
                period=self.period,
                interval=self.interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame()

        if history.empty:
            return pd.DataFrame()

        history = history.reset_index()
        rename_map = {
            history.columns[0]: "Timestamp",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        history = history.rename(columns=rename_map)
        expected = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        if not all(column in history.columns for column in expected):
            return pd.DataFrame()
        return history[expected]

    def _generate_synthetic_data(self, ticker: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + sum(ord(char) for char in ticker))
        trading_days = get_recent_trading_days(count=5)
        index = build_intraday_index(trading_days)
        samples = len(index)

        base_price = 100 + (sum(ord(char) for char in ticker[:4]) % 150)
        drift = rng.normal(0.00008, 0.0006, size=samples)
        shocks = rng.normal(0, 0.0035, size=samples)
        returns = drift + shocks
        close = base_price * np.exp(np.cumsum(returns))
        open_prices = np.concatenate(([close[0]], close[:-1]))
        high = np.maximum(open_prices, close) * (1 + rng.uniform(0.0001, 0.002, size=samples))
        low = np.minimum(open_prices, close) * (1 - rng.uniform(0.0001, 0.002, size=samples))

        intraday_profile = np.linspace(1.4, 0.8, samples)
        volume = rng.integers(8_000, 30_000, size=samples) * intraday_profile
        event_spikes = rng.random(samples) < 0.025
        volume[event_spikes] *= rng.uniform(2.0, 4.5, size=event_spikes.sum())
        volume = np.maximum(volume.astype(int), 1_000)

        return pd.DataFrame(
            {
                "Timestamp": index,
                "Open": open_prices,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
