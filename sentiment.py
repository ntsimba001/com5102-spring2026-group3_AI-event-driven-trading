from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import clamp

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None


POSITIVE_WORDS = {
    "beat",
    "breakthrough",
    "growth",
    "gain",
    "strong",
    "surge",
    "upgrade",
    "bullish",
    "profit",
    "record",
    "optimistic",
}

NEGATIVE_WORDS = {
    "downgrade",
    "miss",
    "fraud",
    "weak",
    "drop",
    "lawsuit",
    "bearish",
    "loss",
    "decline",
    "risk",
    "investigation",
}


@dataclass
class SentimentAnalyzer:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def score_text(self, text: str) -> float:
        normalized = text.strip()
        if not normalized:
            return 0.0

        if self._pipeline is None:
            self._pipeline = self._build_pipeline()

        if self._pipeline is not None:
            try:
                result = self._pipeline(normalized[:512])[0]
                score = float(result["score"])
                return score if result["label"].upper().startswith("POS") else -score
            except Exception:
                pass

        return self._score_with_lexicon(normalized)

    def _build_pipeline(self):
        if pipeline is None:
            return None
        try:
            return pipeline("sentiment-analysis", model=self.model_name)
        except Exception:
            return None

    def _score_with_lexicon(self, text: str) -> float:
        tokens = [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split()]
        positives = sum(token in POSITIVE_WORDS for token in tokens)
        negatives = sum(token in NEGATIVE_WORDS for token in tokens)
        total = positives + negatives
        if total == 0:
            return 0.0
        return clamp((positives - negatives) / total, -1.0, 1.0)


def simulate_sentiment_series(
    market_data: pd.DataFrame,
    seed: int = 42,
    event_probability: float = 0.03,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    close_returns = market_data["Close"].pct_change().fillna(0.0)
    baseline = np.tanh(close_returns * 40)

    event_shocks = np.zeros(len(market_data))
    event_indices = rng.random(len(market_data)) < event_probability
    event_shocks[event_indices] = rng.uniform(-1.0, 1.0, size=event_indices.sum())
    smoothed_events = pd.Series(event_shocks).replace(0, np.nan).ffill(limit=20).fillna(0.0)

    sentiment = (baseline * 0.35) + (smoothed_events * 0.95)
    return sentiment.clip(-1.0, 1.0).rename("Sentiment")


def score_news_frame(news_data: pd.DataFrame, analyzer: SentimentAnalyzer | None = None) -> pd.DataFrame:
    if news_data.empty:
        return news_data.copy()

    analyzer = analyzer or SentimentAnalyzer()
    scored = news_data.copy()
    scored["sentiment_score"] = scored["title"].map(analyzer.score_text)
    return scored
