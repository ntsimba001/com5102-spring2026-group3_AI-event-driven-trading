from __future__ import annotations

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


USER_AGENT = "Mozilla/5.0 (compatible; IntradayTradingBot/1.0)"


@dataclass(frozen=True)
class NewsItem:
    published_at: pd.Timestamp
    title: str
    source: str
    url: str


def fetch_news_headlines(ticker: str, limit: int = 20) -> pd.DataFrame:
    query = f"{ticker} stock market"
    feeds = [
        ("Google News", f"https://news.google.com/rss/search?q={quote_plus(query)}"),
        ("Google News", f"https://news.google.com/rss/search?q={quote_plus(ticker)}"),
    ]

    collected: list[NewsItem] = []
    for default_source, url in feeds:
        items = _parse_rss_feed(url, default_source=default_source, limit=limit)
        collected.extend(items)
        if len(collected) >= limit:
            break

    if not collected:
        return pd.DataFrame(columns=["published_at", "title", "source", "url"])

    frame = pd.DataFrame(
        {
            "published_at": [item.published_at for item in collected],
            "title": [item.title for item in collected],
            "source": [item.source for item in collected],
            "url": [item.url for item in collected],
        }
    )
    frame = frame.sort_values("published_at").drop_duplicates(subset=["title"]).tail(limit).reset_index(drop=True)
    return frame


def _parse_rss_feed(url: str, default_source: str, limit: int) -> list[NewsItem]:
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=10) as response:
            payload = response.read()
    except Exception:
        return []

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []

    items: list[NewsItem] = []
    for item in root.findall(".//item"):
        parsed = _parse_item(item, default_source)
        if parsed is not None:
            items.append(parsed)
        if len(items) >= limit:
            break
    return items


def _parse_item(node: ET.Element, default_source: str) -> NewsItem | None:
    title = _get_child_text(node, "title")
    if not title:
        return None

    link = _get_child_text(node, "link")
    source = default_source
    source_node = node.find("source")
    if source_node is not None and source_node.text:
        source = source_node.text.strip()

    published_raw = _get_child_text(node, "pubDate")
    published_at = _parse_timestamp(published_raw)
    if published_at is None:
        return None

    return NewsItem(
        published_at=published_at.tz_localize(None),
        title=title.strip(),
        source=source,
        url=(link or "").strip(),
    )


def _get_child_text(node: ET.Element, name: str) -> str:
    child = node.find(name)
    return child.text.strip() if child is not None and child.text else ""


def _parse_timestamp(raw: str) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        return pd.Timestamp(parsedate_to_datetime(raw))
    except Exception:
        return None


def align_news_to_market_data(
    market_data: pd.DataFrame,
    news_data: pd.DataFrame,
    sentiment_scores: Iterable[float],
    freshness_minutes: int = 180,
) -> pd.Series:
    if market_data.empty:
        return pd.Series(dtype=float, name="Sentiment")
    if news_data.empty:
        return pd.Series(0.0, index=market_data.index, name="Sentiment")

    news_frame = news_data.copy()
    news_frame["sentiment_score"] = list(sentiment_scores)
    news_frame = news_frame.sort_values("published_at")

    market = market_data[["Timestamp"]].copy().sort_values("Timestamp")
    aligned = pd.merge_asof(
        market,
        news_frame[["published_at", "sentiment_score"]],
        left_on="Timestamp",
        right_on="published_at",
        direction="backward",
        tolerance=pd.Timedelta(minutes=freshness_minutes),
    )
    sentiment = aligned["sentiment_score"].fillna(0.0)
    return pd.Series(sentiment.values, index=market_data.index, name="Sentiment")
