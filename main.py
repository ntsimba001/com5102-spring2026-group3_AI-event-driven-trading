from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import BacktestConfig, IntradayBacktester
from src.data_loader import MarketDataLoader
from src.news import align_news_to_market_data, fetch_news_headlines
from src.sentiment import SentimentAnalyzer, score_news_frame, simulate_sentiment_series
from src.utils import ensure_directory, save_json, save_metrics_csv

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI event-driven intraday trading system")
    parser.add_argument("--ticker", default="TSLA", help="Ticker symbol to backtest")
    parser.add_argument("--headline", default="", help="Optional news headline to score and use as static sentiment")
    parser.add_argument("--use-real-news", action="store_true", help="Fetch recent RSS headlines and align them to market data")
    parser.add_argument("--news-limit", type=int, default=20, help="Maximum number of recent headlines to fetch")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="Starting capital")
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop-loss percent as decimal")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take-profit percent as decimal")
    parser.add_argument("--trailing-stop", type=float, default=0.0075, help="Trailing stop percent as decimal")
    parser.add_argument("--allocation", type=float, default=0.95, help="Max capital allocation per trade")
    parser.add_argument("--cooldown", type=int, default=10, help="Cooldown in minutes after each exit")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible simulated data")
    return parser.parse_args()


def attach_sentiment(
    frame: pd.DataFrame,
    headline: str,
    seed: int,
    ticker: str,
    use_real_news: bool,
    news_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = frame.copy()
    news_frame = pd.DataFrame(columns=["published_at", "title", "source", "url", "sentiment_score"])

    if headline.strip():
        analyzer = SentimentAnalyzer()
        sentiment_score = analyzer.score_text(headline)
        enriched["Sentiment"] = sentiment_score
        return enriched, news_frame

    if use_real_news:
        raw_news = fetch_news_headlines(ticker=ticker, limit=news_limit)
        if not raw_news.empty:
            scored_news = score_news_frame(raw_news)
            enriched["Sentiment"] = align_news_to_market_data(
                market_data=enriched,
                news_data=scored_news,
                sentiment_scores=scored_news["sentiment_score"],
            )
            news_frame = scored_news
            return enriched, news_frame

    enriched["Sentiment"] = simulate_sentiment_series(enriched, seed=seed)
    return enriched, news_frame


def save_outputs(
    ticker: str,
    market_data: pd.DataFrame,
    news_data: pd.DataFrame,
    metrics: dict,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> None:
    results_dir = ensure_directory("results")
    market_data.to_csv(results_dir / f"{ticker.lower()}_market_data.csv", index=False)
    if not news_data.empty:
        news_data.to_csv(results_dir / f"{ticker.lower()}_news.csv", index=False)
    trades.to_csv(results_dir / f"{ticker.lower()}_trades.csv", index=False)
    equity_curve.to_csv(results_dir / f"{ticker.lower()}_equity_curve.csv", index=False)
    save_metrics_csv(results_dir / f"{ticker.lower()}_metrics.csv", metrics)
    save_json(results_dir / f"{ticker.lower()}_metrics.json", metrics)

    if plt is not None and not equity_curve.empty:
        figure_path = Path(results_dir) / f"{ticker.lower()}_equity_curve.png"
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve["Timestamp"], equity_curve["Equity"], linewidth=1.2)
        plt.title(f"{ticker} Equity Curve")
        plt.xlabel("Timestamp")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()


def main() -> None:
    args = parse_args()

    loader = MarketDataLoader(seed=args.seed)
    market_data = loader.load_intraday_data(args.ticker)
    market_data, news_data = attach_sentiment(
        market_data,
        args.headline,
        args.seed,
        args.ticker,
        args.use_real_news,
        args.news_limit,
    )

    backtester = IntradayBacktester(
        BacktestConfig(
            initial_capital=args.initial_capital,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            trailing_stop_pct=args.trailing_stop,
            max_allocation_pct=args.allocation,
            cooldown_minutes=args.cooldown,
        )
    )
    result = backtester.run(market_data)
    save_outputs(args.ticker, market_data, news_data, result.metrics, result.trades, result.equity_curve)

    print(f"Final Capital: {result.metrics['final_capital']:.2f}")
    if args.use_real_news:
        print(f"News Headlines Used: {len(news_data)}")
    print("Metrics:")
    print(result.metrics)


if __name__ == "__main__":
    main()
