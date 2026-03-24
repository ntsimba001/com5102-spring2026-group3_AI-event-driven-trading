# com5102-spring2026-group3_AI-event-driven-trading
Intraday trading strategy using market data, AI sentiment from news/social media
# AI Event-Driven Intraday Trading System

This repository contains a simplified event-driven intraday trading prototype that combines intraday price and volume data with AI-driven sentiment scoring to generate buy and sell signals, backtest the strategy, and export performance outputs.

## Features

- Downloads 1-minute market data with `yfinance` when available
- Fetches recent real news headlines from RSS feeds when requested
- Falls back to synthetic intraday data so the system remains runnable offline
- Scores an optional headline with a Hugging Face sentiment model
- Falls back to deterministic lexicon-based sentiment when transformer dependencies are unavailable
- Applies rule-based signal generation using sentiment, volume spikes, short-term momentum, and moving-average trend confirmation
- Runs a no-overnight intraday backtest with stop-loss, take-profit, trailing stop, cooldown, and capped capital allocation
- Exports trades, equity curve, and metrics to `results/`

## Project Structure

```text
.
├── data/
├── main.py
├── README.md
├── requirements.txt
├── results/
└── src/
    ├── backtest.py
    ├── data_loader.py
    ├── sentiment.py
    ├── signal.py
    └── utils.py
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --ticker TSLA
```

To apply a static sentiment value from a headline:

```bash
python main.py --ticker TSLA --headline "TSLA posts strong growth and record profit"
```

To use real recent news headlines:

```bash
python main.py --ticker TSLA --use-real-news --news-limit 20
```

To tune the strategy:

```bash
python main.py --ticker TSLA --stop-loss 0.01 --take-profit 0.02 --trailing-stop 0.0075 --allocation 0.95 --cooldown 10
```

## Strategy Logic

- `BUY` requires positive sentiment, elevated volume, positive momentum, and fast moving average above slow moving average
- `SELL` requires strongly negative sentiment with volume confirmation and bearish trend
- `EXIT` closes long exposure on bearish reversals even without a full sell signal
- Positions are force-closed by the end of each trading day
- Each trade uses at most the configured capital allocation and enforces a cooldown after exit

## Outputs

The system writes the following files to `results/`:

- `<ticker>_market_data.csv`
- `<ticker>_trades.csv`
- `<ticker>_equity_curve.csv`
- `<ticker>_metrics.csv`
- `<ticker>_metrics.json`
- `<ticker>_news.csv` when real news is fetched successfully
- `<ticker>_equity_curve.png` when `matplotlib` is installed

Metrics include `final_capital`, `total_return`, `win_rate`, `avg_trade_return`, `max_drawdown`, and `num_trades`.
