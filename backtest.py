from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .signal import SignalConfig, generate_signal, prepare_signal_features


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = 0.0075
    max_allocation_pct: float = 0.95
    cooldown_minutes: int = 10
    signal_config: SignalConfig = field(default_factory=SignalConfig)


@dataclass
class BacktestResult:
    metrics: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


class IntradayBacktester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, market_data: pd.DataFrame) -> BacktestResult:
        frame = prepare_signal_features(market_data, self.config.signal_config)
        frame["TradeDate"] = frame["Timestamp"].dt.date

        capital = self.config.initial_capital
        shares = 0
        entry_price = 0.0
        peak_price = 0.0
        entry_time = None
        last_exit_time = None
        equity_records: list[dict] = []
        trades: list[dict] = []

        for index, row in frame.iterrows():
            price = float(row["Close"])
            signal = generate_signal(
                sentiment_score=float(row["Sentiment"]),
                current_volume=float(row["Volume"]),
                average_volume=float(row["AverageVolume"]),
                momentum=float(row["Momentum"]),
                trend_gap=float(row["TrendGap"]),
                config=self.config.signal_config,
            )

            should_force_close = self._is_end_of_day(frame, index)
            risk_exit = self._risk_exit(price, entry_price, peak_price) if shares > 0 else False
            cooldown_active = self._cooldown_active(last_exit_time, row["Timestamp"])

            if shares == 0 and signal == "BUY" and not cooldown_active:
                budget = capital * self.config.max_allocation_pct
                purchasable_shares = int(budget // price)
                if purchasable_shares > 0:
                    shares = purchasable_shares
                    capital -= shares * price
                    entry_price = price
                    peak_price = price
                    entry_time = row["Timestamp"]
            elif shares > 0:
                peak_price = max(peak_price, price)
                should_exit = signal in {"SELL", "EXIT"} or should_force_close or risk_exit
                if not should_exit:
                    equity = capital + (shares * price)
                    equity_records.append(
                        {
                            "Timestamp": row["Timestamp"],
                            "Equity": round(equity, 4),
                            "Cash": round(capital, 4),
                            "PositionShares": shares,
                            "Signal": signal,
                        }
                    )
                    continue

                capital += shares * price
                pnl = (price - entry_price) * shares
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": row["Timestamp"],
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(price, 4),
                        "shares": shares,
                        "pnl": round(pnl, 4),
                        "return_pct": round((price - entry_price) / entry_price, 6),
                        "holding_minutes": int((row["Timestamp"] - entry_time).total_seconds() // 60),
                        "exit_reason": self._get_exit_reason(signal, should_force_close, risk_exit),
                    }
                )
                shares = 0
                entry_price = 0.0
                peak_price = 0.0
                entry_time = None
                last_exit_time = row["Timestamp"]

            equity = capital + (shares * price)
            equity_records.append(
                {
                    "Timestamp": row["Timestamp"],
                    "Equity": round(equity, 4),
                    "Cash": round(capital, 4),
                    "PositionShares": shares,
                    "Signal": signal,
                }
            )

        trades_frame = pd.DataFrame(trades)
        equity_curve = pd.DataFrame(equity_records)
        metrics = self._calculate_metrics(capital, shares, frame, trades_frame, equity_curve)
        return BacktestResult(metrics=metrics, trades=trades_frame, equity_curve=equity_curve)

    def _calculate_metrics(
        self,
        capital: float,
        shares: int,
        frame: pd.DataFrame,
        trades_frame: pd.DataFrame,
        equity_curve: pd.DataFrame,
    ) -> dict:
        final_equity = capital
        if shares > 0 and not frame.empty:
            final_equity += shares * float(frame.iloc[-1]["Close"])

        num_trades = int(len(trades_frame))
        win_rate = float((trades_frame["pnl"] > 0).mean()) if num_trades else 0.0
        total_return = ((final_equity / self.config.initial_capital) - 1) if self.config.initial_capital else 0.0
        avg_trade_return = float(trades_frame["return_pct"].mean()) if num_trades else 0.0
        max_drawdown = self._max_drawdown(equity_curve)

        return {
            "initial_capital": round(self.config.initial_capital, 2),
            "final_capital": round(final_equity, 2),
            "total_return": round(total_return, 6),
            "win_rate": round(win_rate, 6),
            "avg_trade_return": round(avg_trade_return, 6),
            "max_drawdown": round(max_drawdown, 6),
            "num_trades": num_trades,
        }

    def _is_end_of_day(self, frame: pd.DataFrame, index: int) -> bool:
        if index == len(frame) - 1:
            return True
        current_date = frame.iloc[index]["TradeDate"]
        next_date = frame.iloc[index + 1]["TradeDate"]
        return current_date != next_date

    def _risk_exit(self, price: float, entry_price: float, peak_price: float) -> bool:
        if entry_price <= 0:
            return False
        move = (price - entry_price) / entry_price
        stop_loss_hit = self.config.stop_loss_pct is not None and move <= -abs(self.config.stop_loss_pct)
        take_profit_hit = self.config.take_profit_pct is not None and move >= abs(self.config.take_profit_pct)
        trailing_stop_hit = False
        if self.config.trailing_stop_pct is not None and peak_price > 0:
            trailing_move = (price - peak_price) / peak_price
            trailing_stop_hit = trailing_move <= -abs(self.config.trailing_stop_pct)
        return bool(stop_loss_hit or take_profit_hit or trailing_stop_hit)

    def _cooldown_active(self, last_exit_time: pd.Timestamp | None, current_time: pd.Timestamp) -> bool:
        if last_exit_time is None:
            return False
        elapsed_minutes = (current_time - last_exit_time).total_seconds() / 60
        return elapsed_minutes < self.config.cooldown_minutes

    @staticmethod
    def _max_drawdown(equity_curve: pd.DataFrame) -> float:
        if equity_curve.empty:
            return 0.0
        running_peak = equity_curve["Equity"].cummax()
        drawdowns = (equity_curve["Equity"] - running_peak) / running_peak.replace(0, pd.NA)
        return abs(float(drawdowns.min(skipna=True))) if not drawdowns.empty else 0.0

    @staticmethod
    def _get_exit_reason(signal: str, should_force_close: bool, risk_exit: bool) -> str:
        if risk_exit:
            return "RISK"
        if should_force_close:
            return "EOD"
        return signal
