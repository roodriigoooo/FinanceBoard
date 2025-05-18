"""
Helper utility functions for the Finance Board application.
"""

import math
import numpy as np
import pandas as pd


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single‑level column index."""
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels == 2:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(map(str, tup)).strip() for tup in df.columns]
    return df


def pick_price_col(df: pd.DataFrame) -> str | None:
    """Return the preferred price column present in *df* (Adj Close > Close)."""
    for col in ("Adj Close", "Close"):
        if col in df.columns:
            return col
    return None


def annualised(total_ret: float, n_days: int) -> float:
    """Convert a total return over n_days to an annualized return."""
    return (1 + total_ret) ** (365 / n_days) - 1 if n_days else np.nan


def compute_metrics(price: pd.Series, volume: pd.Series) -> dict:
    """Compute various financial metrics from price and volume data."""
    rets = price.pct_change().dropna()
    n = len(price)

    running_max = price.cummax()
    max_dd = (price / running_max - 1).min()

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rsi_series = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())

    # MACD Calculation
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    up, down = (rets > 0).sum(), (rets < 0).sum()

    return {
        "total": price.iloc[-1] / price.iloc[0] - 1 if len(price) > 0 else np.nan,
        "ann": annualised(price.iloc[-1] / price.iloc[0] - 1, n) if len(price) > 0 else np.nan,
        "avg_up": rets[rets > 0].mean() if not rets[rets > 0].empty else np.nan,
        "avg_down": rets[rets < 0].mean() if not rets[rets < 0].empty else np.nan,
        "vol": rets.std() * math.sqrt(252) if not rets.empty else np.nan,
        "mdd": max_dd if not pd.isna(max_dd) else np.nan,
        "var": rets.quantile(0.05) if not rets.empty else np.nan,
        "cvar": rets[rets <= rets.quantile(0.05)].mean() if not rets.empty and not rets[
            rets <= rets.quantile(0.05)].empty else np.nan,
        "rsi_series": rsi_series,  # Full RSI series
        "macd_line": macd_line,  # Full MACD line
        "signal_line": signal_line,  # Full Signal line
        "macd_histogram": macd_histogram,  # Full MACD histogram
        "sharpe": (annualised(price.iloc[-1] / price.iloc[0] - 1, n) / (rets.std() * math.sqrt(252))) if len(
            price) > 0 and rets.std() > 0 else np.nan,
        "up_down": up / down if down else np.nan,
        "avg_vol": volume.mean() if not volume.empty else np.nan,
        "vol_trend": volume.tail(5).mean() / volume.mean() - 1 if not volume.empty and volume.mean() > 0 and len(
            volume) >= 5 else np.nan,
        "rets": rets,  # keep for later analysis
        "rsi": rsi_series.iloc[-1] if not rsi_series.empty else np.nan,  # Keep last RSI for KPIs if needed
        "macd": macd_histogram.iloc[-1] if not macd_histogram.empty else np.nan,  # Keep last MACD hist for KPIs
    }


# Formatting helpers
def format_pct(v):
    """Format a value as a percentage."""
    return "–" if pd.isna(v) else f"{v * 100:,.2f}%"


def format_num(v):
    """Format a value as a number."""
    return "–" if pd.isna(v) else f"{v:,.0f}"


def format_currency(v, precision=2):
    """Format a value as currency."""
    return "–" if pd.isna(v) else f"${v:,.{precision}f}"


def format_with_suffix(num):
    """Format large numbers with suffix K, M, B, T."""
    if pd.isna(num):
        return "–"
    
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    suffix = ['', 'K', 'M', 'B', 'T'][magnitude]
    return f"${num:.1f}{suffix}"
