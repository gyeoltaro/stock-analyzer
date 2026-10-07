"""기술적 지표. 입력은 날짜 x 종목 형태의 종가 DataFrame."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return close.rolling(window, min_periods=window).mean()


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder 방식 RSI (0~100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 하락이 전혀 없으면 RSI 100
    return out.where(loss != 0, 100.0).where(gain.notna())


def momentum(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """skip일 전 가격 / lookback일 전 가격 - 1."""
    return close.shift(skip) / close.shift(lookback) - 1


def volatility(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """연율화 변동성."""
    return close.pct_change(fill_method=None).rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)
