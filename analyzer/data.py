"""데이터 소스 선택.

KRX_ID / KRX_PW 환경 변수가 있으면 KRX(pykrx), 없으면 네이버 금융을 쓴다.
DATA_SOURCE=naver|krx 로 강제할 수도 있다.
"""
import os
from datetime import timedelta

import pandas as pd


def source_name() -> str:
    forced = os.getenv("DATA_SOURCE", "").lower()
    if forced in ("naver", "krx"):
        return forced
    return "krx" if os.getenv("KRX_ID") and os.getenv("KRX_PW") else "naver"


def _src():
    if source_name() == "krx":
        from . import data_krx as m
    else:
        from . import data_naver as m
    return m


def ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def lookback_start(end: str, days: int) -> str:
    return ymd(pd.Timestamp(end) - timedelta(days=days))


def has_point_in_time_fundamentals() -> bool:
    """과거 시점 PER/PBR 조회 가능 여부 (네이버는 현재 값만 제공)."""
    return source_name() == "krx"


def latest_business_day(date=None):
    return _src().latest_business_day(date)


def universe(date, top_n=None):
    return _src().universe(date, top_n)


def close_prices(tickers, start, end):
    return _src().close_prices(tickers, start, end)


FIELDS = {"open": "시가", "high": "고가", "low": "저가", "close": "종가", "volume": "거래량"}


def build_panel(frames: dict) -> dict:
    """{ticker: 일봉 DataFrame(시가·고가·저가·종가·거래량)} -> {필드: 날짜 x 종목 DataFrame}."""
    panel = {}
    for key, col in FIELDS.items():
        wide = pd.DataFrame({t: df[col] for t, df in frames.items() if col in df}).sort_index().astype(float)
        if key != "volume":
            wide = wide.where(wide > 0)  # 거래정지일 0 가격은 결측
        panel[key] = wide
    return panel


def ohlcv_panel(tickers, start, end) -> dict:
    return build_panel(_src().ohlcv_frames(tickers, start, end))


def fundamentals(date, tickers=None):
    if source_name() == "naver":
        return _src().fundamentals(date, tickers)
    return _src().fundamentals(date)


def benchmark(start, end):
    return _src().benchmark(start, end)
