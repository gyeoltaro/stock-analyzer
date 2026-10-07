"""pykrx를 이용한 KRX 데이터 수집.

pykrx 1.2.9부터 KRX 정보데이터시스템(data.krx.co.kr) 로그인이 필요하다.
환경 변수 KRX_ID, KRX_PW 를 설정해야 한다.
"""
import time
from datetime import datetime, timedelta

import pandas as pd

KOSPI200 = "1028"


def _krx():
    from pykrx import stock  # 지연 import: 테스트에서 네트워크 없이 다른 모듈 사용
    return stock


def ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def latest_business_day(date=None) -> str:
    date = ymd(date or datetime.now())
    return _krx().get_nearest_business_day_in_a_week(date, prev=True)


def universe(date: str, top_n: int | None = None, min_trading_value: float = 1e8) -> pd.DataFrame:
    """KOSPI+KOSDAQ 종목 (top_n 이 없거나 0이면 전 종목). 거래대금이 너무 적은 종목은 제외.

    반환: index=ticker, columns=[name, market_cap, trading_value]
    """
    stock = _krx()
    cap = stock.get_market_cap(date, market="ALL")
    cap = cap.rename(columns={"시가총액": "market_cap", "거래대금": "trading_value"})
    cap = cap[cap["trading_value"] >= min_trading_value]
    # 우선주(티커 끝자리 0이 아님)와 스팩·ETF 성격 종목 제외
    cap = cap[cap.index.str.endswith("0")]
    cap = cap.sort_values("market_cap", ascending=False)
    cap = (cap.head(top_n) if top_n else cap).copy()
    cap["name"] = [stock.get_market_ticker_name(t) for t in cap.index]
    cap = cap[~cap["name"].str.contains("스팩|리츠", na=False)]
    return cap[["name", "market_cap", "trading_value"]]


def close_prices(tickers, start: str, end: str, pause: float = 0.2) -> pd.DataFrame:
    """종목별 수정주가 종가. index=날짜, columns=ticker."""
    stock = _krx()
    series = {}
    for i, t in enumerate(tickers, 1):
        try:
            df = stock.get_market_ohlcv(start, end, t, adjusted=True)
        except Exception as e:  # 개별 종목 실패는 건너뜀
            print(f"[warn] {t} 시세 조회 실패: {e}")
            continue
        if not df.empty:
            series[t] = df["종가"].replace(0, pd.NA)
        if pause:
            time.sleep(pause)
        if i % 50 == 0:
            print(f"  시세 {i}/{len(tickers)}")
    return pd.DataFrame(series).sort_index().astype(float)


def fundamentals(date: str) -> pd.DataFrame:
    """PER, PBR, DIV 등. index=ticker."""
    return _krx().get_market_fundamental(date, market="ALL")


def benchmark(start: str, end: str, index_code: str = KOSPI200) -> pd.Series:
    df = _krx().get_index_ohlcv(start, end, index_code)
    return df["종가"].astype(float)


def lookback_start(end: str, days: int) -> str:
    return ymd(pd.Timestamp(end) - timedelta(days=days))
