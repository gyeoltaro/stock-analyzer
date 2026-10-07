"""네이버 금융 데이터 수집 (로그인 불필요).

공식 API가 아니라 네이버 금융 웹페이지/차트 데이터를 읽는 방식이라
네이버가 페이지를 바꾸면 동작하지 않을 수 있다.
"""
import ast
import re
import time
from datetime import datetime

import pandas as pd
import requests

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Referer": "https://finance.naver.com/",
}
SISE_JSON = "https://api.finance.naver.com/siseJson.naver"
KOSPI200 = "KPI200"

_session = requests.Session()
_session.headers.update(HEADERS)


def ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def parse_sise_json(text: str) -> pd.DataFrame:
    """siseJson 응답(파이썬 리스트 형태 텍스트)을 DataFrame으로."""
    rows = ast.literal_eval(text.strip())
    if len(rows) < 2:
        return pd.DataFrame()
    header = [str(h).strip() for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header)
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), format="%Y%m%d")
    return df.set_index("날짜").sort_index()


def ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    params = {"symbol": symbol, "requestType": 1, "startTime": ymd(start),
              "endTime": ymd(end), "timeframe": "day"}
    r = _session.get(SISE_JSON, params=params, timeout=15)
    r.raise_for_status()
    return parse_sise_json(r.text)


def latest_business_day(date=None) -> str:
    end = pd.Timestamp(date or datetime.now())
    df = ohlcv("005930", end - pd.Timedelta(days=14), end)
    return ymd(df.index[-1])


MARKET_VALUE = "https://m.stock.naver.com/api/stocks/marketValue/{market}"
INTEGRATION = "https://m.stock.naver.com/api/stock/{ticker}/integration"

_NUM = re.compile(r"[^0-9.\-]")


def _to_num(s) -> float:
    if s is None:
        return float("nan")
    s = _NUM.sub("", str(s))
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_market_value(payload: dict) -> pd.DataFrame:
    """m.stock.naver.com 시가총액 순위 응답을 DataFrame으로."""
    rows = []
    for st in payload.get("stocks", []):
        if st.get("stockEndType") != "stock":
            continue
        rows.append({
            "ticker": st["itemCode"],
            "name": st.get("stockName", ""),
            "close": _to_num(st.get("closePriceRaw")),
            "market_cap": _to_num(st.get("marketValueRaw")),
            "trading_value": _to_num(st.get("accumulatedTradingValueRaw")),
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def market_value_ranking(pages_kospi: int = 3, pages_kosdaq: int = 1, page_size: int = 100) -> pd.DataFrame:
    frames = []
    for market, pages in (("KOSPI", pages_kospi), ("KOSDAQ", pages_kosdaq)):
        for page in range(1, pages + 1):
            r = _session.get(MARKET_VALUE.format(market=market),
                             params={"page": page, "pageSize": page_size}, timeout=15)
            r.raise_for_status()
            frames.append(parse_market_value(r.json()))
            time.sleep(0.2)
    df = pd.concat(frames)
    if df.empty:
        raise RuntimeError("네이버 시가총액 순위를 가져오지 못했습니다.")
    return df[~df.index.duplicated()]


def parse_integration(payload: dict) -> dict:
    """종목 상세 응답에서 PER, PBR, 배당수익률 추출."""
    info = {i.get("code"): i.get("value") for i in payload.get("totalInfos") or []}
    return {"PER": _to_num(info.get("per")), "PBR": _to_num(info.get("pbr")),
            "DIV": _to_num(info.get("dividendYieldRatio"))}


def universe(date: str, top_n: int = 200, min_trading_value: float = 1e9) -> pd.DataFrame:
    """현재 시가총액 상위 종목 (네이버는 과거 시점 조회 불가, date는 무시)."""
    df = market_value_ranking()
    df = df[df["trading_value"] >= min_trading_value]
    df = df[df.index.str.endswith("0")]  # 우선주 제외
    df = df[~df["name"].str.contains("스팩|리츠", na=False)]
    return df.sort_values("market_cap", ascending=False).head(top_n)[["name", "market_cap", "trading_value"]]


def close_prices(tickers, start: str, end: str, pause: float = 0.1) -> pd.DataFrame:
    series = {}
    for i, t in enumerate(tickers, 1):
        try:
            df = ohlcv(t, start, end)
        except Exception as e:
            print(f"[warn] {t} 시세 조회 실패: {e}")
            continue
        if not df.empty:
            series[t] = df["종가"].astype(float).replace(0, float("nan"))
        if pause:
            time.sleep(pause)
        if i % 50 == 0:
            print(f"  시세 {i}/{len(tickers)}")
    return pd.DataFrame(series).sort_index()


def fundamentals(date: str, tickers=None, pause: float = 0.1) -> pd.DataFrame:
    """현재 PER, PBR (네이버는 과거 시점 조회 불가, date는 무시)."""
    rows = {}
    for t in tickers or []:
        try:
            r = _session.get(INTEGRATION.format(ticker=t), timeout=15)
            r.raise_for_status()
            rows[t] = parse_integration(r.json())
        except Exception as e:
            print(f"[warn] {t} 재무지표 조회 실패: {e}")
        if pause:
            time.sleep(pause)
    return pd.DataFrame.from_dict(rows, orient="index")


def benchmark(start: str, end: str, index_code: str = KOSPI200) -> pd.Series:
    return ohlcv(index_code, start, end)["종가"].astype(float)
