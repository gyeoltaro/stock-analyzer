"""네이버 금융 데이터 수집 (로그인 불필요).

공식 API가 아니라 네이버 금융 웹페이지/차트 데이터를 읽는 방식이라
네이버가 페이지를 바꾸면 동작하지 않을 수 있다.
"""
import ast
import re
import time
from concurrent.futures import ThreadPoolExecutor
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

WORKERS = 8  # 동시 요청 수 (너무 높이면 네이버가 차단할 수 있음)


def _get_json_or_text(url: str, params=None, as_json=True, retries: int = 3):
    """재시도 포함 GET. 스레드마다 별도 요청을 보낸다."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json() if as_json else r.text
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _fetch_many(func, tickers, label: str) -> dict:
    """func(ticker)를 병렬 실행해 {ticker: 결과} 반환. 실패한 종목은 건너뛴다."""
    tickers = list(tickers)
    out, failed = {}, 0

    def run(t):
        try:
            return t, func(t)
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (t, res) in enumerate(ex.map(run, tickers), 1):
            if res is None:
                failed += 1
            else:
                out[t] = res
            if i % 250 == 0 or i == len(tickers):
                print(f"  {label} {i}/{len(tickers)} (실패 {failed})")
    return out


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
    return parse_sise_json(_get_json_or_text(SISE_JSON, params, as_json=False))


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


def market_value_ranking(page_size: int = 100, max_pages: int = 40) -> pd.DataFrame:
    """코스피·코스닥 전 종목 (시가총액 순)."""
    frames = []
    for market in ("KOSPI", "KOSDAQ"):
        for page in range(1, max_pages + 1):
            payload = _get_json_or_text(MARKET_VALUE.format(market=market),
                                        {"page": page, "pageSize": page_size})
            if not payload.get("stocks"):
                break
            frames.append(parse_market_value(payload))
            time.sleep(0.1)
    df = pd.concat(frames) if frames else pd.DataFrame()
    if df.empty:
        raise RuntimeError("네이버 종목 목록을 가져오지 못했습니다.")
    print(f"  종목 목록 {len(df)}개")
    return df[~df.index.duplicated()]


def parse_integration(payload: dict) -> dict:
    """종목 상세 응답에서 PER, PBR, 배당수익률 추출."""
    info = {i.get("code"): i.get("value") for i in payload.get("totalInfos") or []}
    return {"PER": _to_num(info.get("per")), "PBR": _to_num(info.get("pbr")),
            "DIV": _to_num(info.get("dividendYieldRatio"))}


def universe(date: str, top_n: int | None = None, min_trading_value: float = 1e8) -> pd.DataFrame:
    """코스피·코스닥 보통주 (네이버는 과거 시점 조회 불가, date는 무시).

    top_n 이 없거나 0이면 전 종목. 당일 거래대금이 min_trading_value 미만인 종목은 제외.
    """
    df = market_value_ranking()
    df = df[df["trading_value"] >= min_trading_value]
    df = df[df.index.str.endswith("0")]  # 우선주 제외
    df = df[~df["name"].str.contains("스팩|리츠", na=False)]
    df = df.sort_values("market_cap", ascending=False)
    if top_n:
        df = df.head(top_n)
    return df[["name", "market_cap", "trading_value"]]


def close_prices(tickers, start: str, end: str) -> pd.DataFrame:
    def one(t):
        df = ohlcv(t, start, end)
        return df["종가"].astype(float).replace(0, float("nan")) if not df.empty else None

    series = {t: v for t, v in _fetch_many(one, tickers, "시세").items() if v is not None}
    return pd.DataFrame(series).sort_index()


def ohlcv_frames(tickers, start: str, end: str) -> dict:
    """{ticker: 일봉 DataFrame}. 빈 결과는 제외."""
    got = _fetch_many(lambda t: ohlcv(t, start, end), tickers, "일봉")
    return {t: df for t, df in got.items() if df is not None and not df.empty}


def fundamentals(date: str, tickers=None) -> pd.DataFrame:
    """현재 PER, PBR (네이버는 과거 시점 조회 불가, date는 무시)."""
    def one(t):
        return parse_integration(_get_json_or_text(INTEGRATION.format(ticker=t)))

    rows = _fetch_many(one, tickers or [], "재무지표")
    return pd.DataFrame.from_dict(rows, orient="index")


def benchmark(start: str, end: str, index_code: str = KOSPI200) -> pd.Series:
    return ohlcv(index_code, start, end)["종가"].astype(float)
