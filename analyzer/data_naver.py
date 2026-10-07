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
MARKET_SUM = "https://finance.naver.com/sise/sise_market_sum.naver"
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


_NUM = re.compile(r"[^0-9.\-]")


def _to_num(s: str) -> float:
    s = _NUM.sub("", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html).strip()


def parse_market_sum(html: str) -> pd.DataFrame:
    """시가총액 페이지 표를 파싱한다.

    기본 열 순서: 현재가, 전일비, 등락률, 액면가, 시가총액(억), 상장주식수, 외국인비율, 거래량, PER, ROE
    """
    out = []
    for row in re.split(r"<tr[\s>]", html):
        m = re.search(r'code=(\d{6})"\s*class="tltle">([^<]+)</a>', row)
        if not m:
            continue
        nums = [_strip_tags(x) for x in re.findall(r'<td class="number[^"]*">(.*?)</td>', row, re.S)]
        if len(nums) < 10:
            continue
        price = _to_num(nums[0])
        volume = _to_num(nums[7])
        out.append({
            "ticker": m.group(1),
            "name": m.group(2).strip(),
            "close": price,
            "market_cap": _to_num(nums[4]) * 1e8,
            "trading_value": price * volume,
            "PER": _to_num(nums[8]),
            "ROE": _to_num(nums[9]),
        })
    return pd.DataFrame(out).set_index("ticker") if out else pd.DataFrame()


def market_sum(pages_kospi: int = 4, pages_kosdaq: int = 2) -> pd.DataFrame:
    frames = []
    for sosok, pages in ((0, pages_kospi), (1, pages_kosdaq)):
        for page in range(1, pages + 1):
            r = _session.get(MARKET_SUM, params={"sosok": sosok, "page": page}, timeout=15)
            r.raise_for_status()
            html = r.content.decode("euc-kr", errors="replace")
            parsed = parse_market_sum(html)
            if parsed.empty:
                i = html.find("code=")
                snippet = html[max(0, i - 600): i + 1500] if i >= 0 else html[:2000]
                raise RuntimeError(f"시가총액 표 파싱 실패 (sosok={sosok}, page={page}, "
                                   f"status={r.status_code}, len={len(html)}):\n{snippet}")
            frames.append(parsed)
            time.sleep(0.2)
    df = pd.concat(frames)
    return df[~df.index.duplicated()]


_market_cache: dict = {}


def _snapshot() -> pd.DataFrame:
    if "df" not in _market_cache:
        _market_cache["df"] = market_sum()
    return _market_cache["df"]


def universe(date: str, top_n: int = 200, min_trading_value: float = 1e9) -> pd.DataFrame:
    """현재 시가총액 상위 종목 (네이버는 과거 시점 조회 불가, date는 무시)."""
    df = _snapshot()
    df = df[df["trading_value"] >= min_trading_value]
    df = df[df.index.str.endswith("0")]  # 우선주 제외
    df = df[~df["name"].str.contains("스팩|리츠|ETN", na=False)]
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


def fundamentals(date: str) -> pd.DataFrame:
    """현재 PER, PBR. PBR = PER x ROE / 100 으로 계산 (P/B = P/E x E/B)."""
    df = _snapshot()[["PER", "ROE"]].copy()
    df["PBR"] = df["PER"] * df["ROE"] / 100
    return df


def benchmark(start: str, end: str, index_code: str = KOSPI200) -> pd.Series:
    return ohlcv(index_code, start, end)["종가"].astype(float)
