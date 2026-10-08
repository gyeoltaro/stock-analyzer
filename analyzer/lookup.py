"""개별 종목 평가: 이름으로 찾아 실적·밸류에이션·주가 흐름·위험도를 점검한다."""
import numpy as np
import pandas as pd

from . import indicators as ind

DISCLAIMER = ("> ⚠️ 공개 데이터에 정해진 기준을 적용한 **참고 자료**이며 투자 권유가 아닙니다. "
              "사업 내용·공시·뉴스는 직접 확인하세요.")


def find(listing: pd.DataFrame, query: str) -> pd.DataFrame:
    """종목명에 query 가 들어간 종목 (시가총액 큰 순). 정확히 일치하면 그것만."""
    exact = listing[listing["name"] == query]
    if len(exact):
        return exact
    hits = listing[listing["name"].str.contains(query, regex=False, na=False)]
    return hits.sort_values("market_cap", ascending=False)


def price_stats(df: pd.DataFrame) -> dict:
    c = df["종가"].astype(float).replace(0, np.nan).dropna()
    v = df["거래량"].astype(float)
    frame = c.to_frame("x")
    last = c.iloc[-1]

    def ret(n):
        return c.iloc[-1] / c.iloc[-n - 1] - 1 if len(c) > n else np.nan

    year = c.iloc[-252:]
    return {
        "close": last,
        "ret_1m": ret(21), "ret_3m": ret(63), "ret_1y": ret(252),
        "vol_1y": float(c.pct_change().iloc[-252:].std() * np.sqrt(252)),
        "mdd_1y": float((year / year.cummax() - 1).min()),
        "from_high_1y": float(last / year.max() - 1),
        "rsi14": float(ind.rsi(frame).iloc[-1, 0]),
        "above_ma120": bool(last > ind.sma(frame, 120).iloc[-1, 0]) if len(c) >= 120 else None,
        "avg_value_20d": float((c * v.reindex(c.index)).iloc[-20:].mean()),
    }


def _num(info: dict, code: str) -> float:
    raw = info.get(code, (None, None, None))[1]
    if raw is None:
        return np.nan
    s = "".join(ch for ch in str(raw) if ch.isdigit() or ch in ".-")
    try:
        return float(s)
    except ValueError:
        return np.nan


def checklist(info: dict, st: dict, market_cap: float) -> list:
    """(항목, 결과, 설명) 목록. 결과는 '좋음' / '주의' / '나쁨' / '정보없음'."""
    per, pbr = _num(info, "per"), _num(info, "pbr")
    cns_per = _num(info, "cnsPer")
    div = _num(info, "dividendYieldRatio")
    out = []

    if np.isnan(per):
        out.append(("수익성", "주의", "PER 정보 없음 — 적자이거나 실적 정보가 부족할 수 있음"))
    elif per <= 0:
        out.append(("수익성", "나쁨", f"PER {per:.1f} — 최근 실적 적자"))
    else:
        out.append(("수익성", "좋음", f"PER {per:.1f} — 흑자"))

    if not np.isnan(per) and per > 0:
        if per < 10:
            out.append(("가격 수준(PER)", "좋음", f"PER {per:.1f}배 — 이익 대비 싼 편"))
        elif per < 25:
            out.append(("가격 수준(PER)", "보통", f"PER {per:.1f}배 — 보통"))
        else:
            out.append(("가격 수준(PER)", "주의", f"PER {per:.1f}배 — 이익 대비 비싼 편, 성장 기대가 반영됨"))
    if not np.isnan(cns_per) and not np.isnan(per) and per > 0:
        trend = "이익 증가 전망" if cns_per < per else "이익 감소 전망"
        out.append(("이익 전망", "좋음" if cns_per < per else "주의",
                    f"추정 PER {cns_per:.1f}배 (현재 {per:.1f}배) — 애널리스트 {trend}"))
    if not np.isnan(pbr):
        out.append(("자산 대비 가격(PBR)", "좋음" if pbr < 1.5 else ("보통" if pbr < 3 else "주의"),
                    f"PBR {pbr:.2f}배"))
    if not np.isnan(div):
        out.append(("배당", "좋음" if div >= 2 else "보통", f"배당수익률 {div:.2f}%"))

    cap_eok = market_cap / 1e8
    out.append(("회사 규모", "좋음" if cap_eok >= 10000 else ("보통" if cap_eok >= 3000 else "주의"),
                f"시가총액 {cap_eok:,.0f}억 원" + (" — 소형주라 변동이 클 수 있음" if cap_eok < 3000 else "")))

    av = st["avg_value_20d"] / 1e8
    out.append(("거래 활발도", "좋음" if av >= 50 else ("보통" if av >= 10 else "주의"),
                f"20일 평균 거래대금 {av:,.0f}억 원" + (" — 거래가 적어 원하는 가격에 사고팔기 어려울 수 있음" if av < 10 else "")))

    vol = st["vol_1y"]
    out.append(("주가 변동성", "좋음" if vol < 0.35 else ("보통" if vol < 0.6 else "주의"),
                f"연 변동성 {vol * 100:.0f}% (코스피200 대형주는 보통 25~40%)"))
    out.append(("최근 1년 최대 하락", "좋음" if st["mdd_1y"] > -0.25 else ("보통" if st["mdd_1y"] > -0.45 else "주의"),
                f"고점 대비 최대 {st['mdd_1y'] * 100:.0f}%"))
    if st["above_ma120"] is not None:
        out.append(("추세", "좋음" if st["above_ma120"] else "주의",
                    "120일 이동평균 위 (상승 추세)" if st["above_ma120"] else "120일 이동평균 아래 (하락 추세)"))
    rsi = st["rsi14"]
    if not np.isnan(rsi):
        out.append(("단기 과열", "주의" if rsi >= 70 else "보통",
                    f"RSI(14) {rsi:.0f}" + (" — 단기 과열, 추격 매수 주의" if rsi >= 70 else "")))
    return out


def _pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.1f}%"


def render(query: str, rows: list) -> str:
    lines = [f"# 종목 평가: \"{query}\"", "", DISCLAIMER, ""]
    if not rows:
        lines.append("이름에 해당 단어가 들어간 상장 종목을 찾지 못했습니다.")
        return "\n".join(lines) + "\n"
    if len(rows) > 1:
        lines += [f"이름에 \"{query}\"가 들어간 종목이 {len(rows)}개 있어 모두 정리했습니다.", ""]
    for r in rows:
        st, info, checks = r["stats"], r["info"], r["checks"]
        score = sum({"좋음": 1, "보통": 0, "주의": -1, "나쁨": -2}.get(c[1], 0) for c in checks)
        lines += [f"## {r['name']} ({r['ticker']}, {r['market']})", "",
                  f"- 종가 {st['close']:,.0f}원 · 1개월 {_pct(st['ret_1m'])} · 3개월 {_pct(st['ret_3m'])} · "
                  f"1년 {_pct(st['ret_1y'])} · 52주 고점 대비 {_pct(st['from_high_1y'])}",
                  f"- 체크리스트 합계: **{score:+d}점** (좋음 +1, 보통 0, 주의 -1, 나쁨 -2)", "",
                  "| 항목 | 판정 | 내용 |", "|---|---|---|"]
        lines += [f"| {k} | {v} | {d} |" for k, v, d in checks]
        raw = [f"{name} {val}" + (f" ({desc})" if desc else "")
               for code, (name, val, desc) in info.items()
               if code in ("per", "eps", "cnsPer", "cnsEps", "pbr", "bps", "dividendYieldRatio",
                           "foreignRate", "highPriceOf52Weeks", "lowPriceOf52Weeks", "marketValue")]
        if raw:
            lines += ["", "원자료: " + " · ".join(raw)]
        if r.get("deals"):
            lines += ["", "### 최근 투자자별 순매수 (주)", "",
                      "| 날짜 | 종가 | 거래량 | 개인 | 외국인 | 기관 |", "|---|---:|---:|---:|---:|---:|"]
            for d in r["deals"][:7]:
                lines.append(f"| {d.get('bizdate', '')} | {d.get('closePrice', '')} | {d.get('accumulatedTradingVolume', '')} | "
                             f"{d.get('individualPureBuyQuant', '')} | {d.get('foreignerPureBuyQuant', '')} | "
                             f"{d.get('organPureBuyQuant', '')} |")
        if r.get("news"):
            lines += ["", "### 최근 뉴스 (네이버 금융)", ""]
            lines += [f"- {n['datetime'][:12]} {n['office']} — {n['title']}" for n in r["news"]]
        elif r.get("news") is not None:
            lines += ["", "### 최근 뉴스", "", "가져온 뉴스 없음"]
        lines.append("")
    return "\n".join(lines) + "\n"
