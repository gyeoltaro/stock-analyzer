"""적립식(매주 일정 금액) 매수 시뮬레이션. 배당은 세후로 재투자한다.

가정
- 매주 첫 거래일 종가로 비중대로 매수, 소수점 매수 허용, 매수 비용 0.1%.
- 배당: 회계연도 Y의 주당배당금을 Y년 마지막 거래일 보유 주식 수만큼, Y+1년 4월 첫 거래일에 받아
  세후(15.4% 공제) 같은 종목에 재투자. (분기배당도 연간 합계로 근사)
- 배당 자료가 없는 연도는 0으로 본다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TAX = 0.154


@dataclass
class DcaResult:
    name: str
    invested: float
    value: float
    dividends: float
    worst_pct: float          # 기간 중 (평가액/누적 투자금 - 1) 최저
    series: pd.DataFrame      # date, invested, value

    @property
    def gain_pct(self) -> float:
        return self.value / self.invested - 1 if self.invested else np.nan


def weekly_buy_days(index: pd.DatetimeIndex) -> list:
    s = pd.Series(index, index=index)
    return list(s.groupby(index.to_period("W")).min())


def simulate(name: str, closes: dict, weights: dict, amount: float, start, end=None,
             dps: dict | None = None, cost: float = 0.001) -> DcaResult:
    """closes: {ticker: 종가 Series}, weights: {ticker: 비중}, dps: {ticker: {연도: 주당배당금}}."""
    px = pd.DataFrame(closes).sort_index().ffill()
    px = px.loc[pd.Timestamp(start): (pd.Timestamp(end) if end else None)].dropna(how="all")
    tickers = list(weights)
    wsum = sum(weights.values())
    w = {t: weights[t] / wsum for t in tickers}
    dps = dps or {}
    shares = {t: 0.0 for t in tickers}
    buy_days = set(weekly_buy_days(px.index))
    years = sorted({d.year for d in px.index})
    record_days = {y: px.index[px.index.year == y].max() for y in years}
    pay_days = {}
    for y in years:
        nxt = px.index[(px.index.year == y + 1) & (px.index.month >= 4)]
        if len(nxt):
            pay_days[nxt.min()] = y
    held_at_record = {}
    invested = dividends = 0.0
    rows = []
    for d in px.index:
        if d in pay_days:
            y = pay_days[d]
            for t in tickers:
                per_share = dps.get(t, {}).get(y, 0.0) or 0.0
                cash = held_at_record.get((y, t), 0.0) * per_share * (1 - TAX)
                if cash > 0 and not np.isnan(px.at[d, t]):
                    dividends += cash
                    shares[t] += cash * (1 - cost) / px.at[d, t]
        if d in buy_days:
            for t in tickers:
                p = px.at[d, t]
                if not np.isnan(p):
                    shares[t] += amount * w[t] * (1 - cost) / p
                    invested += amount * w[t]
        if d in record_days.values():
            for t in tickers:
                held_at_record[(d.year, t)] = shares[t]
        value = sum(shares[t] * px.at[d, t] for t in tickers if not np.isnan(px.at[d, t]))
        rows.append((d, invested, value))
    series = pd.DataFrame(rows, columns=["date", "invested", "value"]).set_index("date")
    ratio = (series["value"] / series["invested"].replace(0, np.nan) - 1).dropna()
    return DcaResult(name, invested, float(series["value"].iloc[-1]), dividends,
                     float(ratio.min()) if len(ratio) else np.nan, series)


def render(results: list, amount: float, start: str, end: str, notes: list) -> str:
    lines = [f"# 매주 {amount:,.0f}원 적립식 시뮬레이션 ({pd.Timestamp(start):%Y-%m-%d} ~ {pd.Timestamp(end):%Y-%m-%d})", "",
             "> ⚠️ 과거 데이터로 계산한 참고 자료입니다. 미래 수익을 보장하지 않습니다.", "",
             "- 매주 첫 거래일 종가로 매수 (소수점 매수, 매수 비용 0.1%)",
             "- 배당은 세후(15.4% 공제) 이듬해 4월에 같은 종목으로 재투자 · 매도 시 세금·수수료는 미반영",
             "- 코스피200은 지수 자체(배당 미포함)라 실제 ETF보다 약간 낮게 나옵니다", "",
             "| 포트폴리오 | 넣은 돈 | 지금 평가액 | 수익 | 수익률 | 받은 배당(세후) | 중간 최악 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda r: -r.gain_pct):
        lines.append(f"| {r.name} | {r.invested:,.0f}원 | {r.value:,.0f}원 | {r.value - r.invested:+,.0f}원 | "
                     f"{r.gain_pct * 100:+.1f}% | {r.dividends:,.0f}원 | {r.worst_pct * 100:+.1f}% |")
    lines += ["", "- **중간 최악**: 기간 중 평가액이 그때까지 넣은 돈보다 가장 많이 밑돌았던 비율 (버텨야 했던 손실)"]
    if notes:
        lines += ["", "### 참고", ""] + [f"- {n}" for n in notes]
    # 연말별 평가액
    lines += ["", "### 연말 평가액", "", "| 시점 | " + " | ".join(r.name for r in results) + " |",
              "|---|" + "---:|" * len(results)]
    ends = results[0].series.groupby(results[0].series.index.year).tail(1).index
    for d in ends:
        cells = []
        for r in results:
            s = r.series.loc[:d]
            cells.append(f"{s['value'].iloc[-1]:,.0f} / {s['invested'].iloc[-1]:,.0f}" if len(s) else "-")
        lines.append(f"| {d:%Y-%m-%d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("(각 칸: 평가액 / 넣은 돈)")
    return "\n".join(lines) + "\n"
