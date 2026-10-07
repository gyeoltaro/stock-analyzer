"""점수 결과를 마크다운 리포트로 만든다."""
import pandas as pd

DISCLAIMER = ("> ⚠️ 이 리포트는 과거 데이터에 규칙을 적용한 **참고 자료**이며 투자 권유가 아닙니다. "
              "미래 수익을 보장하지 않으며, 투자 판단과 손실 책임은 본인에게 있습니다.")


def _pct(x):
    return "-" if pd.isna(x) else f"{x * 100:+.1f}%"


def _num(x, fmt="{:.1f}"):
    return "-" if pd.isna(x) else fmt.format(x)


def daily_report(date: str, ranked: pd.DataFrame, names: pd.Series, universe_size: int, top_n: int) -> str:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    lines = [f"# 오늘의 종목 점수 리포트 ({d})", "", DISCLAIMER, "",
             f"- 분석 대상: 코스피·코스닥 {universe_size}개 종목 중 필터 통과 {len(ranked)}개",
             "- 필터: 종가 > 120일 이동평균, RSI(14) < 75, 1년 이상 거래 이력",
             "- 점수: 12-1개월 모멘텀 35% · 3개월 모멘텀 15% · 가치(PER/PBR) 25% · 저변동성 25%", "",
             f"## 상위 {min(top_n, len(ranked))}개", "",
             "| 순위 | 종목 | 코드 | 점수 | 종가 | 12-1M 모멘텀 | 3M 모멘텀 | RSI | 변동성 | PER | PBR |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, (t, r) in enumerate(ranked.head(top_n).iterrows(), 1):
        lines.append(
            f"| {rank} | {names.get(t, t)} | {t} | {r['score']:.1f} | {_num(r['close'], '{:,.0f}')} | "
            f"{_pct(r['mom_12_1'])} | {_pct(r['mom_3'])} | {_num(r['rsi'])} | {_num(r['vol_60'] * 100, '{:.1f}%')} | "
            f"{_num(r.get('PER'))} | {_num(r.get('PBR'), '{:.2f}')} |")
    return "\n".join(lines) + "\n"


def backtest_report(summary: dict, holdings: dict, names: pd.Series, period: str) -> str:
    lines = [f"# 백테스트 결과 ({period})", "", DISCLAIMER, "",
             "> 현재 상장된 종목으로만 과거를 테스트하므로(상장폐지 종목 누락) **생존편향**이 있어 실제보다 좋게 나옵니다.", "",
             "| 지표 | " + " | ".join(summary.keys()) + " |",
             "|---|" + "---:|" * len(summary)]
    keys = list(next(iter(summary.values())).keys())
    for k in keys:
        cells = []
        for v in summary.values():
            x = v.get(k)
            cells.append("-" if x is None else (f"{x:.2f}" if k == "샤프" else f"{x * 100:.1f}%"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if holdings:
        last_day, picks = list(holdings.items())[-1]
        lines += ["", f"## 마지막 리밸런싱 ({pd.Timestamp(last_day):%Y-%m-%d}) 보유 종목", "",
                  ", ".join(f"{names.get(t, t)}({t})" for t in picks)]
    return "\n".join(lines) + "\n"


SWING_RULES = {
    "pullback": ("상승 추세(종가 > 120일선, 20일선 > 60일선)에서 RSI(2) < 10 으로 단기 급락",
                 "종가가 5일선 위로 올라오면 다음날 시가 매도 · 최대 10거래일"),
    "breakout": ("20일 최고가 돌파 + 거래량 20일 평균의 2배 이상 + 양봉 (상한가 제외)",
                 "종가가 10일선 아래로 내려가면 다음날 시가 매도 · 최대 15거래일"),
    "ma15": ("15일선 상승 중 + 15일선 > 60일선 + 최근 10일 내 15일선 +5% 이상 상승 이력, "
             "저가가 15일선 +1% 이내로 내려왔다가 종가는 15일선 위",
             "+10% 익절 · 종가가 15일선 아래면 다음날 시가 매도 · 손절 = 신호일 15일선 -3% · 최대 10거래일"),
}


def swing_report(date: str, signals: dict, names: pd.Series, universe_size: int) -> str:
    from .swing import SETUP_NAMES
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    lines = [f"# 단기 스윙 신호 ({d})", "", DISCLAIMER, "",
             f"- 분석 대상: 코스피·코스닥 {universe_size}개 종목 (20일 평균 거래대금 10억 이상, 1,000원 이상)",
             "- 매수: 오늘 신호가 뜬 종목을 **다음 거래일 시가**에 매수 (아래는 우선순위 순)",
             "- 손절가는 오늘 종가 기준 참고값이며, 실제로는 매수가 기준으로 다시 계산하세요.", ""]
    for setup, df in signals.items():
        cond, exit_rule = SWING_RULES[setup]
        lines += [f"## {SETUP_NAMES[setup]} ({len(df)}개 표시)", "",
                  f"- 조건: {cond}", f"- 매도: {exit_rule} · 손절가 이탈 시 즉시 손절", ""]
        if df.empty:
            lines += ["오늘은 신호 없음", ""]
            continue
        lines += ["| 순위 | 종목 | 코드 | 종가 | 손절가(참고) | 손절폭 | RSI(2) | 거래량 배수 | 20일 평균 거래대금 |",
                  "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for i, r in enumerate(df.itertuples(), 1):
            lines.append(
                f"| {i} | {names.get(r.ticker, r.ticker)} | {r.ticker} | {r.close:,.0f} | {r.stop:,.0f} | "
                f"{(r.stop / r.close - 1) * 100:.1f}% | {_num(r.rsi2)} | {_num(r.vol_ratio)}x | "
                f"{r.trade_value / 1e8:,.0f}억 |")
        lines.append("")
    return "\n".join(lines) + "\n"


def swing_backtest_report(results: dict, period: str, cfg) -> str:
    from .swing import SETUP_NAMES
    lines = [f"# 단기 스윙 백테스트 ({period})", "", DISCLAIMER, "",
             "> 현재 상장된 종목으로만 테스트하므로(상장폐지 종목 누락) 실제보다 좋게 나올 수 있습니다.",
             f"> 동시 보유 최대 {cfg.max_positions}종목(균등 분할), 왕복 비용 {cfg.cost_roundtrip * 100:.1f}% 반영, "
             "신호 다음날 시가 매수.", ""]
    cols = {SETUP_NAMES[k]: v.summary()["전략"] for k, v in results.items()}
    first = next(iter(results.values())).summary()
    if "벤치마크" in first:
        cols["코스피200"] = first["벤치마크"]
    keys = []
    for v in cols.values():
        keys += [k for k in v if k not in keys]
    lines += ["| 지표 | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in keys:
        cells = []
        for v in cols.values():
            x = v.get(k)
            if x is None:
                cells.append("-")
            elif k == "샤프":
                cells.append(f"{x:.2f}")
            elif k in ("거래 수",):
                cells.append(f"{x:,.0f}")
            elif k == "평균 보유일":
                cells.append(f"{x:.1f}일")
            else:
                cells.append(f"{x * 100:.1f}%")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    for k, v in results.items():
        t = v.trades
        if t.empty or "reason" not in t:
            continue
        g = t.groupby("reason")["return"].agg(["count", "mean", lambda r: (r > 0).mean()])
        lines += ["", f"### {SETUP_NAMES[k]} 매도 이유별", "", "| 이유 | 거래 수 | 평균 수익률 | 승률 |", "|---|---:|---:|---:|"]
        for reason, r in g.iterrows():
            lines.append(f"| {reason} | {r['count']:,.0f} | {r['mean'] * 100:.2f}% | {r.iloc[2] * 100:.1f}% |")
    return "\n".join(lines) + "\n"
