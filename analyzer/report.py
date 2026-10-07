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
