"""배당주 평가: 배당수익률·배당성향·주가 안정성·배당 이력으로 순위를 매긴다."""
import numpy as np
import pandas as pd

DISCLAIMER = ("> ⚠️ 공개 데이터에 정해진 기준을 적용한 **참고 자료**이며 투자 권유가 아닙니다. "
              "배당은 회사 사정에 따라 줄거나 없어질 수 있습니다.")

WEIGHTS = {"yield": 0.40, "payout": 0.20, "stability": 0.25, "history": 0.15}


def payout_ratio(div_yield_pct: float, per: float) -> float:
    """배당성향(%) = 주당배당금/주당순이익 = 배당수익률 x PER."""
    if np.isnan(div_yield_pct) or np.isnan(per) or per <= 0:
        return np.nan
    return div_yield_pct * per


def payout_score(p: float) -> float:
    """30~60%가 가장 좋고, 너무 낮거나(배당 의지 약함) 높으면(지속 어려움) 감점. 0~1."""
    if np.isnan(p):
        return 0.0
    if 30 <= p <= 60:
        return 1.0
    if p < 30:
        return max(0.3, p / 30)
    return max(0.0, 1 - (p - 60) / 60)  # 120% 이상이면 0


def history_score(dps: list) -> float:
    """연도별 주당배당금 목록(오래된 순)으로 꾸준함 점수 0~1. 정보 없으면 0.5."""
    vals = [d for d in dps if d is not None and not np.isnan(d)]
    if len(vals) < 2:
        return 0.5
    paid = sum(v > 0 for v in vals) / len(vals)
    cuts = sum(b < a * 0.9 for a, b in zip(vals, vals[1:]))
    growth = 1.0 if vals[-1] >= vals[0] else 0.5
    return max(0.0, 0.5 * paid + 0.3 * growth + 0.2 * (1 - cuts / max(len(vals) - 1, 1)))


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """df 열: name, DIV, PER, vol_1y, mdd_1y, dps_hist(list). 점수 내림차순 반환."""
    df = df.copy()
    df["payout"] = [payout_ratio(d, p) for d, p in zip(df["DIV"], df["PER"])]
    yield_s = df["DIV"].rank(pct=True)
    pay_s = df["payout"].map(payout_score)
    stab = pd.concat([(-df["vol_1y"]).rank(pct=True), df["mdd_1y"].rank(pct=True)], axis=1).mean(axis=1)
    hist = df["dps_hist"].map(history_score)
    df["score"] = 100 * (WEIGHTS["yield"] * yield_s + WEIGHTS["payout"] * pay_s
                         + WEIGHTS["stability"] * stab + WEIGHTS["history"] * hist)
    df["history_score"] = hist
    return df.sort_values("score", ascending=False)


def _fmt_hist(h):
    vals = [f"{v:,.0f}" for v in h if v is not None and not np.isnan(v)]
    return " → ".join(vals) if vals else "-"


def render(df: pd.DataFrame, n_universe: int, min_yield: float, top: int, weekly: int) -> str:
    lines = ["# 배당주 순위", "", DISCLAIMER, "",
             f"- 대상: 시가총액 상위 {n_universe}개 중 배당수익률 {min_yield:.1f}% 이상·흑자 기업 {len(df)}개",
             "- 점수: 배당수익률 40% · 배당성향 적정성 20% (30~60% 만점) · 주가 안정성 25% (1년 변동성·최대낙폭) · "
             "배당 이력 15%",
             "- 배당성향 = 배당수익률 × PER (이익 중 배당으로 주는 비율, 100% 넘으면 번 돈보다 많이 주는 것)", "",
             f"## 상위 {min(top, len(df))}개", "",
             "| 순위 | 종목 | 코드 | 점수 | 배당수익률 | 배당성향 | PER | 1년 변동성 | 1년 최대낙폭 | 1년 수익률 | 주당배당금 이력 |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, (t, r) in enumerate(df.head(top).iterrows(), 1):
        lines.append(
            f"| {i} | {r['name']} | {t} | {r['score']:.1f} | {r['DIV']:.2f}% | "
            f"{'-' if np.isnan(r['payout']) else f'{r.payout:.0f}%'} | {r['PER']:.1f} | {r['vol_1y'] * 100:.0f}% | "
            f"{r['mdd_1y'] * 100:.0f}% | {r['ret_1y'] * 100:+.0f}% | {_fmt_hist(r['dps_hist'])} |")
    if weekly and len(df):
        top3 = df.head(3)
        annual = weekly * 52
        y = top3["DIV"].mean() / 100
        lines += ["", f"## 매주 {weekly:,}원을 상위 3개에 나눠 넣으면 (참고 계산)", "",
                  f"- 1년 투자금 {annual:,.0f}원 · 상위 3개 평균 배당수익률 {y * 100:.2f}%",
                  f"- 1년 뒤 보유액 기준 연 배당금 약 {annual * y:,.0f}원 (세전, 주가·배당 변동 없다고 가정)"]
    return "\n".join(lines) + "\n"
