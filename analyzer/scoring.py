"""팩터 점수 계산과 종목 순위.

점수 = 모멘텀(12-1개월, 3개월) + 가치(PBR, 이익수익률) + 저변동성
의 백분위 가중합이다. 학계에서 오래 검증된 팩터들을 단순하게 결합한 것으로,
미래 수익을 보장하지 않는다.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

MIN_HISTORY = 260  # 12개월 모멘텀 계산에 필요한 최소 거래일


@dataclass
class ScoreConfig:
    weights: dict = field(default_factory=lambda: {
        "mom_12_1": 0.35,
        "mom_3": 0.15,
        "value": 0.25,
        "low_vol": 0.25,
    })
    trend_window: int = 120   # 종가가 이 이동평균 위에 있어야 후보
    rsi_max: float = 75.0     # 과열 종목 제외
    top_n: int = 20


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def compute_factors(close: pd.DataFrame, fundamentals: pd.DataFrame | None, cfg: ScoreConfig) -> pd.DataFrame:
    """close의 마지막 날짜 기준 종목별 팩터 표를 만든다."""
    last = close.iloc[-1]
    history = close.notna().sum()
    f = pd.DataFrame(index=close.columns)
    f["close"] = last
    f["mom_12_1"] = ind.momentum(close, 252, 21).iloc[-1]
    f["mom_3"] = ind.momentum(close, 63).iloc[-1]
    f["rsi"] = ind.rsi(close).iloc[-1]
    f["vol_60"] = ind.volatility(close).iloc[-1]
    f["above_trend"] = last > ind.sma(close, cfg.trend_window).iloc[-1]
    f["history"] = history

    if fundamentals is not None and not fundamentals.empty:
        fund = fundamentals.reindex(f.index)
        per = fund.get("PER")
        pbr = fund.get("PBR")
        f["PER"] = per
        f["PBR"] = pbr
        # 적자(PER<=0)나 자본잠식(PBR<=0)은 가치 점수 없음
        ey = (1 / per).where(per > 0) if per is not None else np.nan
        bm = (1 / pbr).where(pbr > 0) if pbr is not None else np.nan
        f["value"] = pd.concat([_pct_rank(pd.Series(ey, index=f.index)),
                                _pct_rank(pd.Series(bm, index=f.index))], axis=1).mean(axis=1)
    else:
        f["value"] = np.nan
    return f


def score(close: pd.DataFrame, fundamentals: pd.DataFrame | None = None,
          cfg: ScoreConfig | None = None) -> pd.DataFrame:
    """필터를 통과한 종목을 점수 내림차순으로 반환한다."""
    cfg = cfg or ScoreConfig()
    f = compute_factors(close, fundamentals, cfg)

    eligible = (
        (f["history"] >= MIN_HISTORY)
        & f["above_trend"].fillna(False).astype(bool)
        & (f["rsi"] < cfg.rsi_max)
        & f["mom_12_1"].notna()
    )
    f = f[eligible].copy()
    if f.empty:
        return f

    parts = {
        "mom_12_1": _pct_rank(f["mom_12_1"]),
        "mom_3": _pct_rank(f["mom_3"]),
        "value": f["value"],
        "low_vol": _pct_rank(-f["vol_60"]),
    }
    total = pd.Series(0.0, index=f.index)
    weight_sum = pd.Series(0.0, index=f.index)
    for name, w in cfg.weights.items():
        p = parts[name]
        total += p.fillna(0) * w
        weight_sum += p.notna() * w
    # 데이터가 없는 팩터는 빼고 나머지 가중치로 다시 정규화
    f["score"] = (total / weight_sum.replace(0, np.nan)) * 100
    return f.sort_values("score", ascending=False)
