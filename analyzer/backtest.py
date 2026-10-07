"""월간 리밸런싱 백테스트.

매월 마지막 거래일에 점수 상위 N 종목을 동일 비중으로 사고, 다음 리밸런싱까지 보유한다.
주의: 현재 시가총액 상위 종목으로 유니버스를 만들면 생존편향이 생겨 성과가 부풀려진다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import ScoreConfig, score

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series          # 전략 누적 가치 (시작 1.0)
    benchmark: pd.Series | None
    holdings: dict             # 리밸런싱 날짜 -> 종목 리스트
    turnover: pd.Series

    def summary(self) -> dict:
        out = {"전략": metrics(self.equity)}
        if self.benchmark is not None:
            b = self.benchmark.reindex(self.equity.index).ffill().dropna()
            out["벤치마크"] = metrics(b / b.iloc[0])
        out["전략"]["평균 회전율"] = float(self.turnover.mean()) if len(self.turnover) else 0.0
        return out


def metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0]
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = rets.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return {"총수익률": float(total - 1), "연복리(CAGR)": float(cagr),
            "변동성": float(vol), "샤프": float(sharpe), "최대낙폭(MDD)": float(mdd)}


def month_end_dates(index: pd.DatetimeIndex) -> list:
    s = pd.Series(index, index=index)
    return list(s.groupby([index.year, index.month]).max())


def run(close: pd.DataFrame, fundamentals_by_date=None, cfg: ScoreConfig | None = None,
        benchmark: pd.Series | None = None, cost: float = 0.003,
        start=None) -> BacktestResult:
    """
    close: 날짜 x 종목 수정종가
    fundamentals_by_date: 날짜(Timestamp) -> 펀더멘털 DataFrame 을 돌려주는 callable (선택)
    cost: 매매금액 대비 편도 비용(수수료+세금+슬리피지). 회전율에 곱해 차감.
    """
    cfg = cfg or ScoreConfig()
    close = close.sort_index()
    rebal = [d for d in month_end_dates(close.index) if start is None or d >= pd.Timestamp(start)]
    rets = close.pct_change(fill_method=None)

    weights = pd.Series(dtype=float)
    holdings, turnover = {}, {}
    daily = pd.Series(0.0, index=close.index)
    rebal_set = set(rebal)
    started = False

    for i, day in enumerate(close.index):
        if started and len(weights):
            r = rets.loc[day, weights.index].fillna(0)  # 거래정지 등은 0% 처리
            day_ret = float((weights * r).sum())
            daily.loc[day] = day_ret
            # 가격 변화에 따라 비중 드리프트
            weights = weights * (1 + r) / (1 + day_ret) if day_ret != -1 else weights
        if day in rebal_set:
            hist = close.loc[:day]
            fund = fundamentals_by_date(day) if fundamentals_by_date else None
            ranked = score(hist, fund, cfg)
            picks = list(ranked.index[: cfg.top_n])
            new_w = pd.Series(1 / len(picks), index=picks) if picks else pd.Series(dtype=float)
            all_idx = weights.index.union(new_w.index)
            to = float((new_w.reindex(all_idx, fill_value=0) - weights.reindex(all_idx, fill_value=0)).abs().sum())
            daily.loc[day] -= to * cost
            weights = new_w
            holdings[day] = picks
            turnover[day] = to
            started = True

    first = rebal[0] if rebal else close.index[0]
    daily = daily.loc[first:]
    equity = (1 + daily).cumprod()
    bench = benchmark.loc[first:] if benchmark is not None else None
    return BacktestResult(equity, bench, holdings, pd.Series(turnover))
