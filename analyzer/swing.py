"""단기 스윙 전략: 매수 신호와 일 단위 백테스트.

두 가지 셋업:
- 눌림목(pullback): 상승 추세 종목이 단기 과매도(RSI(2) < 10)일 때 매수, 종가가 5일선 위로 오르면 매도.
- 돌파(breakout): 20일 고가를 거래량 급증과 함께 넘을 때 매수, 종가가 10일선 아래로 내려가면 매도.

공통: 신호는 당일 종가로 판단하고 다음 거래일 시가에 매수한다(미래 정보 사용 없음).
손절가에 닿으면 즉시 손절, 최대 보유일이 지나면 다음 시가에 정리한다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import metrics

SETUPS = ("pullback", "breakout")
SETUP_NAMES = {"pullback": "눌림목", "breakout": "돌파"}


@dataclass
class SwingConfig:
    min_avg_value: float = 1e9      # 20일 평균 거래대금 하한 (원)
    min_price: float = 1000         # 동전주 제외
    max_positions: int = 5          # 동시 보유 종목 수 (자금을 균등 분할)
    pullback_rsi2: float = 10.0
    pullback_stop: float = 0.07     # 매수가 대비 손절 폭
    pullback_max_hold: int = 10     # 거래일
    breakout_lookback: int = 20
    breakout_vol_mult: float = 2.0  # 20일 평균 대비 거래량 배수
    breakout_atr_stop: float = 2.0  # 손절 = 매수가 - ATR x 배수
    breakout_max_hold: int = 15
    cost_roundtrip: float = 0.005   # 왕복 비용: 수수료 + 거래세 + 슬리피지


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev = close.shift()
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.rolling(window, min_periods=window).mean()


def compute(px: dict, cfg: SwingConfig) -> dict:
    """신호와 청산에 필요한 지표를 한 번에 계산한다."""
    c, h, l, v = px["close"], px["high"], px["low"], px["volume"]
    trade_value = (c * v).rolling(20, min_periods=20).mean()
    liquid = (trade_value >= cfg.min_avg_value) & (c >= cfg.min_price)
    sma5, sma10, sma20 = ind.sma(c, 5), ind.sma(c, 10), ind.sma(c, 20)
    sma60, sma120 = ind.sma(c, 60), ind.sma(c, 120)
    rsi2 = ind.rsi(c, 2)
    a = atr(h, l, c)
    day_ret = c / c.shift() - 1

    pullback = liquid & (c > sma120) & (sma20 > sma60) & (rsi2 < cfg.pullback_rsi2)

    prior_high = h.shift().rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max()
    vol_ratio = v / v.shift().rolling(20, min_periods=20).mean()
    breakout = (liquid & (c > prior_high) & (vol_ratio >= cfg.breakout_vol_mult)
                & (c > sma60) & (day_ret > 0) & (day_ret < 0.29))  # 상한가는 다음날 매수 어려움

    return {
        "signal": {"pullback": pullback.fillna(False), "breakout": breakout.fillna(False)},
        # 여러 신호 중 우선순위: 눌림목은 더 과매도일수록, 돌파는 거래량이 클수록
        "priority": {"pullback": -rsi2, "breakout": vol_ratio},
        "exit": {"pullback": c > sma5, "breakout": c < sma10},
        "atr": a, "rsi2": rsi2, "vol_ratio": vol_ratio, "trade_value": trade_value,
    }


def stop_price(setup: str, entry: float, atr_value: float, cfg: SwingConfig) -> float:
    if setup == "pullback":
        return entry * (1 - cfg.pullback_stop)
    if np.isnan(atr_value):
        return entry * (1 - cfg.pullback_stop)
    return entry - cfg.breakout_atr_stop * atr_value


@dataclass
class SwingResult:
    equity: pd.Series
    trades: pd.DataFrame
    benchmark: pd.Series | None

    def summary(self) -> dict:
        out = {"전략": metrics(self.equity)}
        t = self.trades
        if len(t):
            out["전략"].update({
                "거래 수": float(len(t)),
                "승률": float((t["return"] > 0).mean()),
                "평균 거래 수익률": float(t["return"].mean()),
                "평균 보유일": float(t["days"].mean()),
            })
        if self.benchmark is not None:
            b = self.benchmark.reindex(self.equity.index).ffill().dropna()
            if len(b):
                out["벤치마크"] = metrics(b / b.iloc[0])
        return out


def backtest(px: dict, setup: str, cfg: SwingConfig | None = None, start=None,
             benchmark: pd.Series | None = None) -> SwingResult:
    cfg = cfg or SwingConfig()
    ind_ = compute(px, cfg)
    o, l = px["open"], px["low"]
    c = px["close"].ffill()  # 평가용 (거래정지일은 직전 종가)
    sig, pri, ex = ind_["signal"][setup], ind_["priority"][setup], ind_["exit"][setup]
    a = ind_["atr"]
    max_hold = cfg.pullback_max_hold if setup == "pullback" else cfg.breakout_max_hold
    half_cost = cfg.cost_roundtrip / 2

    dates = c.index
    start_i = int(dates.searchsorted(pd.Timestamp(start))) if start is not None else 1
    start_i = max(start_i, 1)

    cash = 1.0
    exited_today = set()
    pos = {}      # ticker -> dict(units, entry, stop, entry_i, exit_next)
    trades = []
    equity = []

    def close_pos(t, price, i, reason):
        nonlocal cash
        p = pos.pop(t)
        exited_today.add(t)
        proceeds = p["units"] * price * (1 - half_cost)
        cash += proceeds
        trades.append({"ticker": t, "entry_date": dates[p["entry_i"]], "exit_date": dates[i],
                       "entry": p["entry"], "exit": price,
                       "return": proceeds / p["cost"] - 1, "days": i - p["entry_i"], "reason": reason})

    for i in range(start_i, len(dates)):
        d = dates[i]
        exited_today = set()
        # 1) 어제 종가 기준 청산 신호 → 오늘 시가 매도
        for t in [t for t, p in pos.items() if p["exit_next"]]:
            op = o.at[d, t]
            if not np.isnan(op):
                close_pos(t, op, i, pos[t]["exit_reason"])
        # 2) 손절 (갭 하락이면 시가에 체결)
        for t in list(pos):
            p = pos[t]
            lo, op = l.at[d, t], o.at[d, t]
            if not np.isnan(lo) and lo <= p["stop"]:
                close_pos(t, min(op, p["stop"]) if not np.isnan(op) else p["stop"], i, "손절")
        # 3) 어제 신호 → 오늘 시가 매수
        free = cfg.max_positions - len(pos)
        if free > 0:
            prev = dates[i - 1]
            cands = sig.loc[prev]
            cands = cands[cands].index.difference(list(pos) + list(exited_today))
            if len(cands):
                order = pri.loc[prev, cands].sort_values(ascending=False).index
                equity_now = cash + sum(p["units"] * c.at[prev, t] for t, p in pos.items())
                for t in order:
                    if free == 0:
                        break
                    op = o.at[d, t]
                    if np.isnan(op):
                        continue
                    alloc = min(equity_now / cfg.max_positions, cash)
                    if alloc <= 0:
                        break
                    units = alloc * (1 - half_cost) / op
                    cash -= alloc
                    pos[t] = {"units": units, "entry": op, "cost": alloc, "entry_i": i,
                              "stop": stop_price(setup, op, a.at[prev, t], cfg), "exit_next": False,
                              "exit_reason": ""}
                    free -= 1
                    lo = l.at[d, t]  # 매수 당일 손절 (보수적으로 체결 가정)
                    if not np.isnan(lo) and lo <= pos[t]["stop"]:
                        close_pos(t, pos[t]["stop"], i, "손절")
                        free += 1
        # 4) 종가 평가와 다음날 청산 예약
        for t, p in pos.items():
            if bool(ex.at[d, t]):
                p["exit_next"], p["exit_reason"] = True, "매도신호"
            elif i - p["entry_i"] >= max_hold:
                p["exit_next"], p["exit_reason"] = True, "보유기간"
        equity.append(cash + sum(p["units"] * c.at[d, t] for t, p in pos.items()))

    eq = pd.Series(equity, index=dates[start_i:])
    bench = benchmark.loc[dates[start_i]:] if benchmark is not None else None
    return SwingResult(eq, pd.DataFrame(trades), bench)


def data_quality(px: dict) -> dict:
    """일봉 데이터 정합성 점검 (고가/저가가 시가·종가를 감싸는지 등)."""
    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    valid = o.notna() & h.notna() & l.notna() & c.notna()
    n = int(valid.values.sum())
    bad_hi = ((h < np.maximum(o, c) * 0.999) & valid).values.sum()
    bad_lo = ((l > np.minimum(o, c) * 1.001) & valid).values.sum()
    gap = (o / c.shift() - 1).abs()
    return {"rows": n, "high<max(open,close)": bad_hi / max(n, 1), "low>min(open,close)": bad_lo / max(n, 1),
            "median |open/prev_close-1|": float(np.nanmedian(gap.values)),
            "share |gap|>15%": float((gap > 0.15).values.sum() / max(n, 1))}


def today_signals(px: dict, cfg: SwingConfig | None = None, top: int = 10) -> dict:
    """마지막 거래일 기준 셋업별 신호 목록."""
    cfg = cfg or SwingConfig()
    ind_ = compute(px, cfg)
    last = px["close"].index[-1]
    close = px["close"].loc[last]
    out = {}
    for setup in SETUPS:
        mask = ind_["signal"][setup].loc[last]
        tickers = mask[mask].index
        pri = ind_["priority"][setup].loc[last, tickers].sort_values(ascending=False)
        rows = []
        for t in pri.index[:top]:
            cl = close[t]
            rows.append({
                "ticker": t, "close": cl,
                "stop": stop_price(setup, cl, ind_["atr"].at[last, t], cfg),
                "rsi2": ind_["rsi2"].at[last, t], "vol_ratio": ind_["vol_ratio"].at[last, t],
                "trade_value": ind_["trade_value"].at[last, t],
            })
        out[setup] = pd.DataFrame(rows)
    return out
