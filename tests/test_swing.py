import numpy as np
import pandas as pd
import pytest

from analyzer import swing
from analyzer.data import build_panel
from analyzer.report import swing_backtest_report, swing_report

CFG = swing.SwingConfig(min_avg_value=0, min_price=0)


def make_panel(n=400, n_tickers=6, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    frames = {}
    for k in range(n_tickers):
        drift = 0.001 * (k - 2)
        close = 10000 * np.exp(np.cumsum(drift + 0.02 * rng.standard_normal(n)))
        open_ = close * (1 + 0.005 * rng.standard_normal(n))
        high = np.maximum(open_, close) * (1 + np.abs(0.01 * rng.standard_normal(n)))
        low = np.minimum(open_, close) * (1 - np.abs(0.01 * rng.standard_normal(n)))
        vol = rng.integers(50_000, 150_000, n).astype(float)
        frames[f"T{k}"] = pd.DataFrame({"시가": open_, "고가": high, "저가": low, "종가": close, "거래량": vol},
                                       index=idx)
    return build_panel(frames)


def test_build_panel_shapes_and_zero_prices():
    px = make_panel(50, 2)
    assert set(px) == {"open", "high", "low", "close", "volume"}
    px2 = build_panel({"A": pd.DataFrame({"시가": [0.0, 10], "고가": [0.0, 11], "저가": [0.0, 9],
                                          "종가": [0.0, 10], "거래량": [0.0, 5]},
                                         index=pd.bdate_range("2024-01-01", periods=2))})
    assert np.isnan(px2["close"].iloc[0, 0]) and px2["volume"].iloc[0, 0] == 0


@pytest.mark.parametrize("setup", swing.SETUPS)
def test_backtest_runs_and_respects_position_limit(setup):
    px = make_panel()
    res = swing.backtest(px, setup, swing.SwingConfig(min_avg_value=0, min_price=0, max_positions=3))
    assert len(res.equity) > 0 and np.isfinite(res.equity).all()
    if len(res.trades):
        t = res.trades
        assert (t["exit_date"] >= t["entry_date"]).all()
        # 동시에 3종목 초과 보유하지 않음
        for d in res.equity.index[::25]:
            held = ((t["entry_date"] <= d) & (t["exit_date"] > d)).sum()
            assert held <= 3


def test_costs_reduce_returns():
    px = make_panel()
    free = swing.backtest(px, "pullback", swing.SwingConfig(min_avg_value=0, min_price=0, cost_roundtrip=0))
    costly = swing.backtest(px, "pullback", swing.SwingConfig(min_avg_value=0, min_price=0, cost_roundtrip=0.02))
    assert len(free.trades) > 0
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_no_lookahead():
    px = make_panel()
    a = swing.backtest(px, "pullback", CFG)
    tampered = {k: v.copy() for k, v in px.items()}
    for k in ("open", "high", "low", "close"):
        tampered[k].iloc[-20:] *= 3
    b = swing.backtest(tampered, "pullback", CFG)
    cutoff = px["close"].index[-21]
    ea = a.trades[a.trades["entry_date"] <= cutoff].reset_index(drop=True)
    eb = b.trades[b.trades["entry_date"] <= cutoff].reset_index(drop=True)
    assert list(ea["ticker"]) == list(eb["ticker"])
    assert list(ea["entry_date"]) == list(eb["entry_date"])


def test_stop_loss_caps_loss_on_crash():
    """매수 다음날 폭락하면 손절가(또는 갭 시가)에 청산된다."""
    idx = pd.bdate_range("2022-01-03", periods=200)
    close = pd.Series(np.linspace(10000, 20000, 200), index=idx)
    close.iloc[150:153] *= [0.96, 0.92, 0.88]       # 급락 → RSI(2) 과매도 신호
    o = close.shift().fillna(close.iloc[0])
    low = np.minimum(o, close) * 0.995
    high = np.maximum(o, close) * 1.005
    low.iloc[154] = close.iloc[153] * 0.5             # 매수 다음날 장중 폭락
    frames = {"A": pd.DataFrame({"시가": o, "고가": high, "저가": low, "종가": close, "거래량": 1e5})}
    px = build_panel(frames)
    res = swing.backtest(px, "pullback", swing.SwingConfig(min_avg_value=0, min_price=0, cost_roundtrip=0))
    assert len(res.trades)
    worst = res.trades["return"].min()
    assert worst == pytest.approx(-0.07)
    # 손절된 날 같은 종목을 다시 사지 않는다
    t = res.trades
    for _, row in t.iterrows():
        assert not ((t["entry_date"] == row["exit_date"]) & (t.index != row.name)).any()


def test_reports_render():
    px = make_panel()
    sig = swing.today_signals(px, CFG)
    md = swing_report("20230601", sig, pd.Series(dtype=str), 6)
    assert "눌림목" in md and "돌파" in md
    res = {s: swing.backtest(px, s, CFG) for s in swing.SETUPS}
    md2 = swing_backtest_report(res, "test", CFG)
    assert "승률" in md2 or "총수익률" in md2


def _ma15_scenario(after):
    """꾸준한 상승 후 15일선까지 눌림 → 이후 가격 경로(after)."""
    idx = pd.bdate_range("2022-01-03", periods=120 + len(after))
    up = [10000 * 1.01 ** k for k in range(110)]       # 하루 1%씩 상승 (15일선보다 약 7% 위)
    dip = [up[-1] * 0.985 ** k for k in range(1, 11)]  # 하루 1.5%씩 조정
    close = pd.Series(up + dip + list(after), index=idx[: 120 + len(after)])
    o = close.shift().fillna(close.iloc[0])
    low = np.minimum(o, close) * 0.99
    high = np.maximum(o, close) * 1.01
    return close, o, low, high


def test_ma15_signal_and_take_profit():
    close, o, low, high = _ma15_scenario([])
    px = build_panel({"A": pd.DataFrame({"시가": o, "고가": high, "저가": low, "종가": close, "거래량": 1e5})})
    sig = swing.compute(px, CFG)["signal"]["ma15"]["A"]
    assert sig.any(), "15일선 터치 신호가 있어야 함"
    first = sig[sig].index[0]
    # 신호 다음날부터 급등 → +10% 익절
    n_after = 10
    i = close.index.get_loc(first)
    path = list(close.iloc[: i + 1]) + [close.iloc[i] * (1 + 0.04 * k) for k in range(1, n_after + 1)]
    idx = pd.bdate_range("2022-01-03", periods=len(path))
    c2 = pd.Series(path, index=idx)
    o2 = c2.shift().fillna(c2.iloc[0])
    px2 = build_panel({"A": pd.DataFrame({"시가": o2, "고가": np.maximum(o2, c2) * 1.01,
                                          "저가": np.minimum(o2, c2) * 0.99, "종가": c2, "거래량": 1e5})})
    res = swing.backtest(px2, "ma15", swing.SwingConfig(min_avg_value=0, min_price=0, cost_roundtrip=0))
    assert "익절" in set(res.trades["reason"])
    tp = res.trades[res.trades["reason"] == "익절"].iloc[0]
    assert tp["return"] >= 0.10 - 1e-9


def test_ma15_hold_never_exits_on_signal():
    px = make_panel()
    res = swing.backtest(px, "ma15_hold", CFG)
    if len(res.trades):
        assert "매도신호" not in set(res.trades["reason"])


def test_fixed_take_profit_and_stop_loss():
    px = make_panel(seed=3)
    cfg = swing.SwingConfig(min_avg_value=0, min_price=0, cost_roundtrip=0,
                            fixed_take_profit=0.15, fixed_stop_loss=0.05, fixed_max_hold=60)
    res = swing.backtest(px, "pullback", cfg)
    t = res.trades
    assert len(t)
    assert set(t["reason"]) <= {"익절", "손절", "보유기간"}
    assert (t.loc[t["reason"] == "손절", "return"] <= -0.05 + 1e-9).all()
    assert (t.loc[t["reason"] == "익절", "return"] >= 0.15 - 1e-9).all()
    assert t["days"].max() <= 60
