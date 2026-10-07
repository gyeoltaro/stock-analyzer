import numpy as np
import pandas as pd
import pytest

from analyzer import backtest, indicators as ind
from analyzer.report import backtest_report, daily_report
from analyzer.scoring import ScoreConfig, score, technical_candidates


def make_prices(n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    drifts = {"UP": 0.0015, "FLAT": 0.0, "DOWN": -0.001, "NOISY": 0.0008}
    vols = {"UP": 0.01, "FLAT": 0.01, "DOWN": 0.01, "NOISY": 0.04}
    data = {t: 10000 * np.exp(np.cumsum(drifts[t] + vols[t] * rng.standard_normal(n_days))) for t in drifts}
    return pd.DataFrame(data, index=idx)


def test_rsi_bounds_and_monotonic_up():
    up = pd.DataFrame({"A": np.arange(1, 50, dtype=float)})
    r = ind.rsi(up).iloc[-1, 0]
    assert r == pytest.approx(100.0)
    close = make_prices()
    vals = ind.rsi(close).stack()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_momentum_skip():
    close = pd.DataFrame({"A": np.arange(1, 301, dtype=float)})
    m = ind.momentum(close, 252, 21).iloc[-1, 0]
    assert m == pytest.approx(close["A"].iloc[-22] / close["A"].iloc[-253] - 1)


def test_score_filters_downtrend_and_ranks():
    close = make_prices()
    ranked = score(close, None, ScoreConfig(rsi_max=101))
    assert "DOWN" not in ranked.index
    assert "UP" in ranked.index
    assert ranked["score"].between(0, 100).all()
    assert ranked["score"].is_monotonic_decreasing


def test_score_uses_fundamentals_and_skips_losses():
    close = make_prices()
    fund = pd.DataFrame({"PER": [-5.0, 8.0, 10.0, 30.0], "PBR": [0.5, 0.8, 1.0, 3.0]},
                        index=["UP", "FLAT", "DOWN", "NOISY"])
    ranked = score(close, fund, ScoreConfig(rsi_max=101))
    assert ranked["value"].notna().all()


def test_backtest_runs_and_charges_cost():
    close = make_prices(600)
    cfg = ScoreConfig(top_n=2, rsi_max=101)
    free = backtest.run(close, None, cfg, cost=0.0)
    costly = backtest.run(close, None, cfg, cost=0.01)
    assert free.equity.iloc[-1] > costly.equity.iloc[-1]
    assert len(free.holdings) > 0
    s = free.summary()
    assert "연복리(CAGR)" in s["전략"]


def test_backtest_no_lookahead():
    """미래 가격을 바꿔도 그 이전 리밸런싱 선택은 같아야 한다."""
    close = make_prices(600)
    cfg = ScoreConfig(top_n=2, rsi_max=101)
    a = backtest.run(close, None, cfg)
    tampered = close.copy()
    tampered.iloc[-30:] *= 5
    b = backtest.run(tampered, None, cfg)
    cutoff = close.index[-31]
    for day, picks in a.holdings.items():
        if day <= cutoff:
            assert b.holdings[day] == picks


def test_metrics_known_values():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2], index=pd.bdate_range("2024-01-01", periods=4))
    m = backtest.metrics(eq)
    assert m["총수익률"] == pytest.approx(0.2)
    assert m["최대낙폭(MDD)"] == pytest.approx(0.99 / 1.1 - 1)


def test_reports_render():
    close = make_prices()
    ranked = score(close, None, ScoreConfig(rsi_max=101))
    md = daily_report("20230601", ranked, pd.Series({"UP": "상승주"}), 4, 3)
    assert "상승주" in md and "투자 권유가 아닙니다" in md
    res = backtest.run(make_prices(600), None, ScoreConfig(top_n=2, rsi_max=101))
    md2 = backtest_report(res.summary(), res.holdings, pd.Series(dtype=str), "test")
    assert "생존편향" in md2


def test_technical_candidates_match_score_universe():
    close = make_prices()
    cfg = ScoreConfig(rsi_max=101)
    assert set(technical_candidates(close, cfg)) == set(score(close, None, cfg).index)


def test_missing_fundamentals_do_not_help():
    close = make_prices()
    cfg = ScoreConfig(rsi_max=101)
    base = pd.DataFrame({"PER": [5.0, 5.0, 5.0, 5.0], "PBR": [0.5, 0.5, 0.5, 0.5]},
                        index=["UP", "FLAT", "DOWN", "NOISY"])
    with_up = score(close, base, cfg)
    no_up = base.drop(index="UP")
    without_up = score(close, no_up, cfg)
    assert without_up.loc["UP", "score"] < with_up.loc["UP", "score"]
