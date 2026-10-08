import numpy as np
import pandas as pd
import pytest

from analyzer import dca


def flat(start="2023-01-02", periods=600, price=10000.0):
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(price, index=idx)


def test_flat_price_returns_only_costs():
    r = dca.simulate("x", {"A": flat()}, {"A": 1.0}, 50000, "2023-01-02", cost=0.001)
    weeks = len(dca.weekly_buy_days(flat().index))
    assert r.invested == pytest.approx(50000 * weeks)
    assert r.gain_pct == pytest.approx(-0.001, abs=1e-9)


def test_dividend_reinvested_after_tax():
    s = flat()
    r = dca.simulate("x", {"A": s}, {"A": 1.0}, 50000, "2023-01-02", cost=0.0, dps={"A": {2023: 1000.0}})
    # 2023년 말 보유 주식 x 1000원 x (1-15.4%) 가 2024년 4월에 재투자
    weeks_2023 = len([d for d in dca.weekly_buy_days(s.index) if d.year == 2023])
    held = weeks_2023 * 50000 / 10000
    assert r.dividends == pytest.approx(held * 1000 * (1 - dca.TAX))
    assert r.value == pytest.approx(r.invested + r.dividends)


def test_weights_and_render():
    a, b = flat(), flat(price=20000) * np.linspace(1, 2, 600)
    r = dca.simulate("ab", {"A": a, "B": b}, {"A": 1, "B": 1}, 50000, "2023-01-02")
    assert r.value > r.invested
    md = dca.render([r], 50000, "2023-01-02", "2025-04-18", ["note"])
    assert "적립식" in md and "중간 최악" in md
