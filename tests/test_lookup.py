import numpy as np
import pandas as pd

from analyzer import lookup


def test_find_exact_then_contains():
    listing = pd.DataFrame({"name": ["디바이스", "에이디바이스", "삼성전자"], "market_cap": [1, 5, 9]},
                           index=["000010", "000020", "005930"])
    assert list(lookup.find(listing, "디바이스").index) == ["000010"]
    assert list(lookup.find(listing.drop("000010"), "디바이스").index) == ["000020"]


def test_stats_checklist_and_render():
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(np.linspace(10000, 15000, 300), index=idx)
    df = pd.DataFrame({"종가": close, "거래량": 100000.0})
    st = lookup.price_stats(df)
    assert st["ret_1y"] > 0 and st["above_ma120"] is True
    info = {"per": ("PER", "12.04배", "2026.06."), "pbr": ("PBR", "1.10배", None),
            "cnsPer": ("추정PER", "8.0배", None), "dividendYieldRatio": ("배당수익률", "2.5%", None)}
    checks = lookup.checklist(info, st, 5e11)
    names = [c[0] for c in checks]
    assert "수익성" in names and "이익 전망" in names
    md = lookup.render("테스트", [{"ticker": "000010", "name": "테스트", "market": "코스닥",
                                  "stats": st, "info": info, "checks": checks}])
    assert "체크리스트 합계" in md and "투자 권유가 아닙니다" in md


def test_loss_making_company_flagged():
    st = {"avg_value_20d": 5e8, "vol_1y": 0.8, "mdd_1y": -0.6, "above_ma120": False, "rsi14": 40.0}
    checks = dict((k, v) for k, v, _ in lookup.checklist({"per": ("PER", "-3.2배", None)}, st, 1e11))
    assert checks["수익성"] == "나쁨" and checks["거래 활발도"] == "주의" and checks["추세"] == "주의"
