import numpy as np
import pandas as pd

from analyzer import dividend


def test_payout_ratio_and_score():
    assert dividend.payout_ratio(5.0, 8.0) == 40.0
    assert np.isnan(dividend.payout_ratio(5.0, -3.0))
    assert dividend.payout_score(45) == 1.0
    assert dividend.payout_score(150) == 0.0
    assert dividend.payout_score(15) < 1.0


def test_history_score_prefers_steady_growth():
    assert dividend.history_score([1000, 1100, 1200, 1300]) > dividend.history_score([1300, 0, 500, 400])
    assert dividend.history_score([]) == 0.5


def test_rank_and_render():
    df = pd.DataFrame({
        "name": ["안정배당", "고배당위험", "저배당"],
        "DIV": [5.0, 7.0, 2.6], "PER": [8.0, 25.0, 10.0],
        "vol_1y": [0.25, 0.7, 0.3], "mdd_1y": [-0.15, -0.5, -0.2], "ret_1y": [0.1, -0.2, 0.05],
        "dps_hist": [[1000, 1100, 1200], [500, 0, 700], [100, 100, 100]],
    }, index=["A", "B", "C"])
    ranked = dividend.rank(df)
    assert ranked.index[0] == "A"            # 배당 높고 안정적
    assert ranked.loc["B", "payout"] == 175  # 번 돈보다 많이 줌
    md = dividend.render(ranked, 200, 2.5, 10, 50000)
    assert "배당주 순위" in md and "안정배당" in md and "연 배당금" in md
