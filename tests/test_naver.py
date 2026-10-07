import pandas as pd

from analyzer import data, data_naver

SISE = """
 [['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
["20240102", 78200, 79800, 78200, 79600, 17142847, 53.29],
["20240103", 78500, 78800, 77000, 77000, 21753644, 53.27]
]
"""

ROW = '''<tr onMouseOver="mouseOver(this)">
<td class="no">1</td>
<td><a href="/item/main.naver?code=005930" class="tltle">삼성전자</a></td>
<td class="number">60,000</td>
<td class="number"><img src="x.gif"><span class="tah p11 red02">1,000</span></td>
<td class="number"><span class="tah p11 red01">+1.69%</span></td>
<td class="number">100</td>
<td class="number">3,581,869</td>
<td class="number">5,969,783</td>
<td class="number">50.10</td>
<td class="number">10,000,000</td>
<td class="number">12.50</td>
<td class="number">-3.20</td>
</tr>'''


def test_parse_sise_json():
    df = data_naver.parse_sise_json(SISE)
    assert list(df["종가"]) == [79600, 77000]
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_parse_market_sum():
    df = data_naver.parse_market_sum("<table>" + ROW + "</table>")
    r = df.loc["005930"]
    assert r["name"] == "삼성전자"
    assert r["close"] == 60000
    assert r["market_cap"] == 3_581_869e8
    assert r["trading_value"] == 60000 * 10_000_000
    assert r["PER"] == 12.5 and r["ROE"] == -3.2


def test_source_selection(monkeypatch):
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("DATA_SOURCE", raising=False)
    assert data.source_name() == "naver"
    monkeypatch.setenv("KRX_ID", "a")
    monkeypatch.setenv("KRX_PW", "b")
    assert data.source_name() == "krx"
    monkeypatch.setenv("DATA_SOURCE", "naver")
    assert data.source_name() == "naver"
