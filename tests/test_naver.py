import pandas as pd

from analyzer import data, data_naver

SISE = """
 [['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
["20240102", 78200, 79800, 78200, 79600, 17142847, 53.29],
["20240103", 78500, 78800, 77000, 77000, 21753644, 53.27]
]
"""

MARKET = {"stocks": [
    {"stockEndType": "stock", "itemCode": "005930", "stockName": "삼성전자", "sosok": "0",
     "closePriceRaw": "268500", "accumulatedTradingValueRaw": "4420466000000",
     "marketValueRaw": "1569725806248000"},
    {"stockEndType": "etf", "itemCode": "069500", "stockName": "KODEX 200",
     "closePriceRaw": "1", "accumulatedTradingValueRaw": "1", "marketValueRaw": "1"},
]}

INTEGRATION = {"totalInfos": [
    {"code": "per", "key": "PER", "value": "12.04배"},
    {"code": "pbr", "key": "PBR", "value": "3.12배"},
    {"code": "dividendYieldRatio", "key": "배당수익률", "value": "0.62%"},
]}


def test_parse_sise_json():
    df = data_naver.parse_sise_json(SISE)
    assert list(df["종가"]) == [79600, 77000]
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_parse_market_value_skips_non_stock():
    df = data_naver.parse_market_value(MARKET)
    assert list(df.index) == ["005930"]
    r = df.loc["005930"]
    assert r["name"] == "삼성전자" and r["market"] == "코스피"
    assert r["market_cap"] == 1569725806248000
    assert r["trading_value"] == 4420466000000


def test_parse_integration():
    f = data_naver.parse_integration(INTEGRATION)
    assert f == {"PER": 12.04, "PBR": 3.12, "DIV": 0.62}
    missing = data_naver.parse_integration({"totalInfos": [{"code": "per", "value": "N/A"}]})
    assert pd.isna(missing["PER"]) and pd.isna(missing["PBR"])


def test_source_selection(monkeypatch):
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("DATA_SOURCE", raising=False)
    assert data.source_name() == "naver"
    monkeypatch.setenv("KRX_ID", "a")
    monkeypatch.setenv("KRX_PW", "b")
    assert data.source_name() == "krx"
    monkeypatch.setenv("DATA_SOURCE", "naver")
    assert data.source_name() == "naver"


def test_stock_news_flattens_unknown_shape(monkeypatch):
    payload = [{"total": 2, "items": [{"title": "<b>디바이스</b> 수주", "datetime": "202610081530", "officeName": "A"},
                                      {"title": "디바이스 실적", "datetime": "202610071000", "officeName": "B"}]}]
    monkeypatch.setattr(data_naver, "_get_json_or_text", lambda *a, **k: payload)
    news = data_naver.stock_news("187870")
    assert [n["title"] for n in news] == ["디바이스 수주", "디바이스 실적"]
    assert news[0]["office"] == "A"
