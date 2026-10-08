"""사용법:
  python main.py report   [--date YYYYMMDD] [--universe 0] [--top 20]
  python main.py backtest [--years 5] [--universe 0] [--top 20]
  python main.py swing           # 오늘의 단기 스윙 신호
  python main.py swing-backtest [--years 3]

--universe 0 은 전 종목, N 이면 시가총액 상위 N개.
"""
import argparse
from pathlib import Path

import pandas as pd

from analyzer import backtest, data, swing
from analyzer.report import backtest_report, daily_report, swing_backtest_report, swing_report
from analyzer.scoring import ScoreConfig, score, technical_candidates

REPORTS = Path(__file__).parent / "reports"


def cmd_report(args):
    date = data.latest_business_day(args.date)
    print(f"기준일 {date}, 유니버스 조회 중...")
    uni = data.universe(date, args.universe)
    close = data.close_prices(uni.index, data.lookback_start(date, 420), date)
    cfg = ScoreConfig(top_n=args.top)
    candidates = technical_candidates(close, cfg)
    print(f"가격 조건 통과 {len(candidates)}개, 재무지표 조회 중...")
    fund = data.fundamentals(date, candidates)
    ranked = score(close, fund, cfg)
    md = daily_report(date, ranked, uni["name"], len(uni), args.top)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{pd.Timestamp(date):%Y-%m-%d}.md"
    out.write_text(md, encoding="utf-8")
    (REPORTS / "latest.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"저장: {out}")


def cmd_backtest(args):
    end = data.latest_business_day(args.date)
    test_start = data.lookback_start(end, int(args.years * 365))
    fetch_start = data.lookback_start(test_start, 420)  # 첫 리밸런싱에 필요한 과거 데이터
    print(f"백테스트 {test_start}~{end}, 유니버스 조회 중...")
    uni = data.universe(end, args.universe)
    close = data.close_prices(uni.index, fetch_start, end)
    bench = data.benchmark(fetch_start, end)

    cache = {}
    if not data.has_point_in_time_fundamentals():
        print("[안내] 네이버 데이터는 과거 PER/PBR이 없어 백테스트에서 가치 팩터를 제외합니다.")

    def fund_at(day):
        if not data.has_point_in_time_fundamentals():
            return None
        key = data.ymd(day)
        if key not in cache:
            try:
                cache[key] = data.fundamentals(key)
            except Exception as e:
                print(f"[warn] {key} 펀더멘털 실패: {e}")
                cache[key] = None
        return cache[key]

    res = backtest.run(close, fund_at, ScoreConfig(top_n=args.top), bench, cost=args.cost, start=test_start)
    md = backtest_report(res.summary(), res.holdings, uni["name"], f"{test_start}~{end}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "backtest.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_swing(args):
    date = data.latest_business_day(args.date)
    print(f"기준일 {date}, 유니버스 조회 중...")
    uni = data.universe(date, args.universe)
    px = data.ohlcv_panel(uni.index, data.lookback_start(date, 260), date)
    sig = swing.today_signals(px, swing.SwingConfig(), args.top)
    md = swing_report(date, sig, uni["name"], len(uni))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"swing-{pd.Timestamp(date):%Y-%m-%d}.md").write_text(md, encoding="utf-8")
    (REPORTS / "swing-latest.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_swing_backtest(args):
    end = data.latest_business_day(args.date)
    test_start = data.lookback_start(end, int(args.years * 365))
    fetch_start = data.lookback_start(test_start, 220)  # 120일선 계산용
    print(f"스윙 백테스트 {test_start}~{end}, 유니버스 조회 중...")
    uni = data.universe(end, args.universe)
    px = data.ohlcv_panel(uni.index, fetch_start, end)
    bench = data.benchmark(fetch_start, end)
    print("데이터 점검:", swing.data_quality(px))
    cfg = swing.SwingConfig(max_positions=args.positions, cost_roundtrip=args.cost,
                            fixed_take_profit=args.tp, fixed_stop_loss=args.sl, fixed_max_hold=args.max_hold)
    setups = args.setups.split(",") if args.setups else swing.SETUPS
    results = {s: swing.backtest(px, s, cfg, start=test_start, benchmark=bench) for s in setups}
    label = f"{test_start}~{end}, " + (f"시가총액 상위 {args.universe}개" if args.universe else "전 종목")
    if args.tp is not None and args.sl is not None:
        label += f", 고정 매도 +{args.tp * 100:.0f}% 익절 / -{args.sl * 100:.0f}% 손절 / 최대 {args.max_hold}거래일"
    md = swing_backtest_report(results, label, cfg)
    REPORTS.mkdir(exist_ok=True)
    name = "swing-backtest" + (f"-top{args.universe}" if args.universe else "")
    if args.tp is not None and args.sl is not None:
        name += f"-tp{args.tp * 100:.0f}-sl{args.sl * 100:.0f}"
    name += ".md"
    (REPORTS / name).write_text(md, encoding="utf-8")
    print(md)


def cmd_lookup(args):
    from analyzer import data_naver, lookup
    listing = data_naver.market_value_ranking()
    hits = lookup.find(listing, args.query).head(args.max)
    rows = []
    end = data_naver.latest_business_day()
    start = data.lookback_start(end, 400)
    for t, h in hits.iterrows():
        df = data_naver.ohlcv(t, start, end)
        if df.empty:
            continue
        info = data_naver.integration_info(t)
        st = lookup.price_stats(df)
        try:
            news = data_naver.stock_news(t)
        except Exception as e:
            print(f"[warn] 뉴스 조회 실패: {e}")
            news = None
        try:
            deals = data_naver.deal_trends(t)
        except Exception as e:
            print(f"[warn] 매매동향 조회 실패: {e}")
            deals = []
        rows.append({"ticker": t, "name": h["name"], "market": h["market"], "news": news, "deals": deals, "stats": st, "info": info,
                     "checks": lookup.checklist(info, st, h["market_cap"])})
    md = lookup.render(args.query, rows)
    REPORTS.mkdir(exist_ok=True)
    safe = "".join(ch for ch in args.query if ch.isalnum())
    (REPORTS / f"lookup-{safe}.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_dividend(args):
    from analyzer import data_naver, dividend, lookup
    listing = data_naver.market_value_ranking()
    listing = listing[listing.index.str.endswith("0")].sort_values("market_cap", ascending=False).head(args.universe)
    print(f"상위 {len(listing)}개 재무지표 조회 중...")
    fund = data_naver.fundamentals(None, list(listing.index))
    cand = fund[(fund["DIV"] >= args.min_yield) & (fund["PER"] > 0)].index
    print(f"배당수익률 {args.min_yield}% 이상·흑자 {len(cand)}개, 주가·배당 이력 조회 중...")
    end = data_naver.latest_business_day()
    start = data.lookback_start(end, 400)
    rows = {}
    for i, t in enumerate(cand):
        try:
            st = lookup.price_stats(data_naver.ohlcv(t, start, end))
        except Exception as e:
            print(f"[warn] {t} 시세 실패: {e}")
            continue
        try:
            hist = list(data_naver.dividend_history(t, debug=(i == 0)).values())
        except Exception as e:
            print(f"[warn] {t} 배당 이력 실패: {e}")
            hist = []
        rows[t] = {"name": listing.at[t, "name"], "DIV": fund.at[t, "DIV"], "PER": fund.at[t, "PER"],
                   "vol_1y": st["vol_1y"], "mdd_1y": st["mdd_1y"], "ret_1y": st["ret_1y"], "dps_hist": hist}
    ranked = dividend.rank(pd.DataFrame.from_dict(rows, orient="index"))
    md = dividend.render(ranked, len(listing), args.min_yield, args.top, args.weekly)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "dividend.md").write_text(md, encoding="utf-8")
    print(md)


DCA_PORTFOLIOS = {
    "배당주 묶음 (기업은행·KT·KT&G)": {"024110": 0.4, "030200": 0.3, "033780": 0.3},
    "배당주 묶음 (우리금융·KT·KT&G)": {"316140": 0.4, "030200": 0.3, "033780": 0.3},
    "삼성전자": {"005930": 1.0},
    "디바이스": {"187870": 1.0},
}


def cmd_dca(args):
    from analyzer import data_naver, dca
    end = data_naver.latest_business_day()
    start = data.lookback_start(end, int(args.years * 365))
    tickers = sorted({t for p in DCA_PORTFOLIOS.values() for t in p})
    closes, dps, notes = {}, {}, []
    for t in tickers:
        closes[t] = data_naver.ohlcv(t, start, end)["종가"].astype(float)
        try:
            dps[t] = data_naver.dividend_history(t)
        except Exception as e:
            dps[t] = {}
            print(f"[warn] {t} 배당 이력 실패: {e}")
        notes.append(f"{t} 주당배당금: " + (", ".join(f"{y}년 {v:,.0f}원" for y, v in dps[t].items()) or "자료 없음"))
    results = [dca.simulate(name, {t: closes[t] for t in w}, w, args.amount, start, end, dps)
               for name, w in DCA_PORTFOLIOS.items()]
    bench = data_naver.benchmark(start, end)
    results.append(dca.simulate("코스피200 (지수)", {"K200": bench}, {"K200": 1.0}, args.amount, start, end))
    md = dca.render(results, args.amount, start, end, notes)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "dca.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    p = argparse.ArgumentParser(description="한국 주식 규칙 기반 분석")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="오늘의 종목 점수 리포트")
    r.add_argument("--date")
    r.add_argument("--universe", type=int, default=0, help="0=전 종목")
    r.add_argument("--top", type=int, default=20)
    r.set_defaults(func=cmd_report)
    b = sub.add_parser("backtest", help="월간 리밸런싱 백테스트")
    b.add_argument("--date")
    b.add_argument("--years", type=float, default=5)
    b.add_argument("--universe", type=int, default=0, help="0=전 종목")
    b.add_argument("--top", type=int, default=20)
    b.add_argument("--cost", type=float, default=0.003)
    b.set_defaults(func=cmd_backtest)
    sw = sub.add_parser("swing", help="오늘의 단기 스윙 신호")
    sw.add_argument("--date")
    sw.add_argument("--universe", type=int, default=0, help="0=전 종목")
    sw.add_argument("--top", type=int, default=10)
    sw.set_defaults(func=cmd_swing)
    sb = sub.add_parser("swing-backtest", help="단기 스윙 백테스트")
    sb.add_argument("--date")
    sb.add_argument("--years", type=float, default=3)
    sb.add_argument("--universe", type=int, default=0, help="0=전 종목")
    sb.add_argument("--positions", type=int, default=5)
    sb.add_argument("--cost", type=float, default=0.005)
    sb.add_argument("--setups", default="", help="쉼표로 구분: pullback,breakout,ma15,ma15_hold (기본 전부)")
    sb.add_argument("--tp", type=float, default=None, help="고정 익절 (예 0.15)")
    sb.add_argument("--sl", type=float, default=None, help="고정 손절 (예 0.05)")
    sb.add_argument("--max-hold", type=int, default=60, help="고정 모드 최대 보유 거래일")
    sb.set_defaults(func=cmd_swing_backtest)
    lk = sub.add_parser("lookup", help="종목 이름으로 찾아 평가")
    lk.add_argument("query")
    lk.add_argument("--max", type=int, default=5)
    lk.set_defaults(func=cmd_lookup)
    dv = sub.add_parser("dividend", help="배당주 순위")
    dv.add_argument("--universe", type=int, default=200)
    dv.add_argument("--min-yield", type=float, default=2.5)
    dv.add_argument("--top", type=int, default=20)
    dv.add_argument("--weekly", type=int, default=50000)
    dv.set_defaults(func=cmd_dividend)
    dc = sub.add_parser("dca", help="매주 적립식 시뮬레이션")
    dc.add_argument("--amount", type=float, default=50000)
    dc.add_argument("--years", type=float, default=3)
    dc.set_defaults(func=cmd_dca)
    args = p.parse_args()
    print(f"데이터 소스: {data.source_name()}")
    args.func(args)


if __name__ == "__main__":
    main()
