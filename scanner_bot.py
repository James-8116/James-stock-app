import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"
TWSE_T86_URL = "https://openapi." + "twse" + ".com.tw/v1/fund/T86"
STREAMLIT_APP_URL = "https://james-stockapp" + ".streamlit" + ".app/"
LINE_PUSH_URL = "https://api." + "line" + ".me/v2/bot/message/push"
TW_TZ = timezone(timedelta(hours=8))


def get_tw_stock_list():
    try:
        res = requests.get(
            FINMIND_URL,
            params={"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN},
            timeout=15,
        ).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        latest_date = pd.to_datetime(df["date"]).max()
        cutoff = (latest_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

        df = df[
            df["type"].isin(["twse", "tpex"])
            & (df["stock_id"].str.len() == 4)
            & (df["stock_id"].str.isdigit())
            & (df["date"] >= cutoff)
        ]
        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {
                "id": r["stock_id"],
                "name": r["stock_name"],
            }
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))


def get_twii_status():
    try:
        df = yf.download("^TWII", period="18mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            return None
        close = df["Close"]
        curr_c, prev_c = float(close.iloc[-1]), float(close.iloc[-2])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma240 = float(close.rolling(240).mean().iloc[-1]) if len(close) >= 240 else float(close.mean())
        rsi_s = calculate_rsi(close, 14)
        rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
        sub = df.iloc[-60:]
        h60, l60 = float(sub["High"].max()), float(sub["Low"].min())
        swing = h60 - l60
        return {
            "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
            "ma60": ma60, "ma240": ma240, "rsi": rsi, "prev_rsi": prev_rsi,
            "is_bull": curr_c > ma60,
            "fib500": h60 - swing * 0.500,
            "fib618": h60 - swing * 0.618,
            "ext1272": max(0.0, h60 - swing * 1.272),
        }
    except Exception:
        return None
def check_fib_touch(curr_c, curr_l, h60, l60, ma60, tol=0.015):
    swing = h60 - l60
    if swing <= 0:
        return None, 0.0, False
    for k, ratio in [("0.382", 0.382), ("0.500", 0.500), ("0.618", 0.618)]:
        v = h60 - swing * ratio
        if abs(curr_c - v) / v <= tol or abs(curr_l - v) / v <= tol:
            is_confluence = abs(v - ma60) / ma60 <= 0.025
            return k, v, is_confluence
    return None, 0.0, False


def check_inst_buy_5d(stock_id):
    start = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
        "token": FINMIND_TOKEN,
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10).json()
        if res.get("msg") != "success" or not res.get("data"):
            return 0
        df = pd.DataFrame(res["data"])
        df["net"] = df["buy"] - df["sell"]
        recent = sorted(df["date"].unique())[-5:]
        return int(round(df[df["date"].isin(recent)]["net"].sum() / 1000))
    except Exception:
        return 0


def get_top5_institutional(stock_map):
    id_to_name = {v["id"]: v["name"] for v in stock_map.values()}
    net_dict = {}
    for days_back in range(0, 5):
        q_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            res = requests.get(
                FINMIND_URL,
                params={
                    "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                    "start_date": q_date,
                    "end_date": q_date,
                    "token": FINMIND_TOKEN,
                },
                timeout=15,
            ).json()
            if res.get("msg") == "success" and res.get("data"):
                df = pd.DataFrame(res["data"])
                df = df[df["stock_id"].isin(id_to_name.keys())]
                if not df.empty:
                    df["net_lots"] = (df["buy"] - df["sell"]) / 1000
                    net_dict = df.groupby("stock_id")["net_lots"].sum().to_dict()
                    break
        except Exception:
            pass
    if not net_dict:
        try:
            rows = requests.get(TWSE_T86_URL, timeout=15).json()
            for r in rows:
                sid = str(r.get("Code", "")).strip()
                if sid in id_to_name:
                    val = str(r.get("TotalInstitutionalInvestorsNetBuySell", "0")).replace(",", "")
                    net_dict[sid] = float(val) / 1000
        except Exception:
            pass
    if not net_dict:
        return [], []
    items = [{"id": k, "name": id_to_name[k], "net": int(round(v))} for k, v in net_dict.items()]
    top_buy = sorted([x for x in items if x["net"] > 0], key=lambda x: x["net"], reverse=True)[:5]
    top_sell = sorted([x for x in items if x["net"] < 0], key=lambda x: x["net"])[:5]
    return top_buy, top_sell
def run_screener(stock_map, is_bull_market=True, is_intraday=False):
    tickers = list(stock_map.keys())
    if not tickers:
        return []
    passed = []
    for i in range(0, len(tickers), 250):
        batch = tickers[i : i + 250]
        try:
            data = yf.download(batch, period="4mo", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].dropna(subset=["Close", "Volume"])
                if len(df) < 62:
                    continue
                close, low_p = df["Close"], df["Low"]
                vol_ma5 = (df["Volume"] / 1000).rolling(5).mean().iloc[-1]
                ma5 = float(close.rolling(5).mean().iloc[-1])
                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma60 = float(close.rolling(60).mean().iloc[-1])
                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                curr_c, prev_c, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(low_p.iloc[-1])
                h20 = float(df["High"].iloc[-20:].max())
                h60 = float(df["High"].iloc[-60:].max())
                l60 = float(df["Low"].iloc[-60:].min())
                if vol_ma5 < 1000 or curr_c < 15:
                    continue
                swing = h60 - l60
                dual_stop = max(ma60, h60 - swing * 0.786)
                is_rsi_hook = (rsi < 30) and (rsi > prev_rsi)
                info = stock_map[t]

                if is_bull_market:
                    if curr_c <= ma60 or ((h20 - curr_c) / h20) * 100 < 3.0:
                        continue
                    fib_hit, fib_p, is_conf = check_fib_touch(curr_c, curr_l, h60, l60, ma60, 0.015)
                    if not fib_hit or rsi >= 48:
                        continue
                    if is_intraday and not (curr_c <= fib_p * 1.005 and (is_conf or is_rsi_hook)):
                        continue
                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "close": curr_c,
                        "pct": ((curr_c - prev_c) / prev_c) * 100, "rsi": rsi,
                        "fib_hit": fib_hit, "fib_price": fib_p, "is_conf": is_conf,
                        "is_rsi_hook": is_rsi_hook, "sell1": ma20, "sell2": h20, "dual_stop": dual_stop,
                    })
                else:
                    if not (rsi < 28 and rsi > prev_rsi):
                        continue
                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "close": curr_c,
                        "pct": ((curr_c - prev_c) / prev_c) * 100, "rsi": rsi,
                        "fib_hit": "超賣底", "fib_price": curr_c, "is_conf": False,
                        "is_rsi_hook": True, "sell1": ma5, "sell2": ma20, "dual_stop": curr_l * 0.97,
                    })
        except Exception:
            continue

    passed.sort(key=lambda x: (not x["is_conf"], not x["is_rsi_hook"], x["rsi"]))
    top_picks = passed[:8]
    for item in top_picks:
        item["inst_5d"] = check_inst_buy_5d(item["stock_id"])
    return top_picks
def build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str):
    is_bull = twii["is_bull"] if twii else True
    header_bg = "#B71C1C" if is_bull else "#1B5E20"
    mode_tag = "⚡ 盤中即時預警" if is_intraday else "📊 盤後完整總結"
    regime_text = "多頭順風 (站上季線)" if is_bull else "空頭防守 (跌破季線)"

    header_contents = [
        {"type": "text", "text": f"{mode_tag} • {now_str}", "color": "#FFCDD2" if is_bull else "#C8E6C9", "size": "xs", "weight": "bold"},
        {"type": "text", "text": f"大盤風向：{regime_text}", "color": "#FFFFFF", "size": "lg", "weight": "bold", "margin": "xs"},
    ]
    if twii:
        sign = "+" if twii["pct"] >= 0 else ""
        header_contents.append({
            "type": "text",
            "text": f"加權 {twii['close']:,.0f} ({sign}{twii['pct']:.2f}%) | RSI {twii['rsi']:.1f} | 季線 {twii['ma60']:,.0f}",
            "color": "#FFFFFF", "size": "xs", "margin": "sm",
        })

    body_contents = []
    if not picks:
        body_contents.append({
            "type": "box", "layout": "vertical", "paddingAll": "12px", "backgroundColor": "#F5F5F5", "cornerRadius": "8px",
            "contents": [{"type": "text", "text": "今日無符合嚴格回測條件之標的，建議耐心觀望。", "size": "sm", "color": "#666666", "wrap": True}],
        })
    else:
        for idx, s in enumerate(picks):
            if idx > 0:
                body_contents.append({"type": "separator", "margin": "md"})
            pct_color = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
            pct_sign = "+" if s["pct"] >= 0 else ""
            inst_color = "#D32F2F" if s["inst_5d"] > 0 else ("#2E7D32" if s["inst_5d"] < 0 else "#757575")
            inst_sign = "+" if s["inst_5d"] > 0 else ""
            badges = [b for b, cond in [("🔥均線共振", s["is_conf"]), ("🪝RSI勾頭", s["is_rsi_hook"])] if cond]
            badge_str = " ".join(badges) if badges else f"Fib {s['fib_hit']}"

            body_contents.append({
                "type": "box", "layout": "vertical", "margin": "md" if idx > 0 else "none",
                "contents": [
                    {"type": "box", "layout": "horizontal", "contents": [
                        {"type": "text", "text": f"{s['stock_id']} {s['name']}", "weight": "bold", "size": "md", "color": "#111111", "flex": 5},
                        {"type": "text", "text": badge_str, "size": "xs", "weight": "bold", "color": "#E65100", "align": "end", "flex": 4},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "contents": [
                        {"type": "text", "text": f"現價 {s['close']:.2f} ({pct_sign}{s['pct']:.1f}%)", "size": "sm", "weight": "bold", "color": pct_color, "flex": 5},
                        {"type": "text", "text": f"RSI {s['rsi']:.1f} | 5日法人 {inst_sign}{s['inst_5d']}張", "size": "xs", "color": inst_color, "align": "end", "flex": 6},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "backgroundColor": "#F8F9FA", "paddingAll": "6px", "cornerRadius": "6px", "contents": [
                        {"type": "text", "text": f"支撐 {s['fib_price']:.1f}", "size": "xs", "color": "#333333", "flex": 1},
                        {"type": "text", "text": f"停損 {s['dual_stop']:.1f}", "size": "xs", "color": "#2E7D32", "weight": "bold", "align": "center", "flex": 1},
                        {"type": "text", "text": f"目標 {s['sell1']:.1f}/{s['sell2']:.1f}", "size": "xs", "color": "#D32F2F", "align": "end", "flex": 2},
                    ]},
                ],
            })

    if not is_intraday and (top_buy or top_sell):
        body_contents.append({"type": "separator", "margin": "lg"})
        buy_str = "、".join([f"{x['name']}(+{x['net']})" for x in top_buy[:3]]) if top_buy else "無"
        sell_str = "、".join([f"{x['name']}({x['net']})" for x in top_sell[:3]]) if top_sell else "無"
        body_contents.append({
            "type": "box", "layout": "vertical", "margin": "md",
            "contents": [
                {"type": "text", "text": "🏦 三大法人當日焦點 (張)", "size": "xs", "weight": "bold", "color": "#444444"},
                {"type": "text", "text": f"買超：{buy_str}", "size": "xs", "color": "#D32F2F", "wrap": True, "margin": "xs"},
                {"type": "text", "text": f"賣超：{sell_str}", "size": "xs", "color": "#2E7D32", "wrap": True, "margin": "xs"},
            ],
        })

    bubble = {
        "type": "bubble", "size": "mega",
        "header": {"type": "box", "layout": "vertical", "backgroundColor": header_bg, "paddingAll": "14px", "contents": header_contents},
        "body": {"type": "box", "layout": "vertical", "paddingAll": "14px", "contents": body_contents},
        "footer": {"type": "box", "layout": "vertical", "paddingAll": "10px", "contents": [
            {"type": "button", "style": "primary", "color": "#1E3A8A", "height": "sm", "action": {"type": "uri", "label": "開啟 Streamlit 互動看盤儀表板", "uri": STREAMLIT_APP_URL}}
        ]},
    }
    return {"type": "flex", "altText": f"{mode_tag}：共篩選出 {len(picks)} 檔潛力標的", "contents": bubble}


def push_line_message(flex_msg):
    if not LINE_TOKEN or not LINE_USER_ID:
        print("Missing LINE_TOKEN or LINE_USER_ID")
        return
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    res = requests.post(LINE_PUSH_URL, headers=headers, json={"to": LINE_USER_ID, "messages": [flex_msg]}, timeout=15)
    print("LINE Push Status:", res.status_code, res.text)


def main():
    now_tw = datetime.now(TW_TZ)
    is_intraday = now_tw.hour < 14
    now_str = now_tw.strftime("%m/%d %H:%M")
    print(f"[{now_str}] 開始執行台股掃描 (盤中模式={is_intraday})...")
    twii = get_twii_status()
    is_bull = twii["is_bull"] if twii else True
    stock_map = get_tw_stock_list()
    print(f"取得有效上市櫃標的共 {len(stock_map)} 檔")
    picks = run_screener(stock_map, is_bull_market=is_bull, is_intraday=is_intraday)
    print(f"篩選出 {len(picks)} 檔標的")
    top_buy, top_sell = ([], []) if is_intraday else get_top5_institutional(stock_map)
    flex_msg = build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str)
    push_line_message(flex_msg)


if __name__ == "__main__":
    main()
