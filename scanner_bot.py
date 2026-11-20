import os
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://api." + "finmindtrade" + ".com/api/v4/data"
TWSE_T86_URL = "https://openapi." + "twse" + ".com.tw/v1/fund/T86"
STREAMLIT_APP_URL = "https://james-stockapp" + ".streamlit" + ".app/"

def get_tw_stock_list():
    try:
        res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN}, timeout=15).json()
        if res.get("msg") != "success":
            return {}
        df = pd.DataFrame(res["data"])
        df = df[df["type"].isin(["twse", "tpex"]) & (df["stock_id"].str.len() == 4) & (df["stock_id"].str.isdigit())]
        return {r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {"id": r["stock_id"], "name": r["stock_name"]} for _, r in df.iterrows()}
    except Exception:
        return {}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

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
    params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start, "token": FINMIND_TOKEN}
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10).json()
        if res.get("msg") != "success" or not res.get("data"):
            return 0
        df = pd.DataFrame(res["data"])
        df["net"] = df["buy"] - df["sell"]
        recent = sorted(df["date"].unique())[-5:]
        return df[df["date"].isin(recent)]["net"].sum() / 1000
    except Exception:
        return 0

def get_top5_institutional(stock_map):
    """自動取得今日三大法人合計買超前5名與賣超前5名（只統計4碼一般股票）"""
    id_to_name = {v["id"]: v["name"] for v in stock_map.values()}
    net_dict = {}

    # 1. 優先嘗試從 FinMind 取得全市場最新一日法人買賣超
    for days_back in range(0, 5):
        q_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "start_date": q_date, "end_date": q_date, "token": FINMIND_TOKEN}, timeout=15).json()
            if res.get("msg") == "success" and res.get("data"):
                df = pd.DataFrame(res["data"])
                df = df[df["stock_id"].isin(id_to_name.keys())]
                if not df.empty:
                    df["net_lots"] = (df["buy"] - df["sell"]) / 1000
                    grouped = df.groupby("stock_id")["net_lots"].sum()
                    net_dict = grouped.to_dict()
                    break
        except Exception:
            pass

    # 2. 若 FinMind 無回傳，自動改用證交所 OpenAPI 備援
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
def run_screener(stock_map):
    tickers = list(stock_map.keys())
    if not tickers:
        return []
    passed = []
    for i in range(0, len(tickers), 250):
        batch = tickers[i:i + 250]
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
                ma20, ma60 = close.rolling(20).mean().iloc[-1], close.rolling(60).mean().iloc[-1]
                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                curr_c, prev_c, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(low_p.iloc[-1])
                h20, h60, l60 = float(df["High"].iloc[-20:].max()), float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
                if vol_ma5 < 1000 or curr_c < 15 or curr_c <= ma60 or ((h20 - curr_c) / h20) * 100 < 3.0:
                    continue
                fib_hit, fib_p, is_conf = check_fib_touch(curr_c, curr_l, h60, l60, ma60, 0.015)
                if not fib_hit or rsi >= 48:
                    continue
                swing = h60 - l60
                dual_stop = max(ma60, h60 - swing * 0.786)
                is_rsi_hook = (rsi < 30) and (rsi > prev_rsi)
                info = stock_map[t]
                passed.append({
                    "stock_id": info["id"], "name": info["name"], "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
                    "rsi": rsi, "fib_hit": fib_hit, "fib_price": fib_p, "is_conf": is_conf, "is_rsi_hook": is_rsi_hook,
                    "ma20": ma20, "dual_stop": dual_stop, "high_20d": h20
                })
        except Exception:
            pass
        time.sleep(0.5)

    passed = sorted(passed, key=lambda x: (not x["is_conf"], x["rsi"]))[:40]
    final_list = []
    for item in passed:
        net = check_inst_buy_5d(item["stock_id"])
        if net > 0:
            item["net_5d"] = int(net)
            final_list.append(item)
    if not final_list:
        return []
    return pd.DataFrame(final_list).sort_values(by=["is_conf", "rsi", "net_5d"], ascending=[False, True, False]).head(5).to_dict("records")

def build_flex_message(today_str, top_buy, top_sell, top5):
    body = []

    # 1. 今日三大法人買超 & 賣超 TOP 5 並排雙欄表格
    body.append({"type": "text", "text": "🏦 今日三大法人買賣超 TOP 5（張）", "weight": "bold", "size": "sm", "color": "#1E3A8A"})
    if not top_buy and not top_sell:
        body.append({"type": "text", "text": "今日法人籌碼尚未更新。", "size": "xs", "color": "#666666", "margin": "xs"})
    else:
        body.append({"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
            {"type": "text", "text": "🔥 買超前五名", "size": "xs", "weight": "bold", "color": "#D32F2F", "flex": 1},
            {"type": "text", "text": "❄️ 賣超前五名", "size": "xs", "weight": "bold", "color": "#2E7D32", "flex": 1, "align": "end"},
        ]})
        body.append({"type": "separator", "margin": "xs"})
        for idx in range(5):
            b_txt = f"{idx+1}.{top_buy[idx]['name']} +{top_buy[idx]['net']:,}" if idx < len(top_buy) else "-"
            s_txt = f"{idx+1}.{top_sell[idx]['name']} {top_sell[idx]['net']:,}" if idx < len(top_sell) else "-"
            body.append({"type": "box", "layout": "horizontal", "margin": "xs", "contents": [
                {"type": "text", "text": b_txt, "size": "xxs", "color": "#D32F2F", "flex": 1},
                {"type": "text", "text": s_txt, "size": "xxs", "color": "#2E7D32", "flex": 1, "align": "end"},
            ]})
    body.append({"type": "separator", "margin": "md"})

    # 2. 黃金分割低接精選 TOP 5
    body.append({"type": "text", "text": "📉 黃金分割＋季線共振低接 TOP 5", "weight": "bold", "size": "sm", "color": "#1E3A8A", "margin": "md"})
    if not top5:
        body.append({"type": "text", "text": "今日全市場無同時符合黃金分割支撐與法人買超之標的。", "size": "xs", "color": "#666666", "wrap": True, "margin": "sm"})
    else:
        body.append({"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
            {"type": "text", "text": "標的/法人", "size": "xs", "color": "#888888", "flex": 3},
            {"type": "text", "text": "現價/支撐共振", "size": "xs", "color": "#888888", "flex": 4, "align": "center"},
            {"type": "text", "text": "賣點/雙重防守", "size": "xs", "color": "#888888", "flex": 4, "align": "end"},
        ]})
        body.append({"type": "separator", "margin": "xs"})
        for s in top5:
            c = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
            tags = ("🔥共振" if s["is_conf"] else "") + ("🟢止跌" if s["is_rsi_hook"] else "")
            body.append({"type": "box", "layout": "horizontal", "margin": "sm", "alignItems": "center", "contents": [
                {"type": "box", "layout": "vertical", "flex": 3, "contents": [
                    {"type": "text", "text": f"{s['name']}{tags}", "size": "xs", "weight": "bold"},
                    {"type": "text", "text": f"{s['stock_id']}(+{s['net_5d']}張)", "size": "xxs", "color": "#666666"}
                ]},
                {"type": "box", "layout": "vertical", "flex": 4, "contents": [
                    {"type": "text", "text": f"{s['close']:.1f}({s['pct']:+.1f}%)", "size": "xs", "weight": "bold", "color": c, "align": "center"},
                    {"type": "text", "text": f"Fib {s['fib_hit']}({s['fib_price']:.1f})", "size": "xxs", "color": "#1D4ED8", "align": "center"}
                ]},
                {"type": "box", "layout": "vertical", "flex": 4, "contents": [
                    {"type": "text", "text": f"賣:{s['ma20']:.1f}/{s['high_20d']:.1f}", "size": "xxs", "color": "#D32F2F", "align": "end"},
                    {"type": "text", "text": f"守:{s['dual_stop']:.1f}", "size": "xxs", "color": "#2E7D32", "align": "end"}
                ]}
            ]})
            body.append({"type": "separator", "margin": "xs", "color": "#F0F0F0"})

    return {
        "type": "flex", "altText": f"📊 台股 AI 法人籌碼與波段選股 ({today_str})",
        "contents": {
            "type": "bubble", "size": "giga",
            "header": {"type": "box", "layout": "vertical", "backgroundColor": "#1E293B", "paddingAll": "14px", "contents": [
                {"type": "text", "text": "📊 台股 AI 法人籌碼與波段選股", "color": "#FFFFFF", "weight": "bold", "size": "md"},
                {"type": "text", "text": f"資料日期：{today_str} | 法人買賣超＋黃金分割選股", "color": "#94A3B8", "size": "xs", "margin": "xs"}
            ]},
            "body": {"type": "box", "layout": "vertical", "paddingAll": "12px", "contents": body},
            "footer": {"type": "box", "layout": "vertical", "paddingAll": "10px", "contents": [
                {"type": "button", "style": "primary", "color": "#2563EB", "height": "sm",
                 "action": {"type": "uri", "label": "🌐 開啟互動看盤與持股診斷", "uri": STREAMLIT_APP_URL}}
            ]}
        }
    }

def send_line_flex(flex_msg):
    if not LINE_TOKEN or not LINE_USER_ID:
        return
    url = "https://api." + "line" + ".me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"}
    res = requests.post(url, headers=headers, json={"to": LINE_USER_ID, "messages": [flex_msg]})
    print("LINE Flex Push Status:", res.status_code, res.text)

if __name__ == "__main__":
    today_str = datetime.now().strftime("%Y-%m-%d")
    stock_map = get_tw_stock_list()
    top_buy, top_sell = get_top5_institutional(stock_map)
    send_line_flex(build_flex_message(today_str, top_buy, top_sell, run_screener(stock_map)))
