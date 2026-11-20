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
STREAMLIT_APP_URL = "https://james-stockapp" + ".streamlit" + ".app/"

MY_HOLDINGS = {
    "2891": {"name": "中信金", "cost": 71.4},
    "5876": {"name": "上海商銀", "cost": 50.9},
}

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

def check_fib_touch(curr_c, curr_l, h60, l60, tol=0.015):
    swing = h60 - l60
    if swing <= 0:
        return None, 0.0
    for k, ratio in [("0.382", 0.382), ("0.500", 0.500), ("0.618", 0.618)]:
        v = h60 - swing * ratio
        if abs(curr_c - v) / v <= tol or abs(curr_l - v) / v <= tol:
            return k, v
    return None, 0.0

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

def monitor_holdings(stock_map):
    if not MY_HOLDINGS:
        return []
    id_to_ticker = {v["id"]: k for k, v in stock_map.items()}
    results = []
    for stock_id, info in MY_HOLDINGS.items():
        ticker = id_to_ticker.get(stock_id, stock_id + ".TW")
        cost, name = float(info["cost"]), info["name"]
        try:
            df = yf.download(ticker, period="4mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 60:
                continue
            close = df["Close"]
            curr_c = float(close.iloc[-1])
            ma5, ma20, ma60 = float(close.rolling(5).mean().iloc[-1]), float(close.rolling(20).mean().iloc[-1]), float(close.rolling(60).mean().iloc[-1])
            rsi = float(calculate_rsi(close, 14).iloc[-1])
            h20 = float(df["High"].iloc[-20:].max())
            h60, l60 = float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
            swing = h60 - l60

            # 黃金賣點與雙重防守停損 (取 季線 與 黃金0.786 較高者)
            fib_res_618 = l60 + swing * 0.618
            fib_ext_1272 = l60 + swing * 1.272
            fib_stop_786 = h60 - swing * 0.786
            dual_stop = max(ma60, fib_stop_786)
            stop_tag = "季線" if ma60 >= fib_stop_786 else "Fib0.786"
            pnl = ((curr_c - cost) / cost) * 100

            sigs = []
            if rsi >= 75: sigs.append("🔥RSI過熱高賣")
            if curr_c >= fib_ext_1272 * 0.985 and pnl > 5: sigs.append(f"🚀達黃金1.272({fib_ext_1272:.1f})")
            elif abs(curr_c - fib_res_618) / fib_res_618 <= 0.015 and pnl > 2: sigs.append(f"📐達黃金0.618({fib_res_618:.1f})")
            elif curr_c >= h20 * 0.985 and pnl > 2: sigs.append(f"🎯達前高({h20:.1f})")
            if pnl >= 3.0 and curr_c < ma5: sigs.append("📉破5日線停利")
            if curr_c < dual_stop or pnl <= -7.0: sigs.append(f"🚨破{stop_tag}({dual_stop:.1f})停損")

            status = " / ".join(sigs) if sigs else f"✅續抱(賣:{fib_res_618:.1f}/守:{dual_stop:.1f})"
            results.append({"id": stock_id, "name": name, "cost": cost, "close": curr_c, "pnl": pnl, "status": status, "alert": len(sigs) > 0})
        except Exception:
            pass
    return results
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
                rsi = float(calculate_rsi(close, 14).iloc[-1])
                curr_c, prev_c, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(low_p.iloc[-1])
                h20, h60, l60 = float(df["High"].iloc[-20:].max()), float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
                if vol_ma5 < 1000 or curr_c < 15 or curr_c <= ma60 or ((h20 - curr_c) / h20) * 100 < 3.0:
                    continue
                fib_hit, fib_p = check_fib_touch(curr_c, curr_l, h60, l60, 0.015)
                if not fib_hit or rsi >= 48:
                    continue
                swing = h60 - l60
                dual_stop = max(ma60, h60 - swing * 0.786)
                info = stock_map[t]
                passed.append({"stock_id": info["id"], "name": info["name"], "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100, "rsi": rsi, "fib_hit": fib_hit, "fib_price": fib_p, "ma20": ma20, "dual_stop": dual_stop, "high_20d": h20})
        except Exception:
            pass
        time.sleep(0.5)

    passed = sorted(passed, key=lambda x: x["rsi"])[:40]
    final_list = []
    for item in passed:
        net = check_inst_buy_5d(item["stock_id"])
        if net > 0:
            item["net_5d"] = int(net)
            final_list.append(item)
    if not final_list:
        return []
    return pd.DataFrame(final_list).sort_values(by=["rsi", "net_5d"], ascending=[True, False]).head(5).to_dict("records")

def build_flex_message(today_str, holdings, top5):
    body = []
    if holdings:
        body.append({"type": "text", "text": "🔔 持股黃金高賣與雙重防守監控", "weight": "bold", "size": "sm", "color": "#1E3A8A"})
        body.append({"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
            {"type": "text", "text": "標的", "size": "xs", "color": "#888888", "flex": 3},
            {"type": "text", "text": "成本➔現價", "size": "xs", "color": "#888888", "flex": 4, "align": "center"},
            {"type": "text", "text": "損益/黃金攻守", "size": "xs", "color": "#888888", "flex": 4, "align": "end"},
        ]})
        body.append({"type": "separator", "margin": "xs"})
        for h in holdings:
            c = "#D32F2F" if h["pnl"] >= 0 else "#2E7D32"
            body.append({"type": "box", "layout": "horizontal", "margin": "sm", "alignItems": "center", "contents": [
                {"type": "box", "layout": "vertical", "flex": 3, "contents": [
                    {"type": "text", "text": h["name"], "size": "xs", "weight": "bold"},
                    {"type": "text", "text": h["id"], "size": "xxs", "color": "#888888"}
                ]},
                {"type": "text", "text": f"{h['cost']:.1f}➔{h['close']:.1f}", "size": "xs", "flex": 4, "align": "center"},
                {"type": "box", "layout": "vertical", "flex": 4, "contents": [
                    {"type": "text", "text": f"{h['pnl']:+.1f}%", "size": "xs", "weight": "bold", "color": c, "align": "end"},
                    {"type": "text", "text": h["status"], "size": "xxs", "color": "#D32F2F" if h["alert"] else "#555555", "align": "end", "wrap": True}
                ]}
            ]})
        body.append({"type": "separator", "margin": "md"})

    body.append({"type": "text", "text": "📉 黃金分割低接精選 TOP 5", "weight": "bold", "size": "sm", "color": "#1E3A8A", "margin": "md"})
    if not top5:
        body.append({"type": "text", "text": "今日全市場無同時符合黃金分割支撐與法人買超之標的。", "size": "xs", "color": "#666666", "wrap": True, "margin": "sm"})
    else:
        body.append({"type": "box", "layout": "horizontal", "margin": "sm", "contents": [
            {"type": "text", "text": "標的/法人", "size": "xs", "color": "#888888", "flex": 3},
            {"type": "text", "text": "現價/支撐", "size": "xs", "color": "#888888", "flex": 4, "align": "center"},
            {"type": "text", "text": "賣點/雙重防守", "size": "xs", "color": "#888888", "flex": 4, "align": "end"},
        ]})
        body.append({"type": "separator", "margin": "xs"})
        for s in top5:
            c = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
            body.append({"type": "box", "layout": "horizontal", "margin": "sm", "alignItems": "center", "contents": [
                {"type": "box", "layout": "vertical", "flex": 3, "contents": [
                    {"type": "text", "text": s["name"], "size": "xs", "weight": "bold"},
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
        "type": "flex", "altText": f"📊 台股 AI 波段監控與選股 ({today_str})",
        "contents": {
            "type": "bubble", "size": "giga",
            "header": {"type": "box", "layout": "vertical", "backgroundColor": "#1E293B", "paddingAll": "14px", "contents": [
                {"type": "text", "text": "📊 台股 AI 波段監控與選股", "color": "#FFFFFF", "weight": "bold", "size": "md"},
                {"type": "text", "text": f"資料日期：{today_str} | 黃金分割攻守＋季線保護", "color": "#94A3B8", "size": "xs", "margin": "xs"}
            ]},
            "body": {"type": "box", "layout": "vertical", "paddingAll": "12px", "contents": body},
            "footer": {"type": "box", "layout": "vertical", "paddingAll": "10px", "contents": [
                {"type": "button", "style": "primary", "color": "#2563EB", "height": "sm",
                 "action": {"type": "uri", "label": "🌐 開啟互動看盤與走勢圖", "uri": STREAMLIT_APP_URL}}
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
    send_line_flex(build_flex_message(today_str, monitor_holdings(stock_map), run_screener(stock_map)))
