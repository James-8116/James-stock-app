import os
import time
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
TW_TZ = timezone(timedelta(hours=8))

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
            "ext1272": max(0.0, h60 - swing * 1.272)
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
    id_to_name = {v["id"]: v["name"] for v in stock_map.values()}
    net_dict = {}
    for days_back in range(0, 5):
        q_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            res = requests.get(FINMIND_URL, params={"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "start_date": q_date, "end_date": q_date, "token": FINMIND_TOKEN}, timeout=15).json()
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
                ma5 = float(close.rolling(5).mean().iloc[-1])
                ma20, ma60 = float(close.rolling(20).mean().iloc[-1]), float(close.rolling(60).mean().iloc[-1])
                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                curr_c, prev_c, curr_l = float(close.iloc[-1]), float(close.iloc[-2]), float(low_p.iloc[-1])
                h20, h60, l60 = float(df["High"].iloc[-20:].max()), float(df["High"].iloc[-60:].max()), float(df["Low"].iloc[-60:].min())
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
                        "stock_id": info["id"], "name": info["name"], "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
                        "rsi": rsi, "fib_hit": fib_hit, "fib_price": fib_p, "is_conf": is_conf, "is_rsi_hook": is_rsi_hook,
                        "sell1": ma20, "sell2": h20, "dual_stop": dual_stop
                    })
                else:
                    # 空頭市場模式：改抓 RSI < 28 超賣且開始止跌勾頭的短線反彈股
                    if not (rsi < 28 and rsi > prev_rsi):
                        continue
                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100,
                        "rsi": rsi, "fib_hit": "超賣底", "fib_price": curr_c, "is_conf": False, "is_rsi_hook": True,
                        "sell1": ma5, "sell2": ma20, "dual_stop": curr_c * 0.96
                    })
        except Exception:
            pass
        time.sleep(0.5)

    passed = sorted(passed, key=lambda x: (not x["is_conf"], x["rsi"]))[:40]
    final_list = []
    for item in passed:
        net = check_inst_buy_5d(item["stock_id"])
        if is_bull_market and net <= 0:
            continue
        item["net_5d"] = int(net)
        final_list.append(item)
    if not final_list:
        return []
    return pd.DataFrame(final_list).sort_values(by=["is_conf", "rsi", "net_5d"], ascending=[False, True, False]).head(5).to_dict("records")
def build_flex_message(now_str, is_intraday, twii, top_buy, top_sell, top5):
    body = []
    if twii:
        mood = "🟢季線之上(正常低接)" if twii["is_bull"] else f"🚨跌破季線(縮手/恐慌底線{twii['ext1272']:,.0f})"
        body.append({"type": "text", "text": f"🇹🇼 大盤 {twii['close']:,.0f} ({twii['pct']:+.1f}%) | {mood}", "size": "xs", "weight": "bold", "color": "#1E3A8A" if twii["is_bull"] else "#D32F2F", "wrap": True})
        body.append({"type": "separator", "margin": "sm"})

    if not is_intraday:
        body.append({"type": "text", "text": "🏦 今日三大法人買賣超 TOP 5（張）", "weight": "bold", "size": "sm", "color": "#1E3A8A", "margin": "sm"})
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

    sec_title = "🔔 盤中到價低接警報" if is_intraday else ("📉 黃金分割低接 TOP 5" if (not twii or twii["is_bull"]) else "🟢 空頭超賣止跌反彈觀察 TOP 5")
    body.append({"type": "text", "text": sec_title, "weight": "bold", "size": "sm", "color": "#1E3A8A", "margin": "md"})
    if not top5:
        body.append({"type": "text", "text": "今日無符合低接或超賣止跌條件之標的，建議保留現金觀望。", "size": "xs", "color": "#666666", "wrap": True, "margin": "sm"})
    else:
        for s in top5:
            c = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
            tags = ("🔥共振" if s["is_conf"] else "") + ("🟢止跌" if s["is_rsi_hook"] else "")
            body.append({"type": "box", "layout": "horizontal", "margin": "sm", "alignItems": "center", "contents": [
                {"type": "box", "layout": "vertical", "flex": 3, "contents": [
                    {"type": "text", "text": f"{s['name']}{tags}", "size": "xs", "weight": "bold"},
                    {"type": "text", "text": f"{s['stock_id']}({s['net_5d']:+}張)", "size": "xxs", "color": "#666666"}
                ]},
                {"type": "box", "layout": "vertical", "flex": 4, "contents": [
                    {"type": "text", "text": f"{s['close']:.1f}({s['pct']:+.1f}%)", "size": "xs", "weight": "bold", "color": c, "align": "center"},
                    {"type": "text", "text": f"買:{s['fib_price']*0.99:.1f}~{s['fib_price']:.1f}", "size": "xxs", "color": "#1D4ED8", "align": "center"}
                ]},
                {"type": "box", "layout": "vertical", "flex": 4, "contents": [
                    {"type": "text", "text": f"賣:{s['sell1']:.1f}/{s['sell2']:.1f}", "size": "xxs", "color": "#D32F2F", "align": "end"},
                    {"type": "text", "text": f"守:{s['dual_stop']:.1f}", "size": "xxs", "color": "#2E7D32", "align": "end"}
                ]}
            ]})
            body.append({"type": "separator", "margin": "xs", "color": "#F0F0F0"})

    title_txt = "🔔 台股盤中低接達標警報" if is_intraday else "📊 台股 AI 盤後籌碼與波段總結"
    return {
        "type": "flex", "altText": f"{title_txt} ({now_str})",
        "contents": {
            "type": "bubble", "size": "giga",
            "header": {"type": "box", "layout": "vertical", "backgroundColor": "#991B1B" if is_intraday else "#1E293B", "paddingAll": "14px", "contents": [
                {"type": "text", "text": title_txt, "color": "#FFFFFF", "weight": "bold", "size": "md"},
                {"type": "text", "text": f"時間：{now_str} (台灣時間)", "color": "#CBD5E1", "size": "xs", "margin": "xs"}
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
    now_tw = datetime.now(TW_TZ)
    now_str = now_tw.strftime("%Y-%m-%d %H:%M")
    # 台灣時間 15:00 之前執行視為「盤中警報模式」，15:00 之後視為「盤後完整總結模式」
    is_intraday = now_tw.hour < 15
    stock_map = get_tw_stock_list()
    twii = get_twii_status()
    is_bull = twii["is_bull"] if twii else True
    top5 = run_screener(stock_map, is_bull_market=is_bull, is_intraday=is_intraday)

    if is_intraday:
        # 盤中模式：有掃出達標買點才發送推播，沒事不吵你
        if top5:
            send_line_flex(build_flex_message(now_str, True, twii, [], [], top5))
        else:
            print("盤中無符合低接買點標的，本次略過推播以節省額度。")
    else:
        top_buy, top_sell = get_top5_institutional(stock_map)
        send_line_flex(build_flex_message(now_str, False, twii, top_buy, top_sell, top5))
