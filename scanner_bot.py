import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

# ================= 環境變數與設定 =================
LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://finmindtrade.com"
TWSE_T86_URL = "https://twse.com.tw"
STREAMLIT_APP_URL = "https://streamlit.app"
LINE_PUSH_URL = "https://line.me"
TW_TZ = timezone(timedelta(hours=8))

# 鎖定電子、半導體、AI 相關產業類別
TECH_INDUSTRIES = {
    "半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
    "通信網路業", "電子通路業", "資訊服務業", "其他電子業"
}


# ================= 1. 取得電子/AI/半導體股票清單 =================
def get_tech_stock_list():
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
            & (df["industry_category"].isin(TECH_INDUSTRIES))
        ]
        df = df.drop_duplicates(subset=["stock_id"], keep="last")

        return {
            r["stock_id"] + (".TW" if r["type"] == "twse" else ".TWO"): {
                "id": r["stock_id"],
                "name": r["stock_name"],
                "industry": r["industry_category"],
            }
            for _, r in df.iterrows()
        }
    except Exception as e:
        print("取得股票清單失敗:", e)
        return {}


# ================= 2. 技術指標計算工具 =================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))


def calculate_kd(df, period=9):
    low_min = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


def check_fib_touch(curr_c, curr_l, h60, l60, ma60, tol=0.025):
    swing = h60 - l60
    if swing <= 0:
        return None, 0.0, False
    for k, ratio in [("0.382", 0.382), ("0.500", 0.500), ("0.618", 0.618)]:
        v = h60 - swing * ratio
        if abs(curr_c - v) / v <= tol or abs(curr_l - v) / v <= tol:
            is_confluence = abs(v - ma60) / ma60 <= 0.025
            return k, v, is_confluence
    return None, 0.0, False


# ================= 3. 大盤多空狀態與法人籌碼 =================
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
        rsi_s = calculate_rsi(close, 14)
        rsi = float(rsi_s.iloc[-1])
        return {
            "close": curr_c,
            "pct": ((curr_c - prev_c) / prev_c) * 100,
            "ma60": ma60,
            "rsi": rsi,
            "is_bull": curr_c > ma60,
        }
    except Exception:
        return None


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


def get_top5_tech_institutional(stock_map):
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
        return [], []
    items = [{"id": k, "name": id_to_name[k], "net": int(round(v))} for k, v in net_dict.items()]
    top_buy = sorted([x for x in items if x["net"] > 0], key=lambda x: x["net"], reverse=True)[:5]
    top_sell = sorted([x for x in items if x["net"] < 0], key=lambda x: x["net"])[:5]
    return top_buy, top_sell
# ================= 4. 單股 2 年歷史回測引擎 (含跳空修正) =================
def run_quick_backtest(df):
    if len(df) < 120:
        return 0, 0.0, 0.0

    close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    vol_ma5 = vol.rolling(5).mean()
    rsi = calculate_rsi(close, 14)
    k, d = calculate_kd(df, 9)

    trades = []
    in_pos, entry_p, stop_p, target_p = False, 0.0, 0.0, 0.0

    for i in range(65, len(df) - 1):
        c, o, l, h = float(close.iloc[i]), float(open_p.iloc[i]), float(low.iloc[i]), float(high.iloc[i])
        v, v_prev, v_ma5 = float(vol.iloc[i]), float(vol.iloc[i - 1]), float(vol_ma5.iloc[i - 1])
        m20, m60 = float(ma20.iloc[i]), float(ma60.iloc[i])
        m60_prev = float(ma60.iloc[i - 5])
        r = float(rsi.iloc[i])
        k_now, d_now = float(k.iloc[i]), float(d.iloc[i])
        k_prev, d_prev = float(k.iloc[i - 1]), float(d.iloc[i - 1])
        h20 = float(high.iloc[i - 20 : i].max())
        h60, l60 = float(high.iloc[i - 60 : i].max()), float(low.iloc[i - 60 : i].min())
        swing = h60 - l60

        if not in_pos:
            is_kd_gold = (k_prev <= d_prev) and (k_now > d_now) and (k_now < 45)
            is_smart_money = (v_prev < v_ma5) and (c > o) and (v > v_prev * 1.2)
            if (
                c > m60 and m60 > m60_prev
                and ((h20 - c) / h20) >= 0.03
                and (38 <= r <= 52)
                and is_kd_gold and is_smart_money and swing > 0
            ):
                fibs = [h60 - swing * ratio for ratio in (0.382, 0.500, 0.618)]
                if any(abs(c - f) / f <= 0.025 or abs(l - f) / f <= 0.025 for f in fibs):
                    struct_stop = max(m60, h60 - swing * 0.786)
                    if (c - struct_stop) / c <= 0.065:
                        in_pos = True
                        entry_p = c
                        stop_p = max(struct_stop, c * 0.94)
                        target_p = m20
        else:
            if l <= stop_p:
                exit_p = min(o, stop_p)
                trades.append((exit_p - entry_p) / entry_p * 100)
                in_pos = False
            elif h >= target_p:
                exit_p = max(o, target_p)
                trades.append((exit_p - entry_p) / entry_p * 100)
                in_pos = False

    if not trades:
        return 0, 0.0, 0.0
    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades) * 100
    avg_ret = sum(trades) / len(trades)
    return len(trades), win_rate, avg_ret


# ================= 5. 核心選股掃描器 (篩出前 10 名) =================
def run_screener(stock_map, is_bull_market=True, is_intraday=False):
    tickers = list(stock_map.keys())
    if not tickers:
        return []
    passed = []

    for i in range(0, len(tickers), 200):
        batch = tickers[i : i + 200]
        try:
            data = yf.download(batch, period="2y", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].dropna(subset=["Close", "Volume"])
                if len(df) < 66:
                    continue

                close, open_p, low_p, vol = df["Close"], df["Open"], df["Low"], df["Volume"]
                vol_ma5_s = (vol / 1000).rolling(5).mean()
                vol_ma5, vol_ma5_prev = float(vol_ma5_s.iloc[-1]), float(vol_ma5_s.iloc[-2])
                curr_v, prev_v = float(vol.iloc[-1]) / 1000, float(vol.iloc[-2]) / 1000

                curr_c = float(close.iloc[-1])
                if vol_ma5 < 1000 or curr_c < 20:
                    continue

                ma5 = float(close.rolling(5).mean().iloc[-1])
                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma60_s = close.rolling(60).mean()
                ma60, ma60_5d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-6])

                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                k_s, d_s = calculate_kd(df, 9)
                k_now, d_now = float(k_s.iloc[-1]), float(d_s.iloc[-1])
                k_prev, d_prev = float(k_s.iloc[-2]), float(d_s.iloc[-2])

                prev_c, curr_o, curr_l = float(close.iloc[-2]), float(open_p.iloc[-1]), float(low_p.iloc[-1])
                h20 = float(df["High"].iloc[-20:].max())
                h60 = float(df["High"].iloc[-60:].max())
                l60 = float(df["Low"].iloc[-60:].min())
                swing = h60 - l60

                struct_stop = max(ma60, h60 - swing * 0.786)
                is_kd_gold = (k_prev <= d_prev) and (k_now > d_now) and (k_now < 45)
                is_smart_money = (prev_v < vol_ma5_prev) and (curr_c > curr_o) and (is_intraday or curr_v > prev_v * 1.2)
                info = stock_map[t]

                if is_bull_market:
                    if curr_c <= ma60 or ma60 <= ma60_5d_ago or ((h20 - curr_c) / h20) * 100 < 3.0:
                        continue
                    if not (38 <= rsi <= 52 and is_kd_gold and is_smart_money):
                        continue
                    if (curr_c - struct_stop) / curr_c > 0.065:
                        continue
                    fib_hit, fib_p, is_conf = check_fib_touch(curr_c, curr_l, h60, l60, ma60, 0.025)
                    if not fib_hit:
                        continue

                    dual_stop = max(struct_stop, curr_c * 0.94)
                    bt_cnt, bt_win, bt_ret = run_quick_backtest(df)

                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "industry": info["industry"],
                        "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100, "rsi": rsi,
                        "fib_hit": fib_hit, "fib_price": fib_p, "is_conf": is_conf,
                        "sell1": ma20, "sell2": h20, "dual_stop": dual_stop,
                        "bt_cnt": bt_cnt, "bt_win": bt_win, "bt_ret": bt_ret,
                        "vol_ratio": curr_v / vol_ma5 if vol_ma5 > 0 else 1.0
                    })
                else:
                    if not (rsi < 28 and rsi > prev_rsi and is_kd_gold):
                        continue
                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "industry": info["industry"],
                        "close": curr_c, "pct": ((curr_c - prev_c) / prev_c) * 100, "rsi": rsi,
                        "fib_hit": "超賣底", "fib_price": curr_c, "is_conf": False,
                        "sell1": ma5, "sell2": ma20, "dual_stop": curr_l * 0.97,
                        "bt_cnt": 0, "bt_win": 0.0, "bt_ret": 0.0,
                        "vol_ratio": curr_v / vol_ma5 if vol_ma5 > 0 else 1.0
                    })
        except Exception:
            continue

    passed.sort(key=lambda x: (not x["is_conf"], -x["bt_win"], x["rsi"]))
    candidates = passed[:20]

    for item in candidates:
        item["inst_5d"] = check_inst_buy_5d(item["stock_id"])
        score = 0
        if item["is_conf"]:
            score += 30
        if item["inst_5d"] > 0:
            score += min(25, item["inst_5d"] / 100)
        if item["bt_cnt"] >= 2 and item["bt_win"] >= 60:
            score += 25
        score += min(20, item["vol_ratio"] * 8)
        item["score"] = round(score, 1)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]
# ================= 6. 建構 LINE Flex Message 卡片 =================
def build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str):
    is_bull = twii["is_bull"] if twii else True
    header_bg = "#1E3A8A" if is_bull else "#1B5E20"
    mode_tag = "⚡ 盤中電子AI預警" if is_intraday else "📊 盤後電子AI精選 TOP 10"
    regime_text = "多頭順風 (站上季線)" if is_bull else "空頭防守 (跌破季線)"

    header_contents = [
        {"type": "text", "text": f"{mode_tag} • {now_str}", "color": "#93C5FD", "size": "xs", "weight": "bold"},
        {"type": "text", "text": f"大盤：{regime_text}", "color": "#FFFFFF", "size": "lg", "weight": "bold", "margin": "xs"},
    ]
    if twii:
        sign = "+" if twii["pct"] >= 0 else ""
        header_contents.append({
            "type": "text",
            "text": f"加權 {twii['close']:,.0f} ({sign}{twii['pct']:.2f}%) | RSI {twii['rsi']:.1f} | 季線 {twii['ma60']:,.0f}",
            "color": "#E2E8F0", "size": "xs", "margin": "sm",
        })

    body_contents = []
    if not picks:
        body_contents.append({
            "type": "box", "layout": "vertical", "paddingAll": "12px", "backgroundColor": "#F8FAFC", "cornerRadius": "8px",
            "contents": [{"type": "text", "text": "今日電子/AI族群無符合洗盤出量＋KD金叉之標的，建議保留現金觀望。", "size": "sm", "color": "#64748B", "wrap": True}],
        })
    else:
        for idx, s in enumerate(picks):
            if idx > 0:
                body_contents.append({"type": "separator", "margin": "md"})
            pct_color = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
            pct_sign = "+" if s["pct"] >= 0 else ""
            inst_color = "#D32F2F" if s["inst_5d"] > 0 else ("#2E7D32" if s["inst_5d"] < 0 else "#64748B")
            inst_sign = "+" if s["inst_5d"] > 0 else ""
            
            ind_short = s["industry"].replace("業", "").replace("設備", "")
            badge_str = f"🔥共振({ind_short})" if s["is_conf"] else f"⚡出量({ind_short})"
            bt_str = f"回測{s['bt_cnt']}次勝率{s['bt_win']:.0f}%" if s["bt_cnt"] > 0 else "首現訊號"

            body_contents.append({
                "type": "box", "layout": "vertical", "margin": "md" if idx > 0 else "none",
                "contents": [
                    {"type": "box", "layout": "horizontal", "contents": [
                        {"type": "text", "text": f"#{idx+1} {s['stock_id']} {s['name']}", "weight": "bold", "size": "sm", "color": "#0F172A", "flex": 5},
                        {"type": "text", "text": badge_str, "size": "xs", "weight": "bold", "color": "#D97706", "align": "end", "flex": 5},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "contents": [
                        {"type": "text", "text": f"{s['close']:.1f} ({pct_sign}{s['pct']:.1f}%)", "size": "sm", "weight": "bold", "color": pct_color, "flex": 4},
                        {"type": "text", "text": f"5日法人:{inst_sign}{s['inst_5d']}張 | {bt_str}", "size": "xxs", "color": inst_color, "align": "end", "flex": 7},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "backgroundColor": "#F1F5F9", "paddingAll": "5px", "cornerRadius": "4px", "contents": [
                        {"type": "text", "text": f"支撐 {s['fib_price']:.1f}", "size": "xxs", "color": "#334155", "flex": 1},
                        {"type": "text", "text": f"停損 {s['dual_stop']:.1f}", "size": "xxs", "color": "#15803D", "weight": "bold", "align": "center", "flex": 1},
                        {"type": "text", "text": f"目標 {s['sell1']:.1f}/{s['sell2']:.1f}", "size": "xxs", "color": "#B91C1C", "align": "end", "flex": 2},
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
                {"type": "text", "text": "🏦 電子/半導體法人當日焦點 (張)", "size": "xs", "weight": "bold", "color": "#334155"},
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
    is_intraday = now_tw.hour < 16
    now_str = now_tw.strftime("%m/%d %H:%M")
    print(f"[{now_str}] 開始執行台股電子/AI/半導體掃描 (盤中模式={is_intraday})...")

    twii = get_twii_status()
    is_bull = twii["is_bull"] if twii else True

    stock_map = get_tech_stock_list()
    print(f"取得電子/AI/半導體有效標的共 {len(stock_map)} 檔")

    picks = run_screener(stock_map, is_bull_market=is_bull, is_intraday=is_intraday)
    print(f"綜合評分篩選出 TOP {len(picks)} 檔標的")

    if is_intraday and not picks:
        print("盤中無符合條件之標的，略過推播。")
        return

    top_buy, top_sell = ([], []) if is_intraday else get_top5_tech_institutional(stock_map)
    flex_msg = build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str)
    push_line_message(flex_msg)


if __name__ == "__main__":
    main()
