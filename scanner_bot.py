import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

# ================= 環境變數與設定 =================
LINE_TOKEN = os.environ.get("LINE_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

FINMIND_URL = "https://api." + "://finmindtrade.com"
STREAMLIT_APP_URL = "https://james-" + "stockapp.streamlit.app/"
LINE_PUSH_URL = "https://api." + "line.me/v2/bot/message/push"
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


# ================= 2. 技術指標計算工具 (RSI & ATR) =================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))


def calculate_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


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
# ================= 4. V6 精準突破 2 年回測引擎 (含跳空與稅費修正) =================
def run_v6_backtest(df):
    if len(df) < 130:
        return 0, 0.0, 0.0

    close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    vol_ma5 = vol.rolling(5).mean()
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(df, 14)

    trades = []
    in_pos, entry_p, stop_p, highest_p = False, 0.0, 0.0, 0.0
    fee = 0.005

    # 排除最後 1 根（即今日觸發訊號的這根）
    for i in range(122, len(df) - 1):
        c, o, l, h = float(close.iloc[i]), float(open_p.iloc[i]), float(low.iloc[i]), float(high.iloc[i])
        c_prev = float(close.iloc[i - 1])
        v, v_ma5 = float(vol.iloc[i]), float(vol_ma5.iloc[i])
        m20, m60, m120 = float(ma20.iloc[i]), float(ma60.iloc[i]), float(ma120.iloc[i])
        m60_20d_ago = float(ma60.iloc[i - 20])
        r = float(rsi.iloc[i])
        curr_atr = float(atr.iloc[i])

        if pd.isna(curr_atr) or m60_20d_ago <= 0:
            continue

        prev_20d_high = float(close.iloc[i - 20 : i].max())
        box_h = float(high.iloc[i - 5 : i].max())
        box_l = float(low.iloc[i - 5 : i].min())
        box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0

        if not in_pos:
            is_strong_trend = (c > m20 > m60 > m120) and ((m60 - m60_20d_ago) / m60_20d_ago >= 0.025)
            is_tight_box = box_range <= 0.085
            body = c - o
            upper_shadow = h - c
            pct_chg = (c - c_prev) / c_prev
            is_clean_breakout = (
                (c > prev_20d_high)
                and (body > 0)
                and (upper_shadow <= body * 0.35)
                and (0.02 <= pct_chg <= 0.08)
                and (v > v_ma5 * 1.25)
                and (55 <= r <= 73)
            )
            if is_strong_trend and is_tight_box and is_clean_breakout:
                in_pos = True
                entry_p = c
                highest_p = c
                stop_p = max(l * 0.99, c - 1.8 * curr_atr, c * 0.945)
        else:
            highest_p = max(highest_p, h)
            trailing_stop = max(stop_p, highest_p - 2.2 * curr_atr)
            if l <= trailing_stop:
                exit_p = min(o, trailing_stop)
                trades.append(((exit_p - entry_p) / entry_p - fee) * 100)
                in_pos = False
            elif c < m20:
                trades.append(((c - entry_p) / entry_p - fee) * 100)
                in_pos = False

    if not trades:
        return 0, 0.0, 0.0
    wins = [t for t in trades if t > 0]
    return len(trades), len(wins) / len(trades) * 100, sum(trades) / len(trades)


# ================= 5. V6 核心選股掃描器 (篩出前 10 名) =================
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
                if len(df) < 125:
                    continue

                close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
                vol_ma5_s = (vol / 1000).rolling(5).mean()
                vol_ma5 = float(vol_ma5_s.iloc[-1])
                curr_v = float(vol.iloc[-1]) / 1000
                curr_c = float(close.iloc[-1])

                # 流動性門檻：5日均量 >= 1000張，股價 >= 20元
                if vol_ma5 < 1000 or curr_c < 20:
                    continue

                ma5 = float(close.rolling(5).mean().iloc[-1])
                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma60_s = close.rolling(60).mean()
                ma60, ma60_20d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-21])
                ma120 = float(close.rolling(120).mean().iloc[-1])

                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                atr_s = calculate_atr(df, 14)
                curr_atr = float(atr_s.iloc[-1])

                c_prev = float(close.iloc[-2])
                curr_o, curr_h, curr_l = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
                info = stock_map[t]

                if is_bull_market:
                    if pd.isna(curr_atr) or ma60_20d_ago <= 0:
                        continue
                    prev_20d_high = float(close.iloc[-21:-1].max())
                    box_h = float(high.iloc[-6:-1].max())
                    box_l = float(low.iloc[-6:-1].min())
                    box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0

                    # 1. 四線多頭排列 + 季線近20日斜率向上超過 +2.5%
                    is_strong_trend = (curr_c > ma20 > ma60 > ma120) and ((ma60 - ma60_20d_ago) / ma60_20d_ago >= 0.025)
                    # 2. 箱型壓縮蓄勢 (前5日震幅 <= 8.5%)
                    is_tight_box = box_range <= 0.085
                    # 3. 實體飽滿紅K突破20日收盤新高 (防隔日沖長上影)
                    body = curr_c - curr_o
                    upper_shadow = curr_h - curr_c
                    pct_chg = (curr_c - c_prev) / c_prev
                    vol_ok = is_intraday or (curr_v > vol_ma5 * 1.25)
                    is_clean_breakout = (
                        (curr_c > prev_20d_high)
                        and (body > 0)
                        and (upper_shadow <= body * 0.35)
                        and (0.02 <= pct_chg <= 0.08)
                        and vol_ok
                        and (55 <= rsi <= 73)
                    )

                    if not (is_strong_trend and is_tight_box and is_clean_breakout):
                        continue

                    init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
                    trail_ref = max(ma20, curr_c - 2.2 * curr_atr)
                    bt_cnt, bt_win, bt_ret = run_v6_backtest(df)

                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "industry": info["industry"],
                        "close": curr_c, "pct": pct_chg * 100, "rsi": rsi,
                        "break_lvl": prev_20d_high, "init_stop": init_stop, "trail_ref": trail_ref,
                        "bt_cnt": bt_cnt, "bt_win": bt_win, "bt_ret": bt_ret,
                        "vol_ratio": curr_v / vol_ma5 if vol_ma5 > 0 else 1.0,
                        "is_breakout_mode": True
                    })
                else:
                    # 大盤跌破季線時，自動轉為超賣極端反彈防守模式
                    if not (rsi < 28 and rsi > prev_rsi and curr_c > curr_o):
                        continue
                    passed.append({
                        "stock_id": info["id"], "name": info["name"], "industry": info["industry"],
                        "close": curr_c, "pct": ((curr_c - c_prev) / c_prev) * 100, "rsi": rsi,
                        "break_lvl": curr_c, "init_stop": curr_l * 0.97, "trail_ref": ma5,
                        "bt_cnt": 0, "bt_win": 0.0, "bt_ret": 0.0,
                        "vol_ratio": curr_v / vol_ma5 if vol_ma5 > 0 else 1.0,
                        "is_breakout_mode": False
                    })
        except Exception:
            continue

    # 先篩出前 20 檔查詢 5 日法人買賣超，再綜合評分排序取 TOP 10
    passed.sort(key=lambda x: (-x["bt_ret"], -x["vol_ratio"]))
    candidates = passed[:20]

    for item in candidates:
        item["inst_5d"] = check_inst_buy_5d(item["stock_id"])
        # 綜合評分：法人買超籌碼(+35) + 歷史回測正報酬/高勝率(+35) + 突破量能倍數(+30)
        score = 0
        if item["inst_5d"] > 0:
            score += min(35, item["inst_5d"] / 80)
        if item["bt_cnt"] >= 1 and item["bt_ret"] > 0:
            score += min(35, 15 + item["bt_ret"] * 2)
        score += min(30, item["vol_ratio"] * 12)
        item["score"] = round(score, 1)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]
# ================= 6. 建構 LINE Flex Message 卡片 =================
def build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str):
    is_bull = twii["is_bull"] if twii else True
    header_bg = "#1E3A8A" if is_bull else "#1B5E20"
    mode_tag = "⚡ 盤中電子AI突破預警" if is_intraday else "🚀 盤後電子AI強勢突破 TOP 10"
    regime_text = "多頭順風 (四線多頭突破策略)" if is_bull else "空頭防守 (超賣止跌搶反彈)"

    header_contents = [
        {"type": "text", "text": f"{mode_tag} • {now_str}", "color": "#93C5FD", "size": "xs", "weight": "bold"},
        {"type": "text", "text": f"大盤：{regime_text}", "color": "#FFFFFF", "size": "md", "weight": "bold", "margin": "xs"},
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
            "contents": [{"type": "text", "text": "今日電子/AI族群無符合「箱型蓄勢＋實體飽滿帶量突破20日新高」之標的，嚴守紀律空手觀望。", "size": "sm", "color": "#64748B", "wrap": True}],
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
            badge_str = f"🚀突破({ind_short} {s['vol_ratio']:.1f}x量)" if s["is_breakout_mode"] else f"🛡️超賣({ind_short})"
            if s["bt_cnt"] > 0:
                ret_sign = "+" if s["bt_ret"] >= 0 else ""
                bt_str = f"回測{s['bt_cnt']}次勝率{s['bt_win']:.0f}%(均{ret_sign}{s['bt_ret']:.1f}%)"
            else:
                bt_str = "2年首現突破"

            body_contents.append({
                "type": "box", "layout": "vertical", "margin": "md" if idx > 0 else "none",
                "contents": [
                    {"type": "box", "layout": "horizontal", "contents": [
                        {"type": "text", "text": f"#{idx+1} {s['stock_id']} {s['name']}", "weight": "bold", "size": "sm", "color": "#0F172A", "flex": 5},
                        {"type": "text", "text": badge_str, "size": "xs", "weight": "bold", "color": "#D97706", "align": "end", "flex": 5},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "contents": [
                        {"type": "text", "text": f"{s['close']:.1f} ({pct_sign}{s['pct']:.1f}%)", "size": "sm", "weight": "bold", "color": pct_color, "flex": 4},
                        {"type": "text", "text": f"5日法人:{inst_sign}{s['inst_5d']}張 | {bt_str}", "size": "xxs", "color": inst_color, "align": "end", "flex": 8},
                    ]},
                    {"type": "box", "layout": "horizontal", "margin": "xs", "backgroundColor": "#F1F5F9", "paddingAll": "5px", "cornerRadius": "4px", "contents": [
                        {"type": "text", "text": f"突破點 {s['break_lvl']:.1f}", "size": "xxs", "color": "#334155", "flex": 1},
                        {"type": "text", "text": f"硬停損 {s['init_stop']:.1f}", "size": "xxs", "color": "#15803D", "weight": "bold", "align": "center", "flex": 1},
                        {"type": "text", "text": f"移動停利 {s['trail_ref']:.1f}", "size": "xxs", "color": "#B91C1C", "align": "end", "flex": 1},
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
    return {"type": "flex", "altText": f"{mode_tag}：共篩選出 {len(picks)} 檔強勢標的", "contents": bubble}


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
    print(f"[{now_str}] 開始執行 V6 電子/AI/半導體精準突破掃描 (盤中模式={is_intraday})...")

    twii = get_twii_status()
    is_bull = twii["is_bull"] if twii else True

    stock_map = get_tech_stock_list()
    print(f"取得電子/AI/半導體有效標的共 {len(stock_map)} 檔")

    picks = run_screener(stock_map, is_bull_market=is_bull, is_intraday=is_intraday)
    print(f"V6 策略篩選出 TOP {len(picks)} 檔標的")

    if is_intraday and not picks:
        print("盤中無符合條件之標的，略過推播。")
        return

    top_buy, top_sell = ([], []) if is_intraday else get_top5_tech_institutional(stock_map)
    flex_msg = build_flex_message(twii, picks, top_buy, top_sell, is_intraday, now_str)
    push_line_message(flex_msg)


if __name__ == "__main__":
    main()
