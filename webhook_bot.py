import os
import re
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request

app = Flask(__name__)

# ================= 環境變數設定 =================
LINE_TOKEN = "".join((os.environ.get("LINE_TOKEN") or "").split())
FINMIND_TOKEN = "".join((os.environ.get("FINMIND_TOKEN") or "").split())
FINMIND_URL = "https://api." + "://finmindtrade.com"
TWSE_STOCK_URL = "https://openapi." + "twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
LINE_PUSH_URL = "https://api." + "line.me/v2/bot/message/push"

# 記憶體快取中文股名對照表 (避免重複查詢)
STOCK_NAME_CACHE = {}
CACHE_EXPIRE_TS = 0


def get_stock_name(stock_id):
    global STOCK_NAME_CACHE, CACHE_EXPIRE_TS
    now_ts = time.time()

    # 若快取空了或超過 12 小時，重新載入完整台股中文名稱表
    if not STOCK_NAME_CACHE or now_ts > CACHE_EXPIRE_TS:
        temp_map = {}
        # 方法 1：透過 FinMind 取得上市櫃中文名稱
        if FINMIND_TOKEN:
            try:
                res = requests.get(
                    FINMIND_URL,
                    params={"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN},
                    timeout=10
                ).json()
                if res.get("msg") == "success" and res.get("data"):
                    for row in res["data"]:
                        sid = str(row.get("stock_id", "")).strip()
                        sname = str(row.get("stock_name", "")).strip()
                        if sid and sname:
                            temp_map[sid] = sname
            except Exception:
                pass

        # 方法 2：透過證交所 OpenAPI 補充上市股票中文名稱
        if not temp_map:
            try:
                rows = requests.get(TWSE_STOCK_URL, timeout=10).json()
                for r in rows:
                    sid = str(r.get("Code", "")).strip()
                    sname = str(r.get("Name", "")).strip()
                    if sid and sname:
                        temp_map[sid] = sname
            except Exception:
                pass

        if temp_map:
            STOCK_NAME_CACHE = temp_map
            CACHE_EXPIRE_TS = now_ts + 43200

    return STOCK_NAME_CACHE.get(stock_id, stock_id)


def check_inst_buy_5d(stock_id):
    if not FINMIND_TOKEN:
        return None
    start = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
        "token": FINMIND_TOKEN,
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=8).json()
        if res.get("msg") != "success" or not res.get("data"):
            return 0
        df = pd.DataFrame(res["data"])
        df["net"] = df["buy"] - df["sell"]
        recent = sorted(df["date"].unique())[-5:]
        return int(round(df[df["date"].isin(recent)]["net"].sum() / 1000))
    except Exception:
        return None


# ================= 指標與 V6 回測核心 =================
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


def run_v6_backtest(df):
    if len(df) < 130:
        return 0, 0.0, 0.0
    close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    ma20, ma60, ma120 = close.rolling(20).mean(), close.rolling(60).mean(), close.rolling(120).mean()
    vol_ma5 = vol.rolling(5).mean()
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(df, 14)

    trades = []
    in_pos, entry_p, stop_p, highest_p = False, 0.0, 0.0, 0.0
    fee = 0.005

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
                (c > prev_20d_high) and (body > 0) and (upper_shadow <= body * 0.35)
                and (0.02 <= pct_chg <= 0.08) and (v > v_ma5 * 1.25) and (55 <= r <= 73)
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
def analyze_single_stock_v6(stock_id):
    df = yf.download(f"{stock_id}.TW", period="2y", progress=False)
    if df.empty:
        df = yf.download(f"{stock_id}.TWO", period="2y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close", "Volume"])

    if df.empty or len(df) < 125:
        return None

    name = get_stock_name(stock_id)
    inst_5d = check_inst_buy_5d(stock_id)

    close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    curr_c, c_prev = float(close.iloc[-1]), float(close.iloc[-2])
    curr_o, curr_h, curr_l = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
    pct_chg = (curr_c - c_prev) / c_prev * 100

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60_s = close.rolling(60).mean()
    ma60, ma60_20d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-21])
    ma120 = float(close.rolling(120).mean().iloc[-1])
    rsi = float(calculate_rsi(close, 14).iloc[-1])
    curr_atr = float(calculate_atr(df, 14).iloc[-1])

    vol_ma5 = float(vol.rolling(5).mean().iloc[-1])
    curr_v = float(vol.iloc[-1])
    vol_ratio = curr_v / vol_ma5 if vol_ma5 > 0 else 1.0

    prev_20d_high = float(close.iloc[-21:-1].max())
    box_h = float(high.iloc[-6:-1].max())
    box_l = float(low.iloc[-6:-1].min())
    box_range = (box_h - box_l) / box_l * 100 if box_l > 0 else 99.0
    ma60_slope = (ma60 - ma60_20d_ago) / ma60_20d_ago * 100 if ma60_20d_ago > 0 else 0.0

    c1_trend = (curr_c > ma20 > ma60 > ma120) and (ma60_slope >= 2.5)
    c2_box = box_range <= 8.5
    body = curr_c - curr_o
    upper_shadow = curr_h - curr_c
    c3_clean_k = (body > 0) and (upper_shadow <= body * 0.35) and (2.0 <= pct_chg <= 8.0)
    c4_break = (curr_c > prev_20d_high) and (vol_ratio >= 1.25) and (55 <= rsi <= 73)

    is_v6_pass = c1_trend and c2_box and c3_clean_k and c4_break
    init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
    trail_ref = max(ma20, curr_c - 2.2 * curr_atr)
    bt_cnt, bt_win, bt_ret = run_v6_backtest(df)

    # 產生白話買賣價格建議
    if is_v6_pass:
        action_summary = "🔥 強勢突破點火！可於現價附近順勢進場。"
        buy_advice = f"{curr_c * 0.995:.1f} ~ {curr_c * 1.01:.1f} 元（突破確認區）"
        sell_advice = f"看漲不設限，跌破 {trail_ref:.1f} 元再移動停利"
        stop_advice = f"跌破 {init_stop:.1f} 元（今日低點/ATR防線）嚴格停損"
    elif curr_c > ma60 and ma60_slope >= 0:
        action_summary = "⏳ 多頭整理中，尚未帶量過高，建議先等突破再追。"
        buy_advice = f"等帶量站上 {prev_20d_high:.1f} 元（突破頸線）再進場"
        sell_advice = f"若成功突破，波段上看 {prev_20d_high * 1.09:.1f} 元（+9%）"
        stop_advice = f"若持有，跌破季線 {ma60:.1f} 元建議減碼停損"
    else:
        action_summary = "🛡️ 均線偏弱或震盪過大，暫不建議進場追高。"
        buy_advice = f"暫不買進（需先站回月線 {ma20:.1f} 且過前高 {prev_20d_high:.1f}）"
        sell_advice = f"手中若有持股，反彈靠近月線 {ma20:.1f} 元可先調節"
        stop_advice = f"跌破近期支撐 {init_stop:.1f} 元務必嚴格停損"

    return {
        "id": stock_id, "name": name, "inst_5d": inst_5d,
        "close": curr_c, "pct": pct_chg, "rsi": rsi,
        "vol_ratio": vol_ratio, "box_range": box_range, "ma60_slope": ma60_slope,
        "break_lvl": prev_20d_high, "init_stop": init_stop, "trail_ref": trail_ref,
        "c1": c1_trend, "c2": c2_box, "c3": c3_clean_k, "c4": c4_break,
        "is_pass": is_v6_pass, "bt_cnt": bt_cnt, "bt_win": bt_win, "bt_ret": bt_ret,
        "action_summary": action_summary,
        "buy_advice": buy_advice, "sell_advice": sell_advice, "stop_advice": stop_advice
    }


def build_stock_reply_card(s):
    header_bg = "#B91C1C" if s["is_pass"] else "#1E293B"
    status_title = "🚀 符合 V6 精準突破買點！" if s["is_pass"] else "🔍 V6 個股診斷與操作建議"
    pct_sign = "+" if s["pct"] >= 0 else ""

    if s["bt_cnt"] > 0:
        ret_sign = "+" if s["bt_ret"] >= 0 else ""
        bt_text = f"2年回測 {s['bt_cnt']} 次｜勝率 {s['bt_win']:.0f}%｜均報 {ret_sign}{s['bt_ret']:.1f}%"
    else:
        bt_text = "過去 2 年無完全吻合之突破紀錄"

    if s["inst_5d"] is not None:
        inst_sign = "+" if s["inst_5d"] > 0 else ""
        inst_color = "#D32F2F" if s["inst_5d"] > 0 else ("#2E7D32" if s["inst_5d"] < 0 else "#475569")
        inst_text = f"近5日三大法人：{inst_sign}{s['inst_5d']:,} 張"
    else:
        inst_color = "#475569"
        inst_text = "近5日三大法人：暫無數據"

    check_icon = lambda ok: "✅" if ok else "❌"

    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": header_bg, "paddingAll": "16px",
            "contents": [
                {"type": "text", "text": status_title, "color": "#FDE047" if s["is_pass"] else "#93C5FD", "size": "sm", "weight": "bold"},
                {"type": "text", "text": f"{s['id']} {s['name']}", "color": "#FFFFFF", "size": "xxl", "weight": "bold", "margin": "xs"},
                {"type": "text", "text": f"現價：{s['close']:.2f} ({pct_sign}{s['pct']:.2f}%) ｜ RSI：{s['rsi']:.1f}", "color": "#F1F5F9", "size": "md", "margin": "xs"},
            ]
        },
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "16px",
            "contents": [
                # 1. 籌碼與歷史回測摘要
                {"type": "text", "text": f"🏦 {inst_text}", "size": "sm", "weight": "bold", "color": inst_color},
                {"type": "text", "text": f"📊 {bt_text}", "size": "sm", "weight": "bold", "color": "#1E3A8A", "margin": "xs"},

                {"type": "separator", "margin": "md"},

                # 2. 白話買賣價格建議區塊 (放大字體)
                {
                    "type": "box", "layout": "vertical", "margin": "md", "backgroundColor": "#F8FAFC", "paddingAll": "12px", "cornerRadius": "8px",
                    "contents": [
                        {"type": "text", "text": s["action_summary"], "size": "md", "weight": "bold", "color": "#0F172A", "wrap": True},
                        {"type": "text", "text": f"🟢 買進：{s['buy_advice']}", "size": "sm", "color": "#1E40AF", "weight": "bold", "margin": "md", "wrap": True},
                        {"type": "text", "text": f"🔴 停利：{s['sell_advice']}", "size": "sm", "color": "#B91C1C", "weight": "bold", "margin": "sm", "wrap": True},
                        {"type": "text", "text": f"🛡️ 停損：{s['stop_advice']}", "size": "sm", "color": "#15803D", "weight": "bold", "margin": "sm", "wrap": True},
                    ]
                },

                {"type": "separator", "margin": "lg"},

                # 3. 放在下方的 V6 條件診斷細則 (放大字體)
                {"type": "text", "text": "📋 V6 突破條件檢核明細：", "size": "sm", "weight": "bold", "color": "#475569", "margin": "md"},
                {"type": "text", "text": f"{check_icon(s['c1'])} 四線多頭＋季線斜率 ({s['ma60_slope']:+.1f}% / 需>=2.5%)", "size": "sm", "margin": "sm", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c2'])} 前5日箱型壓縮震幅 ({s['box_range']:.1f}% / 需<=8.5%)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c3'])} 實體飽滿紅K防隔日沖 (漲幅2~8%無長上影)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c4'])} 突破20日高 {s['break_lvl']:.1f} ＋ 出量 ({s['vol_ratio']:.1f}x)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
            ]
        }
    }
    return {"type": "flex", "altText": f"{s['id']} {s['name']} V6 診斷與買賣建議", "contents": bubble}


@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_json(silent=True)
    if not body or "events" not in body:
        return "OK"

    for event in body["events"]:
        if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
            user_text = event["message"]["text"].strip()
            user_id = event.get("source", {}).get("userId")

            match = re.search(r"\b(\d{4})\b", user_text)
            if match and user_id:
                stock_id = match.group(1)
                print(f"收到查詢：{stock_id}，來自：{user_id}")
                try:
                    res = analyze_single_stock_v6(stock_id)
                    if res:
                        msg = build_stock_reply_card(res)
                    else:
                        msg = {"type": "text", "text": f"❌ 找不到代號 {stock_id} 或上市櫃資料不足 125 個交易日。"}
                except Exception as e:
                    print("分析錯誤：", e)
                    msg = {"type": "text", "text": f"⚠️ 分析 {stock_id} 時發生錯誤：{e}"}

                r = requests.post(
                    LINE_PUSH_URL,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {LINE_TOKEN}"},
                    json={"to": user_id, "messages": [msg]},
                    timeout=15
                )
                print("LINE Push 結果：", r.status_code, r.text)
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
