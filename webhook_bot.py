import os
import re
import requests
import yfinance as yf
import pandas as pd
from flask import Flask, request

app = Flask(__name__)

# ================= 環境變數設定 =================
LINE_TOKEN = (os.environ.get("LINE_TOKEN") or "").strip()
FINMIND_TOKEN = (os.environ.get("FINMIND_TOKEN") or "").strip()
FINMIND_URL = "https://finmindtrade.com"
LINE_PUSH_URL = "https://line.me"


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


def get_stock_name(stock_id):
    if not FINMIND_TOKEN:
        return stock_id
    try:
        res = requests.get(
            FINMIND_URL,
            params={"dataset": "TaiwanStockInfo", "data_id": stock_id, "token": FINMIND_TOKEN},
            timeout=5
        ).json()
        if res.get("msg") == "success" and res.get("data"):
            return res["data"][-1]["stock_name"]
    except Exception:
        pass
    return stock_id
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

    return {
        "id": stock_id, "name": name, "close": curr_c, "pct": pct_chg,
        "rsi": rsi, "vol_ratio": vol_ratio, "box_range": box_range, "ma60_slope": ma60_slope,
        "break_lvl": prev_20d_high, "init_stop": init_stop, "trail_ref": trail_ref,
        "c1": c1_trend, "c2": c2_box, "c3": c3_clean_k, "c4": c4_break,
        "is_pass": is_v6_pass, "bt_cnt": bt_cnt, "bt_win": bt_win, "bt_ret": bt_ret
    }


def build_stock_reply_card(s):
    header_bg = "#B91C1C" if s["is_pass"] else "#1E293B"
    status_title = "🚀 符合 V6 精準突破買點！" if s["is_pass"] else "🔍 V6 條件診斷報告 (尚未全數達標)"
    pct_sign = "+" if s["pct"] >= 0 else ""

    if s["bt_cnt"] > 0:
        ret_sign = "+" if s["bt_ret"] >= 0 else ""
        bt_text = f"2年回測 {s['bt_cnt']} 次｜勝率 {s['bt_win']:.0f}%｜均報 {ret_sign}{s['bt_ret']:.1f}%"
    else:
        bt_text = "過去 2 年無完全吻合之突破紀錄"

    check_icon = lambda ok: "✅" if ok else "❌"

    bubble = {
        "type": "bubble",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": header_bg, "paddingAll": "14px",
            "contents": [
                {"type": "text", "text": status_title, "color": "#FDE047" if s["is_pass"] else "#93C5FD", "size": "xs", "weight": "bold"},
                {"type": "text", "text": f"{s['id']} {s['name']}", "color": "#FFFFFF", "size": "xl", "weight": "bold", "margin": "xs"},
                {"type": "text", "text": f"現價：{s['close']:.2f} ({pct_sign}{s['pct']:.2f}%) ｜ RSI：{s['rsi']:.1f}", "color": "#F1F5F9", "size": "sm", "margin": "xs"},
            ]
        },
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "14px",
            "contents": [
                {"type": "text", "text": f"📊 {bt_text}", "size": "xs", "weight": "bold", "color": "#1E3A8A"},
                {"type": "separator", "margin": "md"},
                {"type": "text", "text": f"{check_icon(s['c1'])} 四線多頭＋季線斜率 ({s['ma60_slope']:+.1f}% / 需>=2.5%)", "size": "xs", "margin": "md", "color": "#334155"},
                {"type": "text", "text": f"{check_icon(s['c2'])} 前5日箱型壓縮震幅 ({s['box_range']:.1f}% / 需<=8.5%)", "size": "xs", "margin": "xs", "color": "#334155"},
                {"type": "text", "text": f"{check_icon(s['c3'])} 實體飽滿紅K防隔日沖 (漲幅2~8%無長上影)", "size": "xs", "margin": "xs", "color": "#334155"},
                {"type": "text", "text": f"{check_icon(s['c4'])} 突破20日高 {s['break_lvl']:.1f} ＋ 出量 ({s['vol_ratio']:.1f}x)", "size": "xs", "margin": "xs", "color": "#334155"},
                {"type": "separator", "margin": "md"},
                {
                    "type": "box", "layout": "horizontal", "margin": "md", "backgroundColor": "#F8FAFC", "paddingAll": "8px", "cornerRadius": "6px",
                    "contents": [
                        {"type": "text", "text": f"突破頸線\n{s['break_lvl']:.1f}", "size": "xs", "color": "#334155", "align": "center", "wrap": True},
                        {"type": "text", "text": f"初始硬停損\n{s['init_stop']:.1f}", "size": "xs", "color": "#15803D", "weight": "bold", "align": "center", "wrap": True},
                        {"type": "text", "text": f"ATR移動停利\n{s['trail_ref']:.1f}", "size": "xs", "color": "#B91C1C", "weight": "bold", "align": "center", "wrap": True},
                    ]
                }
            ]
        }
    }
    return {"type": "flex", "altText": f"{s['id']} {s['name']} V6 診斷結果", "contents": bubble}


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
