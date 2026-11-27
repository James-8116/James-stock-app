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

STOCK_NAME_CACHE = {}
CACHE_EXPIRE_TS = 0


def get_stock_name(stock_id):
    global STOCK_NAME_CACHE, CACHE_EXPIRE_TS
    now_ts = time.time()
    if not STOCK_NAME_CACHE or now_ts > CACHE_EXPIRE_TS:
        temp_map = {}
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
    start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
        "token": FINMIND_TOKEN,
    }
    try:
        res = requests.get(FINMIND_URL, params=params, timeout=10).json()
        if res.get("msg") == "success" and res.get("data"):
            df = pd.DataFrame(res["data"])
            df["net"] = df["buy"] - df["sell"]
            recent = sorted(df["date"].unique())[-5:]
            return int(round(df[df["date"].isin(recent)]["net"].sum() / 1000))
    except Exception:
        pass
    return None


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
    """引擎 A：V6 順勢箱型突破 2 年回測"""
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


def run_fib_rebound_backtest(df):
    """引擎 B：黃金分割 (0.500 / 0.618) 止跌搶反彈 2 年回測"""
    if len(df) < 70:
        return 0, 0.0, 0.0
    close, open_p, high, low = df["Close"], df["Open"], df["High"], df["Low"]
    ma20 = close.rolling(20).mean()
    rsi = calculate_rsi(close, 14)

    trades = []
    in_pos, entry_p, stop_p, target_p = False, 0.0, 0.0, 0.0
    hold_days = 0
    fee = 0.005

    for i in range(62, len(df) - 1):
        c, o, l, h = float(close.iloc[i]), float(open_p.iloc[i]), float(low.iloc[i]), float(high.iloc[i])
        m20 = float(ma20.iloc[i])
        r, r_prev = float(rsi.iloc[i]), float(rsi.iloc[i - 1])
        h60, l60 = float(high.iloc[i - 60 : i].max()), float(low.iloc[i - 60 : i].min())
        swing = h60 - l60

        if not in_pos:
            if swing <= 0:
                continue
            fib500 = h60 - swing * 0.500
            fib618 = h60 - swing * 0.618
            fib786 = h60 - swing * 0.786

            hit_fib = (abs(l - fib500) / fib500 <= 0.025) or (abs(l - fib618) / fib618 <= 0.025)
            lower_shadow = min(c, o) - l
            is_stop_drop = (r <= 42) and (r > r_prev or lower_shadow > abs(c - o) * 0.8)

            if hit_fib and is_stop_drop and m20 > c * 1.02:
                in_pos = True
                entry_p = c
                stop_p = max(fib786, c * 0.945)
                target_p = m20
                hold_days = 0
        else:
            hold_days += 1
            if l <= stop_p:
                exit_p = min(o, stop_p)
                trades.append(((exit_p - entry_p) / entry_p - fee) * 100)
                in_pos = False
            elif h >= target_p:
                exit_p = max(o, target_p)
                trades.append(((exit_p - entry_p) / entry_p - fee) * 100)
                in_pos = False
            elif hold_days >= 12:
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
    rsi_s = calculate_rsi(close, 14)
    rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
    curr_atr = float(calculate_atr(df, 14).iloc[-1])

    vol_ma5 = float(vol.rolling(5).mean().iloc[-1])
    curr_v = float(vol.iloc[-1])
    vol_ratio = curr_v / vol_ma5 if vol_ma5 > 0 else 1.0

    prev_20d_high = float(close.iloc[-21:-1].max())
    box_h = float(high.iloc[-6:-1].max())
    box_l = float(low.iloc[-6:-1].min())
    box_range = (box_h - box_l) / box_l * 100 if box_l > 0 else 99.0
    ma60_slope = (ma60 - ma60_20d_ago) / ma60_20d_ago * 100 if ma60_20d_ago > 0 else 0.0

    h60 = float(high.iloc[-60:].max())
    l60 = float(low.iloc[-60:].min())
    swing = h60 - l60
    fib500 = h60 - swing * 0.500
    fib618 = h60 - swing * 0.618
    fib786 = h60 - swing * 0.786

    c1_trend = (curr_c > ma20 > ma60 > ma120) and (ma60_slope >= 2.5)
    c2_box = box_range <= 8.5
    body = curr_c - curr_o
    upper_shadow = curr_h - curr_c
    c3_clean_k = (body > 0) and (upper_shadow <= body * 0.35) and (2.0 <= pct_chg <= 8.0)
    c4_break = (curr_c > prev_20d_high) and (vol_ratio >= 1.25) and (55 <= rsi <= 73)
    is_v6_pass = c1_trend and c2_box and c3_clean_k and c4_break

    near_fib500 = abs(curr_c - fib500) / fib500 <= 0.025 or abs(curr_l - fib500) / fib500 <= 0.025
    near_fib618 = abs(curr_c - fib618) / fib618 <= 0.025 or abs(curr_l - fib618) / fib618 <= 0.025
    lower_shadow = min(curr_c, curr_o) - curr_l
    is_rebound_ready = (near_fib500 or near_fib618) and (rsi <= 43) and (rsi > prev_rsi or lower_shadow > abs(body) * 0.8)

    init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
    trail_ref = max(ma20, curr_c - 2.2 * curr_atr)

    v6_cnt, v6_win, v6_ret = run_v6_backtest(df)
    reb_cnt, reb_win, reb_ret = run_fib_rebound_backtest(df)

    if is_v6_pass or (v6_cnt > 0 and curr_c >= ma20):
        bt_label = f"V6突破回測 {v6_cnt}次｜勝率 {v6_win:.0f}%｜均報 {v6_ret:+.1f}%"
    elif reb_cnt > 0:
        bt_label = f"黃金分割反彈回測 {reb_cnt}次｜勝率 {reb_win:.0f}%｜均報 {reb_ret:+.1f}%"
    else:
        bt_label = "近 2 年無完全吻合之波段訊號"

    if is_v6_pass:
        mode = "BREAKOUT"
        action_summary = "🚀 符合 V6 順勢突破！多頭點火可順勢進場。"
        buy_advice = f"{curr_c * 0.995:.1f} ~ {curr_c * 1.01:.1f} 元（突破確認區）"
        sell_advice = f"波段不設限，跌破 {trail_ref:.1f} 元再移動停利"
        stop_advice = f"持股破 {init_stop:.1f} 元停損；空手若跌破後「站回 {init_stop:.1f} 元」可抓破底翻買點"
    elif is_rebound_ready:
        mode = "REBOUND"
        hit_lvl = "0.618 黃金深水區" if near_fib618 else "0.500 半山腰支撐"
        reb_stop = max(fib786, curr_c * 0.945)
        action_summary = f"📐 觸及 {hit_lvl} 且 RSI 止跌，可輕倉搶反彈！"
        buy_advice = f"{curr_l:.1f} ~ {curr_c:.1f} 元（貼近黃金支撐低接）"
        sell_advice = f"短線反彈先看月線 {ma20:.1f} 元（空間 {(ma20-curr_c)/curr_c*100:+.1f}%）"
        stop_advice = f"持股破 {reb_stop:.1f} 元停損；空手若跌破後「站回 {reb_stop:.1f} 元」可抓破底翻買點"
    else:
        mode = "WAIT"
        if curr_c > fib500:
            next_fib_txt = f"0.500 支撐 {fib500:.1f} 元"
            reb_stop = fib618
        elif curr_c > fib618:
            next_fib_txt = f"0.618 支撐 {fib618:.1f} 元"
            reb_stop = fib786
        else:
            next_fib_txt = f"0.786 極限支撐 {fib786:.1f} 元"
            reb_stop = l60
        action_summary = "⏳ 處於半空中間位置，建議等「拉回支撐」或「破底翻站回」。"
        buy_advice = f"低接等 {next_fib_txt}；追強等站上 {prev_20d_high:.1f} 元"
        sell_advice = f"若持有，反彈靠近月線 {ma20:.1f} 元可先減碼調節"
        stop_advice = f"持股破 {reb_stop:.1f} 元停損；空手若跌破後「站回 {reb_stop:.1f} 元」可抓破底翻買點"

    return {
        "id": stock_id, "name": name, "inst_5d": inst_5d, "mode": mode,
        "close": curr_c, "pct": pct_chg, "rsi": rsi,
        "vol_ratio": vol_ratio, "box_range": box_range, "ma60_slope": ma60_slope,
        "break_lvl": prev_20d_high, "fib500": fib500, "fib618": fib618, "fib786": fib786,
        "c1": c1_trend, "c2": c2_box, "c3": c3_clean_k, "c4": c4_break,
        "bt_label": bt_label, "action_summary": action_summary,
        "buy_advice": buy_advice, "sell_advice": sell_advice, "stop_advice": stop_advice
    }
def build_stock_reply_card(s):
    if s["mode"] == "BREAKOUT":
        header_bg, status_title, title_color = "#B91C1C", "🚀 符合 V6 順勢突破買點！", "#FDE047"
    elif s["mode"] == "REBOUND":
        header_bg, status_title, title_color = "#0F766E", "📐 黃金分割支撐止跌（搶反彈訊號）", "#99F6E4"
    else:
        header_bg, status_title, title_color = "#1E293B", "🔍 雙策略診斷與操作建議 (突破＋黃金分割)", "#93C5FD"

    pct_sign = "+" if s["pct"] >= 0 else ""

    if s["inst_5d"] is not None:
        inst_sign = "+" if s["inst_5d"] > 0 else ""
        inst_color = "#D32F2F" if s["inst_5d"] > 0 else ("#2E7D32" if s["inst_5d"] < 0 else "#475569")
        inst_text = f"近5日三大法人：{inst_sign}{s['inst_5d']:,} 張"
    else:
        inst_color = "#475569"
        inst_text = "近5日三大法人：盤後更新中"

    check_icon = lambda ok: "✅" if ok else "❌"

    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": header_bg, "paddingAll": "16px",
            "contents": [
                {"type": "text", "text": status_title, "color": title_color, "size": "sm", "weight": "bold"},
                {"type": "text", "text": f"{s['id']} {s['name']}", "color": "#FFFFFF", "size": "xxl", "weight": "bold", "margin": "xs"},
                {"type": "text", "text": f"現價：{s['close']:.2f} ({pct_sign}{s['pct']:.2f}%) ｜ RSI：{s['rsi']:.1f}", "color": "#F1F5F9", "size": "md", "margin": "xs"},
            ]
        },
        "body": {
            "type": "box", "layout": "vertical", "paddingAll": "16px",
            "contents": [
                {"type": "text", "text": f"🏦 {inst_text}", "size": "sm", "weight": "bold", "color": inst_color},
                {"type": "text", "text": f"📊 {s['bt_label']}", "size": "sm", "weight": "bold", "color": "#1E3A8A", "margin": "xs", "wrap": True},
                {"type": "separator", "margin": "md"},
                {
                    "type": "box", "layout": "vertical", "margin": "md", "backgroundColor": "#F8FAFC", "paddingAll": "12px", "cornerRadius": "8px",
                    "contents": [
                        {"type": "text", "text": s["action_summary"], "size": "md", "weight": "bold", "color": "#0F172A", "wrap": True},
                        {"type": "text", "text": f"🟢 買進：{s['buy_advice']}", "size": "sm", "color": "#1E40AF", "weight": "bold", "margin": "md", "wrap": True},
                        {"type": "text", "text": f"🔴 停利：{s['sell_advice']}", "size": "sm", "color": "#B91C1C", "weight": "bold", "margin": "sm", "wrap": True},
                        {"type": "text", "text": f"🛡️ 防守/低接：{s['stop_advice']}", "size": "sm", "color": "#15803D", "weight": "bold", "margin": "sm", "wrap": True},
                    ]
                },
                {"type": "separator", "margin": "md"},
                {
                    "type": "box", "layout": "horizontal", "margin": "md", "backgroundColor": "#F1F5F9", "paddingAll": "8px", "cornerRadius": "6px",
                    "contents": [
                        {"type": "text", "text": f"Fib 0.500\n{s['fib500']:.1f}", "size": "xs", "color": "#334155", "weight": "bold", "align": "center", "wrap": True},
                        {"type": "text", "text": f"Fib 0.618\n{s['fib618']:.1f}", "size": "xs", "color": "#0F766E", "weight": "bold", "align": "center", "wrap": True},
                        {"type": "text", "text": f"Fib 0.786\n{s['fib786']:.1f}", "size": "xs", "color": "#B91C1C", "weight": "bold", "align": "center", "wrap": True},
                    ]
                },
                {"type": "separator", "margin": "lg"},
                {"type": "text", "text": "📋 V6 順勢突破條件檢核：", "size": "sm", "weight": "bold", "color": "#475569", "margin": "md"},
                {"type": "text", "text": f"{check_icon(s['c1'])} 四線多頭＋季線斜率 ({s['ma60_slope']:+.1f}% / 需>=2.5%)", "size": "sm", "margin": "sm", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c2'])} 前5日箱型壓縮震幅 ({s['box_range']:.1f}% / 需<=8.5%)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c3'])} 實體飽滿紅K防隔日沖 (漲幅2~8%無長上影)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
                {"type": "text", "text": f"{check_icon(s['c4'])} 突破20日高 {s['break_lvl']:.1f} ＋ 出量 ({s['vol_ratio']:.1f}x)", "size": "sm", "margin": "xs", "color": "#334155", "wrap": True},
            ]
        }
    }
    return {"type": "flex", "altText": f"{s['id']} {s['name']} 雙策略診斷報告", "contents": bubble}


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
