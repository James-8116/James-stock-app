import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

# ================= 環境變數與設定 =================
LINE_TOKEN = "".join((os.environ.get("LINE_TOKEN") or "").split())
LINE_USER_ID = "".join((os.environ.get("LINE_USER_ID") or "").split())
FINMIND_TOKEN = "".join((os.environ.get("FINMIND_TOKEN") or "").split())

FINMIND_URL = "https://api." + "://finmindtrade.com"
STREAMLIT_APP_URL = "https://james-" + "stockapp.streamlit.app/"
LINE_PUSH_URL = "https://api." + "line.me/v2/bot/message/push"
TW_TZ = timezone(timedelta(hours=8))

TECH_INDUSTRIES = {
    "半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
    "通信網路業", "電子通路業", "資訊服務業", "其他電子業"
}


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
        ].drop_duplicates(subset=["stock_id"], keep="last")

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
        rsi = float(calculate_rsi(close, 14).iloc[-1])
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
    if not FINMIND_TOKEN:
        return 0
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
    return 0
def run_dual_screener(stock_map, is_bull_market=True, is_intraday=False):
    tickers = list(stock_map.keys())
    if not tickers:
        return [], []

    breakout_list = []
    rebound_list = []

    for i in range(0, len(tickers), 200):
        batch = tickers[i : i + 200]
        try:
            data = yf.download(batch, period="1y", group_by="ticker", progress=False, threads=True)
            for t in batch:
                if t not in data.columns.get_level_values(0):
                    continue
                df = data[t].dropna(subset=["Close", "Volume"])
                if len(df) < 125:
                    continue

                close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
                vol_ma5 = float((vol / 1000).rolling(5).mean().iloc[-1])
                curr_v = float(vol.iloc[-1]) / 1000
                curr_c = float(close.iloc[-1])

                if vol_ma5 < 1000 or curr_c < 20:
                    continue

                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma60_s = close.rolling(60).mean()
                ma60, ma60_20d_ago = float(ma60_s.iloc[-1]), float(ma60_s.iloc[-21])
                ma120 = float(close.rolling(120).mean().iloc[-1])
                rsi_s = calculate_rsi(close, 14)
                rsi, prev_rsi = float(rsi_s.iloc[-1]), float(rsi_s.iloc[-2])
                curr_atr = float(calculate_atr(df, 14).iloc[-1])

                if pd.isna(curr_atr) or ma60_20d_ago <= 0:
                    continue

                c_prev = float(close.iloc[-2])
                curr_o, curr_h, curr_l = float(open_p.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
                pct_chg = (curr_c - c_prev) / c_prev * 100
                vol_ratio = curr_v / vol_ma5 if vol_ma5 > 0 else 1.0
                info = stock_map[t]

                # -------- 引擎 A：V6 順勢精準突破 (大盤多頭時啟動) --------
                if is_bull_market:
                    prev_20d_high = float(close.iloc[-21:-1].max())
                    box_h = float(high.iloc[-6:-1].max())
                    box_l = float(low.iloc[-6:-1].min())
                    box_range = (box_h - box_l) / box_l if box_l > 0 else 1.0
                    ma60_slope = (ma60 - ma60_20d_ago) / ma60_20d_ago

                    is_strong_trend = (curr_c > ma20 > ma60 > ma120) and (ma60_slope >= 0.025)
                    is_tight_box = box_range <= 0.085
                    body = curr_c - curr_o
                    upper_shadow = curr_h - curr_c
                    vol_ok = is_intraday or (vol_ratio >= 1.25)
                    is_clean_breakout = (
                        (curr_c > prev_20d_high) and (body > 0) and (upper_shadow <= body * 0.35)
                        and (2.0 <= pct_chg <= 8.0) and vol_ok and (55 <= rsi <= 73)
                    )
                    if is_strong_trend and is_tight_box and is_clean_breakout:
                        init_stop = max(curr_l * 0.99, curr_c - 1.8 * curr_atr, curr_c * 0.945)
                        trail_ref = max(ma20, curr_c - 2.2 * curr_atr)
                        breakout_list.append({
                            "id": info["id"], "name": info["name"], "tag": "🚀 突破",
                            "close": curr_c, "pct": pct_chg,
                            "buy_range": f"{curr_c*0.995:.1f}~{curr_c*1.01:.1f}",
                            "sell_targets": f"{trail_ref:.1f}/{curr_c*1.09:.1f}",
                            "stop_price": f"{init_stop:.1f}",
                            "sort_key": vol_ratio
                        })

                # -------- 引擎 B：黃金分割 (0.500 / 0.618) ＋ 季線共振低接 --------
                h60 = float(high.iloc[-60:].max())
                l60 = float(low.iloc[-60:].min())
                swing = h60 - l60
                if swing > 0:
                    fib500 = h60 - swing * 0.500
                    fib618 = h60 - swing * 0.618
                    fib786 = h60 - swing * 0.786

                    hit_fib = None
                    if abs(curr_c - fib618) / fib618 <= 0.025 or abs(curr_l - fib618) / fib618 <= 0.025:
                        hit_fib = fib618
                    elif abs(curr_c - fib500) / fib500 <= 0.025 or abs(curr_l - fib500) / fib500 <= 0.025:
                        hit_fib = fib500

                    if hit_fib:
                        is_conf = abs(hit_fib - ma60) / ma60 <= 0.025
                        lower_shadow = min(curr_c, curr_o) - curr_l
                        is_stop_drop = (rsi <= 44) and (rsi >= prev_rsi - 1.0 or lower_shadow > abs(curr_c - curr_o) * 0.7)
                        if is_stop_drop:
                            dual_stop = max(fib786, curr_c * 0.945)
                            tag_str = "🔥 共振" if is_conf else "📐 低接"
                            rebound_list.append({
                                "id": info["id"], "name": info["name"], "tag": tag_str,
                                "close": curr_c, "pct": pct_chg,
                                "buy_range": f"{hit_fib*0.99:.1f}~{hit_fib*1.005:.1f}",
                                "sell_targets": f"{ma20:.1f}/{h60:.1f}",
                                "stop_price": f"{dual_stop:.1f}",
                                "sort_key": (100 if is_conf else 0) + (50 - rsi)
                            })
        except Exception:
            continue

    # 各取前 12 名查詢 5 日法人買賣超，依籌碼與強度各選出 TOP 5 (合計 TOP 10)
    breakout_list.sort(key=lambda x: x["sort_key"], reverse=True)
    rebound_list.sort(key=lambda x: x["sort_key"], reverse=True)

    top_breakouts = breakout_list[:12]
    for item in top_breakouts:
        item["inst_5d"] = check_inst_buy_5d(item["id"])
    top_breakouts.sort(key=lambda x: (x["inst_5d"] > 0, x["inst_5d"], x["sort_key"]), reverse=True)

    top_rebounds = rebound_list[:12]
    for item in top_rebounds:
        item["inst_5d"] = check_inst_buy_5d(item["id"])
    top_rebounds.sort(key=lambda x: (x["inst_5d"] > 0, x["inst_5d"], x["sort_key"]), reverse=True)

    return top_breakouts[:5], top_rebounds[:5]
def build_table_section(title_text, title_color, items):
    blocks = [
        {"type": "text", "text": title_text, "weight": "bold", "size": "md", "color": title_color, "margin": "md"},
        {
            "type": "box", "layout": "horizontal", "margin": "sm",
            "contents": [
                {"type": "text", "text": "標的/法人", "size": "xs", "color": "#64748B", "flex": 4},
                {"type": "text", "text": "現價/低接買點", "size": "xs", "color": "#64748B", "align": "center", "flex": 4},
                {"type": "text", "text": "賣點/雙重防守", "size": "xs", "color": "#64748B", "align": "end", "flex": 4},
            ]
        },
        {"type": "separator", "margin": "xs"}
    ]

    if not items:
        blocks.append({
            "type": "text", "text": "今日無符合此策略嚴格條件之標的", "size": "xs", "color": "#94A3B8", "margin": "sm"
        })
        return blocks

    for s in items:
        pct_color = "#D32F2F" if s["pct"] >= 0 else "#2E7D32"
        pct_sign = "+" if s["pct"] >= 0 else ""
        inst_sign = "+" if s["inst_5d"] > 0 else ""

        blocks.append({
            "type": "box", "layout": "horizontal", "margin": "sm", "alignItems": "center",
            "contents": [
                # 左欄：股名標籤 + 代號(法人張數)
                {
                    "type": "box", "layout": "vertical", "flex": 4,
                    "contents": [
                        {"type": "text", "text": f"{s['name']} {s['tag']}", "size": "xs", "weight": "bold", "color": "#0F172A", "wrap": True},
                        {"type": "text", "text": f"{s['id']}({inst_sign}{s['inst_5d']}張)", "size": "xxs", "color": "#64748B"},
                    ]
                },
                # 中欄：現價(漲跌%) + 建議買區
                {
                    "type": "box", "layout": "vertical", "flex": 4,
                    "contents": [
                        {"type": "text", "text": f"{s['close']:.1f}({pct_sign}{s['pct']:.1f}%)", "size": "xs", "weight": "bold", "color": pct_color, "align": "center"},
                        {"type": "text", "text": f"買:{s['buy_range']}", "size": "xxs", "color": "#1E40AF", "align": "center"},
                    ]
                },
                # 右欄：賣點目標 + 雙重防守停損
                {
                    "type": "box", "layout": "vertical", "flex": 4,
                    "contents": [
                        {"type": "text", "text": f"賣:{s['sell_targets']}", "size": "xxs", "color": "#B91C1C", "align": "end"},
                        {"type": "text", "text": f"守:{s['stop_price']}", "size": "xxs", "color": "#15803D", "weight": "bold", "align": "end"},
                    ]
                }
            ]
        })
        blocks.append({"type": "separator", "margin": "sm"})

    return blocks


def build_flex_message(twii, breakouts, rebounds, is_intraday, now_str):
    is_bull = twii["is_bull"] if twii else True
    header_bg = "#1E3A8A" if is_bull else "#1B5E20"
    mode_tag = "⚡ 盤中電子AI雙策略預警" if is_intraday else "📊 盤後電子AI雙策略 TOP 10"

    header_contents = [
        {"type": "text", "text": f"{mode_tag} • {now_str}", "color": "#93C5FD", "size": "xs", "weight": "bold"},
    ]
    if twii:
        sign = "+" if twii["pct"] >= 0 else ""
        header_contents.append({
            "type": "text",
            "text": f"加權 {twii['close']:,.0f} ({sign}{twii['pct']:.2f}%) | 季線 {twii['ma60']:,.0f}",
            "color": "#FFFFFF", "size": "sm", "weight": "bold", "margin": "xs",
        })

    body_contents = []
    body_contents.extend(build_table_section("🚀 V6 強勢順勢突破 TOP 5", "#B91C1C", breakouts))
    body_contents.extend(build_table_section("📉 黃金分割＋季線共振低接 TOP 5", "#1E40AF", rebounds))

    bubble = {
        "type": "bubble", "size": "mega",
        "header": {"type": "box", "layout": "vertical", "backgroundColor": header_bg, "paddingAll": "12px", "contents": header_contents},
        "body": {"type": "box", "layout": "vertical", "paddingAll": "12px", "contents": body_contents},
        "footer": {"type": "box", "layout": "vertical", "paddingAll": "10px", "contents": [
            {"type": "button", "style": "primary", "color": "#1E3A8A", "height": "sm", "action": {"type": "uri", "label": "開啟 Streamlit 雙策略互動看板", "uri": STREAMLIT_APP_URL}}
        ]},
    }
    total_cnt = len(breakouts) + len(rebounds)
    return {"type": "flex", "altText": f"{mode_tag}：共篩選出 {total_cnt} 檔潛力標的", "contents": bubble}


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
    print(f"[{now_str}] 開始執行雙策略電子/AI掃描 (盤中={is_intraday})...")

    twii = get_twii_status()
    is_bull = twii["is_bull"] if twii else True

    stock_map = get_tech_stock_list()
    breakouts, rebounds = run_dual_screener(stock_map, is_bull_market=is_bull, is_intraday=is_intraday)
    print(f"篩出 V6突破 {len(breakouts)} 檔 + 黃金低接 {len(rebounds)} 檔")

    if is_intraday and not breakouts and not rebounds:
        print("盤中無符合標的，略過推播。")
        return

    flex_msg = build_flex_message(twii, breakouts, rebounds, is_intraday, now_str)
    push_line_message(flex_msg)


if __name__ == "__main__":
    main()
