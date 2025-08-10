# bybit_scalpel_bot.py — гибрид: умный риск-менеджмент + скальперные входы
import os, time, traceback
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# ------------ .ENV ------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

API_KEY    = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TESTNET    = os.getenv("BYBIT_TESTNET", "0") == "1"  # 1=testnet

# Пары; можно менять в .env
SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS", "ETHUSDT,TONUSDT,XRPUSDT,ADAUSDT"
).split(",") if s.strip()]

LEVERAGE = int(os.getenv("LEVERAGE", "5"))
RISK_PCT = float(os.getenv("RISK_PER_TRADE", "0.05"))    # 5% свободного USDT
CHECK_INTERVAL_SEC = float(os.getenv("CHECK_INTERVAL_SEC", "3"))

# Таймфреймы
TF_ENTRY   = int(os.getenv("TF_ENTRY",  "1"))    # вход — 1m
TF_FILTER  = int(os.getenv("TF_FILTER", "5"))    # фильтр — 5m
TF_FILTER2 = int(os.getenv("TF_FILTER2","15"))   # старший фильтр — 15m

# Фильтры рынка
ATR_MIN_PCT   = float(os.getenv("ATR_MIN_PCT", "0.08"))   # 1m ATR%, минимум
VOL_LOOKBACK  = int(os.getenv("VOL_LOOKBACK", "20"))
VOL_MULT      = float(os.getenv("VOL_MULT", "1.10"))      # объём > MA20 * VOL_MULT
COOLDOWN_SEC  = int(os.getenv("COOLDOWN_SEC", "8"))
MAX_POS_MIN   = int(os.getenv("MAX_POS_MIN", "12"))       # страховка по времени позиции (минут)

# Сопровождение позиции
BE_TRIGGER_PCT     = float(os.getenv("BE_TRIGGER_PCT", "0.0005"))    # +0.05% → безубыток
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT","0.0006")) # +0.06% → трейл
TRAIL_DISTANCE_PCT = float(os.getenv("TRAIL_DISTANCE_PCT","0.0004")) # 0.04% дистанция

# Стоп: берём min(ATR*mult, фикс %)
STOP_ATR_MULT  = float(os.getenv("STOP_ATR_MULT", "1.2"))
STOP_PCT_HARD  = float(os.getenv("STOP_PCT_HARD","0.0005"))          # 0.05%

# Дневной стоп
DAILY_STOP_PCT = float(os.getenv("DAILY_STOP_PCT","3"))/100.0

session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)

# ------------ INDICATORS ------------
def ema(arr, n):
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=n, adjust=False).mean().values

def rsi(arr, n=14):
    s = pd.Series(arr, dtype="float64")
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / (dn.replace(0, 1e-12))
    return (100 - (100 / (1 + rs))).values

def macd(arr, fast=12, slow=26, signal=9):
    f = ema(arr, fast); s = ema(arr, slow)
    line = f - s
    sig  = pd.Series(line).ewm(span=signal, adjust=False).mean().values
    hist = line - sig
    return line, sig, hist

def atr(high, low, close, n=14):
    h = pd.Series(high, dtype="float64")
    l = pd.Series(low, dtype="float64")
    c = pd.Series(close, dtype="float64")
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values

# — упрощённая Gann-направленность: луч 1x1 от последнего пивота
def pivots_hl(h, l, left=3, right=3):
    N = len(h)
    piv_hi, piv_lo = [], []
    for i in range(left, N-right):
        if h[i] == max(h[i-left:i+right+1]): piv_hi.append(i)
        if l[i] == min(l[i-left:i+right+1]): piv_lo.append(i)
    return piv_hi, piv_lo

def gann_direction_from_last_pivot(o,h,l,c):
    a = atr(h,l,c,14)
    scale = np.nanmean(a[-20:])
    if not np.isfinite(scale) or scale <= 0: return "neutral"
    piv_hi, piv_lo = pivots_hl(h,l,3,3)
    if not piv_hi and not piv_lo: return "neutral"
    last_hi = piv_hi[-1] if piv_hi else -10**9
    last_lo = piv_lo[-1] if piv_lo else -10**9
    use_low = last_lo > last_hi
    pivot_idx = last_lo if use_low else last_hi
    pivot_px  = l[pivot_idx] if use_low else h[pivot_idx]
    bars_from = (len(c)-1) - pivot_idx
    if bars_from < 1: return "neutral"
    slope = scale * 1.0
    if use_low:
        ray = pivot_px + slope * bars_from
        return "bull" if c[-1] > ray else "neutral"
    else:
        ray = pivot_px - slope * bars_from
        return "bear" if c[-1] < ray else "neutral"

# ------------ DATA ------------
def get_klines(symbol, interval, limit=240):
    r = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    data = r["result"]["list"][::-1]
    if len(data) < 5:
        raise RuntimeError("not enough candles")
    o = np.array([float(x[1]) for x in data], dtype="float64")
    h = np.array([float(x[2]) for x in data], dtype="float64")
    l = np.array([float(x[3]) for x in data], dtype="float64")
    c = np.array([float(x[4]) for x in data], dtype="float64")
    v = np.array([float(x[5]) for x in data], dtype="float64")
    t = np.array([int(x[0])   for x in data], dtype="int64")
    return t,o,h,l,c,v

def get_last_price(symbol):
    r = session.get_tickers(category="linear", symbol=symbol)
    return float(r["result"]["list"][0]["lastPrice"])

def get_free_usdt():
    r = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    lst = r["result"]["list"]
    if not lst: return 0.0
    return float(lst[0]["totalAvailableBalance"])

# ------------ EXCHANGE FILTERS (minQty/step) ------------
_INSTR = {}
def _get_symbol_filters(symbol):
    if symbol not in _INSTR:
        r = session.get_instruments_info(category="linear", symbol=symbol)
        lot = r["result"]["list"][0]["lotSizeFilter"]
        _INSTR[symbol] = {
            "minQty":  Decimal(lot["minOrderQty"]),
            "qtyStep": Decimal(lot["qtyStep"]),
        }
    return _INSTR[symbol]

def _quantize_qty(symbol, qty_float):
    f = _get_symbol_filters(symbol)
    q = Decimal(str(qty_float))
    stepped = (q / f["qtyStep"]).to_integral_exact(rounding=ROUND_DOWN) * f["qtyStep"]
    if stepped < f["minQty"]:
        stepped = f["minQty"]
    return float(stepped)

# ------------ POSITIONS / ORDERS ------------
def fetch_positions_both(symbol):
    r = session.get_positions(category="linear", symbol=symbol)
    lst = r["result"]["list"]
    buy  = next((p for p in lst if p["side"]=="Buy"  and float(p["size"])>0), None)
    sell = next((p for p in lst if p["side"]=="Sell" and float(p["size"])>0), None)
    return buy, sell

def position_side(symbol):
    buy, sell = fetch_positions_both(symbol)
    if buy:  return "Buy"
    if sell: return "Sell"
    return None

def set_leverage(symbol, lev):
    try:
        session.set_leverage(category="linear", symbol=symbol,
                             buyLeverage=str(lev), sellLeverage=str(lev))
    except Exception:
        pass

def close_side_market(symbol, side):
    buy, sell = fetch_positions_both(symbol)
    p = buy if side=="Buy" else sell
    if not p: return
    qty = p["size"]
    close_side = "Sell" if side=="Buy" else "Buy"
    session.place_order(category="linear", symbol=symbol,
                        side=close_side, orderType="Market", qty=qty, reduceOnly=True)

def open_market_with_sl(symbol, side, qty, sl_price):
    session.place_order(category="linear", symbol=symbol,
                        side=side, orderType="Market", qty=qty, reduceOnly=False)
    try:
        session.set_trading_stop(category="linear", symbol=symbol,
                                 stopLoss=str(round(sl_price, 4)), tpSlMode="Full")
    except Exception:
        pass

# ------------ SIZING / STOPS ------------
def compute_qty(symbol, price):
    free = get_free_usdt()
    notional = max(5.0, free * RISK_PCT)
    raw_qty = notional / price
    return _quantize_qty(symbol, raw_qty)

def initial_sl(symbol, side, entry_price, o,h,l,c):
    a = atr(h,l,c,14)[-1]
    atr_stop  = a * STOP_ATR_MULT
    hard_stop = entry_price * STOP_PCT_HARD
    move = min(atr_stop, hard_stop)
    return entry_price - move if side=="Buy" else entry_price + move

def adjust_sl_trailing(symbol):
    buy, sell = fetch_positions_both(symbol)
    pos = buy or sell
    if not pos: return
    side  = pos["side"]
    entry = float(pos["avgPrice"])
    last  = get_last_price(symbol)
    pnl_pct = (last/entry - 1.0) if side=="Buy" else (entry/last - 1.0)
    new_sl = None
    if pnl_pct >= BE_TRIGGER_PCT:
        new_sl = entry
    if pnl_pct >= TRAIL_ACTIVATE_PCT:
        trail = last*(1 - TRAIL_DISTANCE_PCT) if side=="Buy" else last*(1 + TRAIL_DISTANCE_PCT)
        new_sl = max(new_sl, trail) if side=="Buy" else min(new_sl, trail)
    if new_sl is not None:
        try:
            session.set_trading_stop(category="linear", symbol=symbol,
                                     stopLoss=str(round(new_sl, 4)), tpSlMode="Full")
        except Exception:
            pass

# ------------ STATE / DAILY GUARD ------------
STATE = {
    sym: {"last_trade_ts":0, "open_ts":None, "day_date":None, "day_start":None, "disabled":False}
    for sym in SYMBOLS
}

def daily_guard(symbol):
    st = STATE[symbol]
    today = datetime.now(timezone.utc).date()
    free = get_free_usdt()
    if st["day_date"] != today:
        st["day_date"] = today
        st["day_start"] = free
        st["disabled"] = False
    if st["day_start"] and free < st["day_start"]*(1-DAILY_STOP_PCT):
        st["disabled"] = True
    return not st["disabled"]

# ------------ FILTERЫ ------------
def volume_ok(v, lookback=20):
    if len(v) < lookback+1: return False
    ma = pd.Series(v).rolling(lookback).mean().values
    return v[-1] > ma[-1]*VOL_MULT

def atr_ok(h,l,c):
    a = atr(h,l,c,14)[-1]
    pct = (a / c[-1]) * 100.0
    return pct >= ATR_MIN_PCT

def ema_slope(series, span=5):
    if len(series) <= span: return 0.0
    return series[-1] - series[-span]

def higher_trend(symbol):
    # 5m + 15m согласование
    _,o5,h5,l5,c5,v5 = get_klines(symbol, TF_FILTER,  240)
    _,o15,h15,l15,c15,v15 = get_klines(symbol, TF_FILTER2, 240)
    e50_5  = ema(c5, 50);  e200_5  = ema(c5, 200)
    e50_15 = ema(c15,50);  e200_15 = ema(c15,200)
    dir5  = 1 if e50_5[-1]  > e200_5[-1]  else (-1 if e50_5[-1]  < e200_5[-1]  else 0)
    dir15 = 1 if e50_15[-1] > e200_15[-1] else (-1 if e50_15[-1] < e200_15[-1] else 0)
    dsum = dir5 + dir15
    if dsum > 0:  return "bull"
    if dsum < 0:  return "bear"
    return "neutral"

# ------------ СИГНАЛЫ (вход/переворот) ------------
def entry_signal(symbol):
    # 1m fast
    _,o1,h1,l1,c1,v1 = get_klines(symbol, TF_ENTRY, 240)

    # фильтры рынка
    if not atr_ok(h1,l1,c1):            return None, "atr_low"
    if not volume_ok(v1, VOL_LOOKBACK): return None, "vol_low"

    # быстрые индикаторы
    e9  = ema(c1, 9);  e21 = ema(c1, 21)
    r7  = rsi(c1, 7)
    m_line, m_sig, m_hist = macd(c1)

    long_1m  = (e9[-1] > e21[-1]) and (ema_slope(e21) > 0)
    short_1m = (e9[-1] < e21[-1]) and (ema_slope(e21) < 0)

    long_conf  = (45 <= r7[-1] <= 65) and (m_line[-1] > m_sig[-1] or m_hist[-1] > m_hist[-2])
    short_conf = (35 <= r7[-1] <= 55) and (m_line[-1] < m_sig[-1] or m_hist[-1] < m_hist[-2])

    # доп. фильтр Gann (по 1m)
    gdir = gann_direction_from_last_pivot(o1,h1,l1,c1)

    # тренд старших
    trend = higher_trend(symbol)

    allow_long  = (trend in ("bull","neutral")) and (gdir in ("bull","neutral"))
    allow_short = (trend in ("bear","neutral")) and (gdir in ("bear","neutral"))

    if long_1m and long_conf and allow_long:
        return "Buy", None
    if short_1m and short_conf and allow_short:
        return "Sell", None
    return None, "no_match"

def reverse_needed(symbol, current_side):
    sig, _ = entry_signal(symbol)
    if not sig: return False
    return (current_side=="Buy" and sig=="Sell") or (current_side=="Sell" and sig=="Buy")

# ------------ CORE ------------
def maybe_trade(symbol):
    st = STATE[symbol]
    if not daily_guard(symbol): return
    if time.time() - st["last_trade_ts"] < COOLDOWN_SEC: return

    pos_side = position_side(symbol)

    # страховка по времени
    if pos_side and st["open_ts"]:
        if (time.time() - st["open_ts"]) > MAX_POS_MIN*60:
            close_side_market(symbol, pos_side)
            print(f"[TIMEOUT CLOSE] {symbol} {pos_side}")
            st["last_trade_ts"] = time.time()
            return

    # переворот
    if pos_side and reverse_needed(symbol, pos_side):
        close_side_market(symbol, pos_side)
        time.sleep(0.5)
        desired = "Sell" if pos_side=="Buy" else "Buy"
        price = get_last_price(symbol)
        qty   = compute_qty(symbol, price)
        _,o,h,l,c,v = get_klines(symbol, TF_ENTRY, 120)
        sl = initial_sl(symbol, desired, price, o,h,l,c)
        set_leverage(symbol, LEVERAGE)
        open_market_with_sl(symbol, desired, qty, sl)
        st["open_ts"] = time.time()
        st["last_trade_ts"] = time.time()
        print(f"[FLIP] {symbol}: {pos_side}→{desired} qty={qty} @~{price:.4f} SL={sl:.4f}")
        return

    # новый вход
    if not pos_side:
        side, reason = entry_signal(symbol)
        if not side:
            # для отладки можно включить:
            # print(f"{symbol}: skip ({reason})")
            return
        price = get_last_price(symbol)
        qty   = compute_qty(symbol, price)
        _,o,h,l,c,v = get_klines(symbol, TF_ENTRY, 120)
        sl = initial_sl(symbol, side, price, o,h,l,c)
        set_leverage(symbol, LEVERAGE)
        open_market_with_sl(symbol, side, qty, sl)
        st["open_ts"] = time.time()
        st["last_trade_ts"] = time.time()
        print(f"[OPEN] {symbol}: {side} qty={qty} @~{price:.4f} SL={sl:.4f}")
        return

    # сопровождение
    adjust_sl_trailing(symbol)

def main_loop():
    print("SCALPEL started. Symbols:", SYMBOLS)
    while True:
        try:
            for sym in SYMBOLS:
                try:
                    maybe_trade(sym)
                except Exception as e:
                    print(sym, "error:", e)
            time.sleep(CHECK_INTERVAL_SEC)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("loop error:", e)
            traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main_loop()
