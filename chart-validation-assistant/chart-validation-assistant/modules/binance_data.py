"""
binance_data.py
---------------
Mengambil data pasar dari Binance Futures (USDT-M) via API publik (tanpa API key).

Fungsi utama:
    get_market_data(symbol) -> dict
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import requests

BASE_URL = "https://fapi.binance.com"
TIMEOUT = 8  # detik per request
HIST_PERIOD = "1h"
HIST_LIMIT = 24  # ~24 jam untuk perubahan OI & harga


class BinanceError(Exception):
    """Error Binance dengan kategori (kind) agar UI bisa bereaksi berbeda."""

    def __init__(self, message: str, kind: str = "other"):
        super().__init__(message)
        self.kind = kind  # invalid_symbol | rate_limit | blocked | network | other


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def normalize_symbol(symbol: str) -> str:
    """'btc/usdt', 'BTCUSDT.P', 'eth' -> 'BTCUSDT', 'ETHUSDT'."""
    s = (symbol or "").upper().strip()
    s = re.sub(r"\.P$", "", s)  # akhiran perpetual TradingView
    s = re.sub(r"^[A-Z]+:", "", s)  # prefix exchange, mis. BINANCE:BTCUSDT
    s = re.sub(r"[^A-Z0-9]", "", s)
    if not s:
        return ""
    if not s.endswith(("USDT", "USDC")):
        s += "USDT"
    return s


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(path: str, params: Optional[dict] = None) -> Any:
    """GET ke Binance dengan penanganan error yang jelas."""
    try:
        resp = requests.get(BASE_URL + path, params=params, timeout=TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise BinanceError("Koneksi ke Binance timeout. Coba lagi.", "network") from exc
    except requests.exceptions.RequestException as exc:
        raise BinanceError(f"Tidak bisa terhubung ke Binance: {exc}", "network") from exc

    if resp.status_code in (418, 429):
        raise BinanceError("Rate limit Binance tercapai. Tunggu beberapa saat.", "rate_limit")
    if resp.status_code in (403, 451):
        raise BinanceError(
            "Akses ke Binance Futures diblokir dari jaringan/wilayah Anda "
            f"(HTTP {resp.status_code}). Coba ganti jaringan atau gunakan VPN.",
            "blocked",
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise BinanceError(f"Respons Binance tidak valid (HTTP {resp.status_code}).") from exc

    if isinstance(data, dict) and "code" in data and isinstance(data["code"], int) and data["code"] < 0:
        if data["code"] == -1121:
            raise BinanceError("Symbol tidak valid di Binance Futures USDT-M.", "invalid_symbol")
        raise BinanceError(f"Binance error {data['code']}: {data.get('msg', '')}")
    if resp.status_code >= 400:
        raise BinanceError(f"Binance mengembalikan HTTP {resp.status_code}.")
    return data


# ---------------------------------------------------------------------------
# FETCHER PER ENDPOINT
# ---------------------------------------------------------------------------
def _fetch_premium(symbol: str) -> dict:
    d = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
    if isinstance(d, list):  # jaga-jaga
        d = d[0] if d else {}
    return {
        "mark_price": _f(d.get("markPrice")),
        "index_price": _f(d.get("indexPrice")),
        "funding_rate": _f(d.get("lastFundingRate")),
        "next_funding_time": d.get("nextFundingTime"),
    }


def _fetch_open_interest(symbol: str) -> dict:
    d = _get("/fapi/v1/openInterest", {"symbol": symbol})
    return {"open_interest": _f(d.get("openInterest"))}


def _fetch_oi_history(symbol: str) -> dict:
    rows = _get(
        "/futures/data/openInterestHist",
        {"symbol": symbol, "period": HIST_PERIOD, "limit": HIST_LIMIT},
    )
    if not isinstance(rows, list) or len(rows) < 2:
        return {"oi_change_pct": None}
    first, last = _f(rows[0].get("sumOpenInterest")), _f(rows[-1].get("sumOpenInterest"))
    if not first or last is None:
        return {"oi_change_pct": None}
    return {"oi_change_pct": (last - first) / first * 100}


def _fetch_price_change(symbol: str) -> dict:
    rows = _get(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": HIST_PERIOD, "limit": HIST_LIMIT},
    )
    if not isinstance(rows, list) or not rows:
        return {"price_change_pct": None}
    open_first, close_last = _f(rows[0][1]), _f(rows[-1][4])
    if not open_first or close_last is None:
        return {"price_change_pct": None}
    return {"price_change_pct": (close_last - open_first) / open_first * 100}


def _fetch_ratio(endpoint: str, symbol: str) -> Optional[dict]:
    rows = _get(
        f"/futures/data/{endpoint}",
        {"symbol": symbol, "period": HIST_PERIOD, "limit": 1},
    )
    if not isinstance(rows, list) or not rows:
        return None
    r = rows[-1]
    ratio = _f(r.get("longShortRatio"))
    long_pct = _f(r.get("longAccount"))
    short_pct = _f(r.get("shortAccount"))
    return {
        "ratio": ratio,
        "long_pct": long_pct * 100 if long_pct is not None else None,
        "short_pct": short_pct * 100 if short_pct is not None else None,
    }


# ---------------------------------------------------------------------------
# FUNGSI UTAMA
# ---------------------------------------------------------------------------
def get_market_data(symbol: str) -> dict:
    """
    Ambil snapshot data pasar untuk satu symbol (semua request berjalan paralel).

    Raises:
        BinanceError: jika symbol tidak valid / Binance tidak bisa diakses sama sekali.
    Returns:
        dict bersih. Field sekunder yang gagal diambil bernilai None dan dicatat di 'warnings'.
    """
    sym = normalize_symbol(symbol)
    if not sym:
        raise BinanceError("Symbol kosong.", "invalid_symbol")

    jobs = {
        "premium": lambda: _fetch_premium(sym),
        "oi": lambda: _fetch_open_interest(sym),
        "oi_hist": lambda: _fetch_oi_history(sym),
        "price": lambda: _fetch_price_change(sym),
        "global_ls": lambda: _fetch_ratio("globalLongShortAccountRatio", sym),
        "top_account_ls": lambda: _fetch_ratio("topLongShortAccountRatio", sym),
        "top_position_ls": lambda: _fetch_ratio("topLongShortPositionRatio", sym),
    }

    results: dict[str, Any] = {}
    errors: dict[str, BinanceError] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except BinanceError as exc:
                errors[name] = exc
            except Exception as exc:  # error tak terduga
                errors[name] = BinanceError(str(exc))

    # Error kritis (symbol salah / diblokir / rate limit) -> hentikan dengan pesan jelas
    for critical in ("premium", "oi"):
        if critical in errors and errors[critical].kind in ("invalid_symbol", "blocked", "rate_limit", "network"):
            raise errors[critical]
    if "premium" in errors:
        raise errors["premium"]

    premium = results["premium"]
    oi = results.get("oi", {})
    oi_hist = results.get("oi_hist", {})
    price = results.get("price", {})

    open_interest = oi.get("open_interest")
    mark = premium.get("mark_price")
    funding = premium.get("funding_rate")

    warnings = []
    labels = {
        "oi": "Open Interest",
        "oi_hist": "Perubahan OI",
        "price": "Perubahan harga",
        "global_ls": "Long/Short Ratio Global",
        "top_account_ls": "Top Trader Account Ratio",
        "top_position_ls": "Top Trader Position Ratio",
    }
    for key, label in labels.items():
        if key in errors:
            warnings.append(f"{label} tidak tersedia ({errors[key]})")
        elif key in ("global_ls", "top_account_ls", "top_position_ls") and results.get(key) is None:
            warnings.append(f"{label} tidak tersedia untuk symbol ini")

    return {
        "symbol": sym,
        "mark_price": mark,
        "index_price": premium.get("index_price"),
        "funding_rate": funding,  # bentuk desimal, 0.0001 = 0.01%
        "funding_rate_pct": funding * 100 if funding is not None else None,
        "next_funding_time": premium.get("next_funding_time"),
        "open_interest": open_interest,
        "open_interest_usd": open_interest * mark if open_interest and mark else None,
        "oi_change_pct": oi_hist.get("oi_change_pct"),
        "price_change_pct": price.get("price_change_pct"),
        "change_window": f"~{HIST_LIMIT} jam",
        "global_ls": results.get("global_ls"),
        "top_account_ls": results.get("top_account_ls"),
        "top_position_ls": results.get("top_position_ls"),
        "warnings": warnings,
    }
