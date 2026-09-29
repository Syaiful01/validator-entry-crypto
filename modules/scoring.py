"""
scoring.py
----------
Logika perhitungan skor validasi setup (0-100).

    Skor Total = bobot_teknikal * Skor Teknikal + bobot_pasar * Skor Data Pasar

Semua aturan ditulis eksplisit supaya mudah dibaca dan diubah. Setiap aturan yang
aktif menambahkan satu alasan (reason) ke daftar hasil.
"""

from __future__ import annotations

from typing import Optional

# Bobot awal (bisa diubah lewat parameter `weights` atau slider di UI)
DEFAULT_WEIGHTS = {"technical": 0.55, "market": 0.45}

# Ambang funding rate per 8 jam (desimal): 0.0005 = 0.05%, 0.001 = 0.10%
FUNDING_HIGH = 0.0005
FUNDING_EXTREME = 0.001

OK, WARN, BAD, INFO = "✅", "⚠️", "❌", "ℹ️"


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def _clamp(value: float, lo: float = 0, hi: float = 100) -> float:
    return max(lo, min(hi, value))


def _entry_mid(tech: dict) -> Optional[float]:
    z = tech.get("entry_zone") or {}
    lo, hi = z.get("low"), z.get("high")
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    return lo if lo is not None else hi


def _direction(bias: str, tech: dict) -> Optional[str]:
    """Arah trade: dari bias user; jika Neutral, coba simpulkan dari posisi SL vs entry."""
    if bias in ("Long", "Short"):
        return bias
    sl, mid = tech.get("stop_loss"), _entry_mid(tech)
    if sl is not None and mid is not None and sl != mid:
        return "Long" if sl < mid else "Short"
    return None


def score_color(score: float) -> str:
    """Warna skor: hijau >=80, kuning 65-79, abu-abu 50-64, merah <50."""
    if score >= 80:
        return "#16a34a"
    if score >= 65:
        return "#ca8a04"
    if score >= 50:
        return "#6b7280"
    return "#dc2626"


def get_recommendation(score: float) -> tuple[str, str]:
    """Return (label rekomendasi, saran singkat)."""
    if score >= 80:
        return "Sangat Kuat", "Setup dan data pasar selaras. Tetap disiplin pada stop loss dan ukuran posisi."
    if score >= 65:
        return "Bagus", "Setup layak dipertimbangkan. Cari konfirmasi tambahan di area entry dan jaga risk/reward."
    if score >= 50:
        return "Netral - Perlu Konfirmasi", "Sinyal campuran. Tunggu konfirmasi (candle close, retest, volume) sebelum entry."
    return "Lemah - Hindari", "Setup tidak didukung teknikal maupun data pasar. Sebaiknya skip atau tunggu setup yang lebih baik."


# ---------------------------------------------------------------------------
# SKOR TEKNIKAL (dari hasil Vision AI)
# ---------------------------------------------------------------------------
def calculate_technical_score(tech: dict, bias: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    strength = tech.get("setup_strength") or 5
    score = float(strength) * 10
    reasons.append(f"{INFO} Kekuatan setup menurut analisis chart: {strength}/10.")

    # 1) Kesesuaian trend dengan bias
    trend = tech.get("trend", "Sideways")
    if bias in ("Long", "Short"):
        with_trend = (bias == "Long" and trend == "Bullish") or (bias == "Short" and trend == "Bearish")
        against = (bias == "Long" and trend == "Bearish") or (bias == "Short" and trend == "Bullish")
        if with_trend:
            score += 10
            reasons.append(f"{OK} Bias {bias} searah dengan trend utama ({trend}).")
        elif against:
            score -= 20
            reasons.append(f"{BAD} Bias {bias} melawan trend utama ({trend}) - probabilitas lebih rendah.")
        else:
            score -= 8
            reasons.append(f"{WARN} Market sideways: bias {bias} rawan whipsaw, butuh breakout/reject jelas.")
    elif trend == "Sideways":
        reasons.append(f"{WARN} Struktur sideways, arah belum jelas.")

    # 2) Manajemen risiko: SL & risk/reward
    direction = _direction(bias, tech)
    mid, sl, tps = _entry_mid(tech), tech.get("stop_loss"), tech.get("take_profit") or []
    if sl is None:
        score -= 5
        reasons.append(f"{WARN} Stop loss tidak terlihat di chart - invalidasi setup belum terdefinisi.")
    elif mid is not None and tps and direction:
        tp = tps[0]
        valid = (sl < mid < tp) if direction == "Long" else (tp < mid < sl)
        if not valid:
            score -= 8
            reasons.append(f"{WARN} Level entry/SL/TP tidak konsisten dengan arah {direction}; periksa kembali.")
        else:
            rr = abs(tp - mid) / abs(mid - sl)
            if rr >= 2:
                score += 8
                reasons.append(f"{OK} Risk/reward ke TP1 sekitar 1:{rr:.1f} (>= 1:2).")
            elif rr >= 1.5:
                score += 4
                reasons.append(f"{OK} Risk/reward ke TP1 sekitar 1:{rr:.1f} (cukup).")
            elif rr < 1:
                score -= 8
                reasons.append(f"{BAD} Risk/reward ke TP1 hanya 1:{rr:.1f} (< 1:1).")
            else:
                reasons.append(f"{INFO} Risk/reward ke TP1 sekitar 1:{rr:.1f}.")

    # 3) Kualitas gambar
    if not tech.get("chart_readable", True):
        score = min(score, 40)
        note = tech.get("readability_note") or "chart kurang jelas"
        reasons.append(f"{BAD} Chart sulit dibaca ({note}) - skor teknikal dibatasi maks. 40. Upload screenshot yang lebih jelas.")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# SKOR DATA PASAR (dari Binance) - mempertimbangkan bias
# ---------------------------------------------------------------------------
def calculate_market_score(market: dict, bias: str) -> tuple[Optional[float], list[str]]:
    """Return (skor atau None jika tidak ada data yang bisa dinilai, alasan)."""
    reasons: list[str] = []
    score, rules_applied = 50.0, 0
    is_dir = bias in ("Long", "Short")

    def ratio_for_bias(r: float) -> float:
        """Ubah ratio sehingga >1 selalu berarti 'searah bias'."""
        return r if bias == "Long" else 1 / r

    # 1) Funding rate
    fr = market.get("funding_rate")
    if fr is not None:
        rules_applied += 1
        pct = fr * 100
        if is_dir:
            bias_fr = fr if bias == "Long" else -fr  # positif = pihak bias yang membayar (crowded)
            if bias_fr >= FUNDING_EXTREME:
                score -= 18
                reasons.append(f"{BAD} Funding {pct:+.4f}% ekstrem searah bias {bias} - posisi terlalu ramai, risiko squeeze/flush.")
            elif bias_fr >= FUNDING_HIGH:
                score -= 10
                reasons.append(f"{WARN} Funding {pct:+.4f}% cukup tinggi searah bias {bias} - mulai crowded.")
            elif bias_fr <= -FUNDING_HIGH:
                score += 6
                reasons.append(f"{OK} Funding {pct:+.4f}% berlawanan dengan bias - ruang squeeze searah setup {bias}.")
            else:
                score += 3
                reasons.append(f"{OK} Funding {pct:+.4f}% netral/wajar - tidak ada tekanan crowding.")
        else:
            if abs(fr) >= FUNDING_EXTREME:
                score -= 8
                reasons.append(f"{WARN} Funding {pct:+.4f}% ekstrem - risiko squeeze di salah satu sisi.")
            else:
                reasons.append(f"{INFO} Funding {pct:+.4f}% dalam kisaran wajar.")
    
    # 2) Open Interest + arah harga (24 jam)
    oi_chg, px_chg = market.get("oi_change_pct"), market.get("price_change_pct")
    if oi_chg is not None and px_chg is not None:
        rules_applied += 1
        oi_up, oi_down = oi_chg >= 2, oi_chg <= -2
        px_up, px_down = px_chg >= 0.3, px_chg <= -0.3
        desc = f"OI {oi_chg:+.1f}% & harga {px_chg:+.1f}% ({market.get('change_window', '24 jam')})"
        if is_dir and (oi_up or oi_down) and (px_up or px_down):
            # Interpretasi klasik: (harga, OI) -> apa yang terjadi
            if px_up and oi_up:
                meaning, bull, delta = "posisi long baru masuk (tren naik sehat)", True, 12
            elif px_down and oi_up:
                meaning, bull, delta = "posisi short baru masuk (tren turun sehat)", False, 12
            elif px_up and oi_down:
                meaning, bull, delta = "kenaikan didorong short covering (kurang sehat)", True, -4
            else:
                meaning, bull, delta = "penurunan didorong long liquidation (mulai kehabisan tenaga)", False, -4
            aligned = (bias == "Long") == bull
            if delta > 0:  # OI mengonfirmasi arah harga
                if aligned:
                    score += delta
                    reasons.append(f"{OK} {desc}: {meaning}, mendukung bias {bias}.")
                else:
                    score -= delta * 0.8
                    reasons.append(f"{BAD} {desc}: {meaning}, berlawanan dengan bias {bias}.")
            else:  # pergerakan tanpa OI baru
                if aligned:
                    score += delta
                    reasons.append(f"{WARN} {desc}: {meaning}; dukungan terhadap bias {bias} kurang solid.")
                else:
                    score -= delta  # delta negatif -> menambah skor
                    reasons.append(f"{INFO} {desc}: {meaning}; tekanan terhadap bias {bias} mulai melemah.")
        else:
            reasons.append(f"{INFO} {desc}: perubahan tidak cukup signifikan untuk menjadi sinyal.")

    # 3) Long/Short ratio global (crowding retail)
    g = market.get("global_ls") or {}
    if g.get("ratio"):
        rules_applied += 1
        r = g["ratio"]
        txt = f"L/S global {r:.2f} (Long {g.get('long_pct', 0):.0f}% / Short {g.get('short_pct', 0):.0f}%)"
        if is_dir:
            crowd = ratio_for_bias(r)
            if crowd >= 2.0:
                score -= 12
                reasons.append(f"{BAD} {txt}: sisi {bias} sangat crowded - rawan dijadikan likuiditas.")
            elif crowd >= 1.5:
                score -= 6
                reasons.append(f"{WARN} {txt}: sisi {bias} cukup ramai, hati-hati crowded trade.")
            elif crowd <= 0.8:
                score += 6
                reasons.append(f"{OK} {txt}: mayoritas retail di sisi berlawanan - posisi kontrarian menguntungkan.")
            else:
                reasons.append(f"{INFO} {txt}: relatif seimbang.")
        else:
            reasons.append(f"{INFO} {txt}.")

    # 4) Top trader (position ratio lebih diprioritaskan daripada account ratio)
    for key, label, up, down in (
        ("top_position_ls", "Top trader position ratio", 8, 8),
        ("top_account_ls", "Top trader account ratio", 4, 4),
    ):
        t = market.get(key) or {}
        if not t.get("ratio"):
            continue
        rules_applied += 1
        r = t["ratio"]
        txt = f"{label} {r:.2f}"
        if is_dir:
            eff = ratio_for_bias(r)
            if eff >= 1.2:
                score += up
                reasons.append(f"{OK} {txt}: trader besar condong searah bias {bias}.")
            elif eff <= 0.85:
                score -= down
                reasons.append(f"{WARN} {txt}: trader besar condong berlawanan dengan bias {bias}.")
            else:
                reasons.append(f"{INFO} {txt}: trader besar relatif netral.")
        else:
            reasons.append(f"{INFO} {txt}.")

    if rules_applied == 0:
        return None, [f"{WARN} Data pasar tidak cukup untuk dinilai."]
    if not is_dir:
        reasons.append(f"{INFO} Bias Neutral: data pasar tidak menambah/mengurangi skor arah, hanya sebagai konteks.")
    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# FUNGSI UTAMA
# ---------------------------------------------------------------------------
def calculate_score(
    teknikal_result: dict,
    market_data: Optional[dict],
    bias: str,
    weights: Optional[dict] = None,
) -> dict:
    """
    Gabungkan skor teknikal & data pasar.

    Returns dict:
        total_score, teknikal_score, market_score (None jika tidak ada data),
        reasons (list[str]), recommendation, advice, color, weights
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    total_w = (w.get("technical", 0) + w.get("market", 0)) or 1
    w = {"technical": w.get("technical", 0) / total_w, "market": w.get("market", 0) / total_w}

    tech_score, tech_reasons = calculate_technical_score(teknikal_result, bias)

    market_score, market_reasons = (None, [])
    if market_data:
        market_score, market_reasons = calculate_market_score(market_data, bias)

    if market_score is None:
        total = tech_score  # tanpa data pasar, hanya teknikal
        market_reasons = market_reasons or [f"{WARN} Data pasar Binance tidak tersedia - skor total hanya dari teknikal."]
        w = {"technical": 1.0, "market": 0.0}
    else:
        total = tech_score * w["technical"] + market_score * w["market"]

    total = round(_clamp(total), 1)
    recommendation, advice = get_recommendation(total)

    return {
        "total_score": total,
        "teknikal_score": round(tech_score, 1),
        "market_score": round(market_score, 1) if market_score is not None else None,
        "reasons": tech_reasons + market_reasons,
        "recommendation": recommendation,
        "advice": advice,
        "color": score_color(total),
        "weights": w,
    }
