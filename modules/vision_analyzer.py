"""
vision_analyzer.py
------------------
Menganalisis screenshot chart trading dengan Google Gemini Vision dan
mengembalikan hasil terstruktur (dict) yang sudah dinormalisasi.

Fungsi utama:
    analyze_chart(image, symbol=None, bias=None, timeframe=None) -> dict
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from typing import Any, Optional, Union

from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)

# Model bisa diganti lewat .env (GEMINI_MODEL) tanpa mengubah kode.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_IMAGE_SIDE = 1600  # perkecil gambar besar agar upload ke API lebih cepat
MAX_RETRIES = 2
REQUEST_TIMEOUT = 60  # detik


class VisionAnalysisError(Exception):
    """Error yang ramah-pengguna untuk kegagalan analisis gambar."""


# ---------------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """
You are a Senior Technical Analyst with 15+ years of experience reading crypto
trading charts (TradingView, Binance, Bybit, etc.). You are meticulous, sceptical
and honest: you NEVER invent information that is not visible in the image.

## CONTEXT FROM THE TRADER
- Symbol: {symbol}
- Trader's directional bias: {bias}
- Timeframe (if the trader provided it): {timeframe}

## YOUR TASK
Carefully study the chart screenshot and extract the following:
1. Whether the chart is readable (candles, price axis, indicators visible).
2. Main trend on the visible timeframe: Bullish, Bearish or Sideways.
3. Technical patterns you can actually see (e.g. higher highs/higher lows, double
   top/bottom, head & shoulders, flag, wedge, triangle, range, breakout, retest,
   liquidity sweep, order block / FVG if the trader marked them, divergences on
   visible indicators such as RSI/MACD, volume behaviour).
4. Key support and resistance price levels (read them from the price axis or from
   lines/zones drawn on the chart).
5. The ENTRY ZONE drawn by the trader (boxes, horizontal lines, long/short position
   tool). If nothing is drawn, propose a sensible entry zone based on the structure
   and say so in "entry_zone.note".
6. Stop loss and take profit levels if they are drawn (e.g. TradingView
   Long/Short position tool: red area = stop, green area = target). If not visible, use null.
7. Setup strength from 1 to 10 for the trader's bias (see calibration below).
8. Technical risks and weaknesses of this setup.
9. A short summary of the setup (max 3 sentences).

## RULES
- Read price levels as accurately as you can from the price axis. Use plain numbers
  (no currency symbols, no thousand separators), e.g. 67250.5
- If a value is not visible or you are not confident, use null (or an empty list). Do NOT guess.
- If the image is blurry, cropped, has no price axis, is not a trading chart, or is
  otherwise unreliable, set "chart_readable" to false and explain in "readability_note".
  In that case keep "setup_strength" at 3 or lower.
- Judge the setup relative to the trader's bias. If the bias is "Neutral", judge the
  overall quality/clarity of the structure regardless of direction.
- Be critical. Reserve 8-10 for setups with clear structure, confluence, defined
  invalidation and good risk/reward.
- Write all free-text values in INDONESIAN. Keep JSON keys and the enum values
  ("Bullish", "Bearish", "Sideways") in English exactly as specified.

## SETUP STRENGTH CALIBRATION
1-3 : unclear structure / against the trend / no invalidation / chart unreadable
4-5 : mixed signals, weak confluence, poor or unknown risk/reward
6-7 : decent structure with some confluence, acceptable risk/reward
8-9 : strong structure, multiple confluences, clear invalidation, RR >= 2
10  : exceptional, textbook setup (very rare)

## OUTPUT FORMAT
Return ONLY one valid JSON object, no markdown fences, no commentary, with exactly
this schema:

{{
  "chart_readable": true,
  "readability_note": "string, kosongkan jika chart jelas",
  "detected_symbol": "string or null (symbol yang tertulis di chart)",
  "detected_timeframe": "string or null",
  "trend": "Bullish | Bearish | Sideways",
  "trend_reason": "string, alasan singkat",
  "patterns": ["string", "..."],
  "support_levels": [number, "..."],
  "resistance_levels": [number, "..."],
  "entry_zone": {{"low": number or null, "high": number or null, "note": "string"}},
  "stop_loss": number or null,
  "take_profit": [number, "..."],
  "setup_strength": integer 1-10,
  "strength_reason": "string",
  "risk_notes": ["string", "..."],
  "summary": "string"
}}
""".strip()


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def get_model_name() -> str:
    """Nama model Gemini tanpa prefix ``models/`` untuk Google Gen AI SDK."""
    name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return name.removeprefix("models/")


def _load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    """Terima path / bytes / PIL Image, kembalikan PIL Image RGB yang sudah diperkecil."""
    try:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(str(image))
        img = img.convert("RGB")
    except Exception as exc:  # gambar korup / bukan gambar
        raise VisionAnalysisError(
            "File gambar tidak bisa dibaca. Pastikan formatnya PNG/JPG/JPEG yang valid."
        ) from exc

    if max(img.size) > MAX_IMAGE_SIDE:
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    return img


def _image_bytes(img: Image.Image) -> tuple[bytes, str]:
    """Serialisasikan gambar yang sudah dinormalisasi untuk ``Part.from_bytes``."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def _to_float(value: Any) -> Optional[float]:
    """Konversi nilai apa pun (angka / string '67,250.5') ke float, atau None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = re.sub(r"[^\d.\-eE]", "", str(value).replace(",", ""))
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        values = [values] if values is not None else []
    out = [_to_float(v) for v in values]
    return [v for v in out if v is not None]


def _str_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _extract_json(text: str) -> dict:
    """Ambil objek JSON dari respons model (toleran terhadap ```json fences)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Respons tidak mengandung JSON.")
    return json.loads(text[start : end + 1])


def _normalize(raw: dict) -> dict:
    """Rapikan & beri default pada hasil Gemini agar modul lain aman memakainya."""
    trend = str(raw.get("trend", "Sideways")).strip().capitalize()
    if trend not in ("Bullish", "Bearish", "Sideways"):
        trend = "Sideways"

    strength = _to_float(raw.get("setup_strength"))
    strength = int(round(min(10, max(1, strength)))) if strength is not None else 5

    entry = raw.get("entry_zone") if isinstance(raw.get("entry_zone"), dict) else {}
    entry_low, entry_high = _to_float(entry.get("low")), _to_float(entry.get("high"))
    if entry_low is not None and entry_high is not None and entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    readable = raw.get("chart_readable", True)
    if isinstance(readable, str):
        readable = readable.strip().lower() not in ("false", "no", "0")

    return {
        "chart_readable": bool(readable),
        "readability_note": str(raw.get("readability_note") or "").strip(),
        "detected_symbol": raw.get("detected_symbol") or None,
        "detected_timeframe": raw.get("detected_timeframe") or None,
        "trend": trend,
        "trend_reason": str(raw.get("trend_reason") or "").strip(),
        "patterns": _str_list(raw.get("patterns")),
        "support_levels": _float_list(raw.get("support_levels")),
        "resistance_levels": _float_list(raw.get("resistance_levels")),
        "entry_zone": {
            "low": entry_low,
            "high": entry_high,
            "note": str(entry.get("note") or "").strip(),
        },
        "stop_loss": _to_float(raw.get("stop_loss")),
        "take_profit": _float_list(raw.get("take_profit")),
        "setup_strength": strength,
        "strength_reason": str(raw.get("strength_reason") or "").strip(),
        "risk_notes": _str_list(raw.get("risk_notes")),
        "summary": str(raw.get("summary") or "").strip(),
    }


# ---------------------------------------------------------------------------
# FUNGSI UTAMA
# ---------------------------------------------------------------------------
def analyze_chart(
    image: Union[str, bytes, Image.Image],
    symbol: Optional[str] = None,
    bias: Optional[str] = None,
    timeframe: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Analisis screenshot chart dengan Gemini Vision.

    Args:
        image: path file, bytes, atau PIL Image.
        symbol / bias / timeframe: konteks dari trader (opsional tapi disarankan).
    Returns:
        dict hasil analisis teknikal (lihat _normalize untuk field-nya).
    Raises:
        VisionAnalysisError: dengan pesan yang bisa langsung ditampilkan ke user.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise VisionAnalysisError(
            "GEMINI_API_KEY belum diisi. Salin .env.example menjadi .env lalu isi API key Anda."
        )

    img = _load_image(image)
    image_bytes, mime_type = _image_bytes(img)
    prompt = PROMPT_TEMPLATE.format(
        symbol=symbol or "unknown",
        bias=bias or "Neutral",
        timeframe=timeframe or "not provided",
    )

    selected_model = (model_name or get_model_name()).removeprefix("models/")
    client = genai.Client(api_key=api_key)

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,  # rendah = hasil lebih konsisten
                ),
            )
            try:
                text = response.text
            except ValueError as exc:  # diblokir safety filter / kosong
                raise VisionAnalysisError(
                    "Gemini tidak mengembalikan hasil (kemungkinan gambar diblokir filter). "
                    "Coba gambar lain."
                ) from exc
            return _normalize(_extract_json(text))

        except VisionAnalysisError:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc  # JSON rusak -> coba ulang
        except Exception as exc:  # error jaringan / API
            last_error = exc
            msg = str(exc).lower()
            if any(k in msg for k in ("api key", "api_key", "permission", "unauthenticated", "401", "403")):
                raise VisionAnalysisError(
                    "GEMINI_API_KEY ditolak. Periksa apakah key benar dan masih aktif."
                ) from exc
            if "not found" in msg or "404" in msg:
                raise VisionAnalysisError(
                    f"Model '{selected_model}' tidak ditemukan. "
                    "Ubah GEMINI_MODEL di file .env (contoh: gemini-3.5-flash-lite)."
                ) from exc
            if "quota" in msg or "429" in msg or "resource" in msg:
                raise VisionAnalysisError(
                    "Kuota / rate limit Gemini tercapai. Tunggu sebentar lalu coba lagi."
                ) from exc

        if attempt <= MAX_RETRIES:
            time.sleep(1.5 * attempt)

    raise VisionAnalysisError(
        f"Gagal menganalisis gambar dengan Gemini setelah beberapa percobaan: {last_error}"
    )
