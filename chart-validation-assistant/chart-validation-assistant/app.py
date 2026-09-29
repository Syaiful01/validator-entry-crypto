"""
Chart Validation Assistant (CVA)
Jalankan dengan:  streamlit run app.py
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from modules import database as db
from modules.binance_data import BinanceError, get_market_data, normalize_symbol
from modules.scoring import DEFAULT_WEIGHTS, calculate_score
from modules.vision_analyzer import VisionAnalysisError, analyze_chart, get_model_name

load_dotenv()

st.set_page_config(page_title="Chart Validation Assistant", page_icon="📈", layout="wide")
db.init_db()

NO_TF = "(tidak diisi)"
TF_OPTIONS = [NO_TF, "1m", "5m", "15m", "30m", "1H", "2H", "4H", "8H", "1D", "3D", "1W"]

st.markdown(
    """
<style>
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; height: 3.2rem; font-size: 1.05rem; font-weight: 700;
}
.cva-badge {display:inline-block;padding:6px 16px;border-radius:999px;
    background:rgba(0,0,0,.25);font-weight:700;margin-top:8px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# FORMAT HELPER
# ---------------------------------------------------------------------------
def fmt_num(v, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def fmt_price(v) -> str:
    """Jumlah desimal menyesuaikan besar harga."""
    if v is None:
        return "—"
    v = float(v)
    digits = 2 if abs(v) >= 1000 else 4 if abs(v) >= 1 else 6
    return f"{v:,.{digits}f}"


def fmt_levels(levels: list) -> str:
    return ", ".join(fmt_price(x) for x in levels) if levels else "—"


def fmt_ratio(r: dict | None) -> str:
    if not r or r.get("ratio") is None:
        return "—"
    return f"{r['ratio']:.2f}  (L {fmt_num(r.get('long_pct'), 1)}% / S {fmt_num(r.get('short_pct'), 1)}%)"


def fmt_entry(z: dict) -> str:
    lo, hi = z.get("low"), z.get("high")
    if lo is None and hi is None:
        return "—"
    if lo is not None and hi is not None:
        return f"{fmt_price(lo)} - {fmt_price(hi)}"
    return fmt_price(lo if lo is not None else hi)


def build_full_text(symbol, bias, timeframe, tech, market, scoring) -> str:
    """Ringkasan analisis dalam bentuk teks/markdown (disimpan ke history)."""
    lines = [
        f"# {symbol} - Bias {bias}" + (f" - {timeframe}" if timeframe else ""),
        f"Skor total: {scoring['total_score']}% ({scoring['recommendation']})",
        f"Skor teknikal: {scoring['teknikal_score']} | Skor data pasar: {scoring['market_score']}",
        "",
        "## Teknikal",
        f"- Trend: {tech['trend']} - {tech['trend_reason']}",
        f"- Pola: {', '.join(tech['patterns']) or '—'}",
        f"- Support: {fmt_levels(tech['support_levels'])}",
        f"- Resistance: {fmt_levels(tech['resistance_levels'])}",
        f"- Entry zone: {fmt_entry(tech['entry_zone'])} {tech['entry_zone'].get('note', '')}",
        f"- Stop loss: {fmt_price(tech['stop_loss'])}",
        f"- Take profit: {fmt_levels(tech['take_profit'])}",
        f"- Kekuatan setup: {tech['setup_strength']}/10 - {tech['strength_reason']}",
        f"- Risiko: {'; '.join(tech['risk_notes']) or '—'}",
        f"- Ringkasan: {tech['summary']}",
        "",
        "## Alasan validasi",
        *[f"- {r}" for r in scoring["reasons"]],
        "",
        f"## Rekomendasi: {scoring['recommendation']}",
        scoring["advice"],
    ]
    if market:
        lines[8:8] = [
            "## Data pasar",
            f"- Mark price: {fmt_price(market.get('mark_price'))}",
            f"- Funding rate: {fmt_num(market.get('funding_rate_pct'), 4, '%')}",
            f"- Open interest: {fmt_num(market.get('open_interest'), 2)} ({fmt_num(market.get('oi_change_pct'), 2, '%')})",
            f"- L/S global: {fmt_ratio(market.get('global_ls'))}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TAMPILAN HASIL (dipakai untuk analisis baru & detail history)
# ---------------------------------------------------------------------------
def render_result(rec: dict) -> None:
    """
    rec: {symbol, bias, timeframe, notes, timestamp, image_blob, teknikal, market, scoring}
    """
    tech, market, scoring = rec["teknikal"], rec.get("market"), rec["scoring"]

    st.subheader(
        f"{rec['symbol']} - Bias {rec['bias']}"
        + (f" - {rec['timeframe']}" if rec.get("timeframe") else "")
    )
    st.caption(f"Dianalisis pada {rec.get('timestamp', '')}")

    if not tech.get("chart_readable", True):
        st.warning(f"⚠️ Chart sulit dibaca: {tech.get('readability_note') or 'kualitas gambar kurang baik'}. "
                   "Hasil kurang dapat diandalkan - coba upload screenshot yang lebih jelas.")

    # --- Baris atas: gambar + skor ---
    col_img, col_score = st.columns([1, 1])
    with col_img:
        if rec.get("image_blob"):
            st.image(rec["image_blob"], caption="Chart yang dianalisis")
    with col_score:
        st.markdown(
            f'<div style="background:{scoring["color"]};border-radius:16px;padding:24px;'
            f'text-align:center;color:white">'
            f'<div style="font-size:1rem;opacity:.9">Total Skor Validasi</div>'
            f'<div style="font-size:4.5rem;font-weight:800;line-height:1.1">{scoring["total_score"]:.0f}%</div>'
            f'<div class="cva-badge">{scoring["recommendation"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric("Skor Teknikal (Vision AI)", fmt_num(scoring["teknikal_score"], 0, "%"))
        m2.metric("Skor Data Pasar", fmt_num(scoring["market_score"], 0, "%") if scoring["market_score"] is not None else "N/A")
        w = scoring.get("weights", DEFAULT_WEIGHTS)
        st.caption(f"Bobot: teknikal {w['technical']*100:.0f}% - data pasar {w['market']*100:.0f}%")

    st.divider()

    # --- Detail teknikal & pasar ---
    col_t, col_m = st.columns(2)
    with col_t:
        st.markdown("### 🔎 Analisis Teknikal (Gemini)")
        st.markdown(f"**Trend utama:** {tech['trend']}")
        if tech.get("trend_reason"):
            st.caption(tech["trend_reason"])
        st.markdown(f"**Pola terdeteksi:** {', '.join(tech['patterns']) or '—'}")
        st.markdown(f"**Support:** {fmt_levels(tech['support_levels'])}")
        st.markdown(f"**Resistance:** {fmt_levels(tech['resistance_levels'])}")
        st.markdown(f"**Estimasi entry zone:** {fmt_entry(tech['entry_zone'])}")
        if tech["entry_zone"].get("note"):
            st.caption(tech["entry_zone"]["note"])
        st.markdown(f"**Stop loss:** {fmt_price(tech['stop_loss'])}")
        st.markdown(f"**Take profit:** {fmt_levels(tech['take_profit'])}")
        st.markdown(f"**Kekuatan setup:** {tech['setup_strength']}/10")
        if tech.get("strength_reason"):
            st.caption(tech["strength_reason"])
        if tech["risk_notes"]:
            st.markdown("**Catatan risiko teknikal:**")
            for note in tech["risk_notes"]:
                st.markdown(f"- {note}")
        if tech.get("summary"):
            st.info(tech["summary"])

    with col_m:
        st.markdown("### 📊 Data Pasar (Binance Futures)")
        if not market:
            st.warning("Data pasar tidak tersedia untuk analisis ini.")
        else:
            a, b = st.columns(2)
            a.metric("Mark Price", fmt_price(market.get("mark_price")))
            b.metric("Funding Rate", fmt_num(market.get("funding_rate_pct"), 4, "%"))
            c, d = st.columns(2)
            c.metric("Open Interest", fmt_num(market.get("open_interest"), 2))
            d.metric(
                f"Perubahan OI ({market.get('change_window', '24 jam')})",
                fmt_num(market.get("oi_change_pct"), 2, "%"),
            )
            if market.get("open_interest_usd"):
                st.caption(f"Nilai OI ≈ ${market['open_interest_usd']:,.0f}")
            st.markdown(f"**Long/Short Ratio (Global):** {fmt_ratio(market.get('global_ls'))}")
            st.markdown(f"**Top Trader L/S (Account):** {fmt_ratio(market.get('top_account_ls'))}")
            st.markdown(f"**Top Trader L/S (Position):** {fmt_ratio(market.get('top_position_ls'))}")
            st.markdown(f"**Perubahan harga ({market.get('change_window', '24 jam')}):** "
                        f"{fmt_num(market.get('price_change_pct'), 2, '%')}")
            for warn in market.get("warnings", []):
                st.caption(f"⚠️ {warn}")

    st.divider()

    # --- Alasan & rekomendasi ---
    st.markdown("### 🧾 Alasan Validasi")
    for reason in scoring["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("### 🎯 Rekomendasi Final")
    st.markdown(
        f'<div style="border-left:6px solid {scoring["color"]};padding:12px 16px;'
        f'background:rgba(128,128,128,.12);border-radius:6px">'
        f'<b>{scoring["recommendation"]}</b><br>{scoring["advice"]}</div>',
        unsafe_allow_html=True,
    )
    if rec.get("notes"):
        st.markdown("**📝 Catatan pribadi:**")
        st.write(rec["notes"])
    st.caption("⚠️ Alat bantu edukasi, bukan saran finansial. Selalu kelola risiko Anda sendiri.")


# ---------------------------------------------------------------------------
# PROSES ANALISIS
# ---------------------------------------------------------------------------
def handle_analysis(uploaded, symbol_input, bias, timeframe, notes, weights) -> None:
    # Validasi input
    if uploaded is None:
        st.error("Upload screenshot chart terlebih dahulu.")
        return
    symbol = normalize_symbol(symbol_input)
    if not symbol:
        st.error("Symbol wajib diisi, contoh: BTCUSDT.")
        return
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY belum diisi. Salin `.env.example` menjadi `.env`, isi key Anda, lalu restart aplikasi.")
        return

    image_bytes = uploaded.getvalue()

    with st.spinner("Menganalisis chart dengan Gemini & mengambil data Binance..."):
        # Gemini & Binance dijalankan paralel supaya total waktu lebih singkat
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vision = pool.submit(analyze_chart, image_bytes, symbol, bias, timeframe or None)
            f_market = pool.submit(get_market_data, symbol)

            tech = market = tech_err = market_err = None
            try:
                tech = f_vision.result()
            except Exception as exc:
                tech_err = exc
            try:
                market = f_market.result()
            except Exception as exc:
                market_err = exc

    # Symbol salah -> hentikan (skor tidak berarti)
    if isinstance(market_err, BinanceError) and market_err.kind == "invalid_symbol":
        st.error(f"❌ Symbol '{symbol}' tidak ditemukan di Binance Futures USDT-M. Periksa penulisan symbol.")
        return
    if tech_err is not None:
        if isinstance(tech_err, VisionAnalysisError):
            st.error(f"❌ {tech_err}")
        else:
            st.error(f"❌ Terjadi kesalahan saat analisis gambar: {tech_err}")
        return
    if market_err is not None:
        st.warning(f"⚠️ Data Binance tidak bisa diambil ({market_err}). Skor dihitung dari analisis teknikal saja.")
        market = None

    scoring = calculate_score(tech, market, bias, weights)
    full_text = build_full_text(symbol, bias, timeframe, tech, market, scoring)

    try:
        history_id = db.save_analysis(symbol, bias, timeframe, notes, scoring, tech, market, full_text, image_bytes)
    except Exception as exc:
        history_id = None
        st.warning(f"Hasil tidak tersimpan ke history: {exc}")

    st.session_state["last_result"] = {
        "id": history_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "bias": bias,
        "timeframe": timeframe,
        "notes": notes,
        "image_blob": image_bytes,
        "teknikal": tech,
        "market": market,
        "scoring": scoring,
    }


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 CVA")
    st.caption("Chart Validation Assistant")

    uploaded = st.file_uploader("Upload screenshot chart", type=["png", "jpg", "jpeg"])
    symbol_input = st.text_input("Symbol *", placeholder="contoh: BTCUSDT", help="Wajib diisi. Futures USDT-M Binance.")
    bias = st.selectbox("Arah Bias", ["Long", "Short", "Neutral"])
    tf_choice = st.selectbox("Timeframe (opsional)", TF_OPTIONS)
    notes = st.text_area("Catatan pribadi (opsional)", height=100)

    with st.expander("⚙️ Bobot skor (lanjutan)"):
        tech_pct = st.slider("Bobot teknikal (%)", 10, 90, int(DEFAULT_WEIGHTS["technical"] * 100), step=5)
        st.caption(f"Bobot data pasar otomatis: {100 - tech_pct}%")

    analyze_clicked = st.button("🔍 Analisis Setup Sekarang", type="primary")

    if os.getenv("GEMINI_API_KEY"):
        st.success(f"Gemini siap ({get_model_name()})", icon="✅")
    else:
        st.error("GEMINI_API_KEY belum diisi di .env", icon="🔑")

timeframe = "" if tf_choice == NO_TF else tf_choice
weights = {"technical": tech_pct / 100, "market": (100 - tech_pct) / 100}

# ---------------------------------------------------------------------------
# TAB
# ---------------------------------------------------------------------------
tab_new, tab_history = st.tabs(["Analisis Baru", "History"])

with tab_new:
    if analyze_clicked:
        handle_analysis(uploaded, symbol_input, bias, timeframe, notes, weights)

    result = st.session_state.get("last_result")
    if result:
        render_result(result)
    elif uploaded is not None:
        st.subheader("Preview chart")
        st.image(uploaded.getvalue(), caption=uploaded.name)
        st.info("Isi symbol & bias di sidebar, lalu klik **Analisis Setup Sekarang**.")
    else:
        st.title("Chart Validation Assistant")
        st.markdown(
            "Validasi setup teknikal Anda **sebelum entry**:\n\n"
            "1. Upload screenshot chart (TradingView dll.) di sidebar\n"
            "2. Isi symbol (mis. `BTCUSDT`) dan arah bias\n"
            "3. Klik **Analisis Setup Sekarang** untuk mendapat skor 0-100% beserta alasannya"
        )

with tab_history:
    st.subheader("History Analisis")

    f1, f2 = st.columns([2, 2])
    hist_symbol = f1.text_input("Filter symbol", placeholder="mis. BTC", key="hist_symbol")
    hist_min = f2.slider("Skor minimal", 0, 100, 0, key="hist_min")

    # Detail item yang dipilih
    detail_id = st.session_state.get("detail_id")
    if detail_id is not None:
        rec = db.get_history_by_id(detail_id)
        if rec is None:
            st.session_state.pop("detail_id", None)
        else:
            with st.container(border=True):
                if st.button("✖ Tutup detail", key="close_detail"):
                    st.session_state.pop("detail_id", None)
                    st.rerun()
                render_result({**rec, "image_blob": rec.get("image_blob")})
            st.divider()

    rows = db.get_all_history(hist_symbol, hist_min)
    if not rows:
        st.info("Belum ada history yang sesuai filter.")
    else:
        st.caption(f"{len(rows)} analisis ditemukan")
        header = st.columns([1.2, 2, 1.4, 1.2, 1.2, 2.2, 1.6])
        for col, title in zip(header, ["Thumbnail", "Waktu", "Symbol", "Arah", "Skor", "Rekomendasi", "Aksi"]):
            col.markdown(f"**{title}**")

        for row in rows:
            cols = st.columns([1.2, 2, 1.4, 1.2, 1.2, 2.2, 1.6], vertical_alignment="center")
            if row["thumb_blob"]:
                cols[0].image(row["thumb_blob"])
            else:
                cols[0].write("—")
            cols[1].write(row["timestamp"])
            cols[2].write(f"**{row['symbol']}**" + (f" ({row['timeframe']})" if row["timeframe"] else ""))
            cols[3].write(row["bias"])
            cols[4].write(f"**{row['total_score']:.0f}%**")
            cols[5].write(row["recommendation"])
            with cols[6]:
                b1, b2 = st.columns(2)
                if b1.button("🔍", key=f"detail_{row['id']}", help="Lihat detail"):
                    st.session_state["detail_id"] = row["id"]
                    st.rerun()
                if b2.button("🗑️", key=f"del_{row['id']}", help="Hapus item ini"):
                    db.delete_history(row["id"])
                    if st.session_state.get("detail_id") == row["id"]:
                        st.session_state.pop("detail_id", None)
                    st.rerun()

    st.divider()
    with st.expander("🚨 Zona berbahaya"):
        confirm = st.checkbox("Saya yakin ingin menghapus SEMUA history", key="confirm_clear")
        if st.button("Clear All History", disabled=not confirm):
            db.clear_all()
            st.session_state.pop("detail_id", None)
            st.session_state.pop("confirm_clear", None)
            st.rerun()
