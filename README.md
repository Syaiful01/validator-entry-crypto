# Chart Validation Assistant (CVA)

Aplikasi lokal untuk memvalidasi setup teknikal dari screenshot chart sebelum entry.
Alurnya: screenshot chart → analisis Gemini Vision → digabung data Binance Futures
(Open Interest, Funding Rate, Long/Short Ratio) → skor validasi 0-100% + alasan + rekomendasi.

> Alat bantu edukasi, bukan saran finansial.

## Fitur
- Upload PNG/JPG/JPEG, preview, input symbol / bias / timeframe / catatan
- Skor total, skor teknikal, skor data pasar (berwarna) dan badge rekomendasi
- Detail teknikal (trend, pola, S/R, entry, SL/TP, kekuatan setup, risiko)
- Detail pasar (OI, perubahan OI, funding, L/S global, top trader, mark price)
- History di SQLite: filter, detail, hapus per item, clear all
- Gemini dan Binance dipanggil paralel agar cepat

## Instalasi
Butuh Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env`, lalu isi `GEMINI_API_KEY`
(gratis di https://aistudio.google.com/apikey).

## Menjalankan
```bash
streamlit run app.py
```
Buka http://localhost:8501

## Cara pakai
1. Upload screenshot chart di sidebar
2. Isi symbol (mis. `BTCUSDT`; `BTC`, `BTCUSDT.P` juga diterima), pilih bias
3. Klik **Analisis Setup Sekarang**

## Sistem skor
Skor total = 55% teknikal + 45% data pasar (bobot bisa diubah lewat slider di sidebar,
atau `DEFAULT_WEIGHTS` di `modules/scoring.py`).

| Skor | Rekomendasi |
|------|-------------|
| >= 80 | Sangat Kuat |
| 65-79 | Bagus |
| 50-64 | Netral - Perlu Konfirmasi |
| < 50 | Lemah - Hindari |

Aturan data pasar (untuk bias Long/Short):
- Funding ekstrem searah bias → skor turun
- OI naik bersama harga searah bias → skor naik (short covering / liquidation dinilai lebih lemah)
- L/S ratio global terlalu crowded di sisi bias → skor turun
- Top trader condong searah bias → skor naik
- Chart tidak terbaca → skor teknikal dibatasi maks. 40
- Data Binance gagal → skor hanya dari teknikal

Semua ambang batas ada di `modules/scoring.py`.

## Struktur
```
app.py                     UI Streamlit
modules/__init__.py         Penanda package modul
modules/vision_analyzer.py Gemini Vision + prompt
modules/binance_data.py    Binance Futures public API
modules/scoring.py         logika skor
modules/database.py        SQLite (data/history.db, dibuat otomatis)
```

## Troubleshooting
- **GEMINI_API_KEY ditolak / model tidak ditemukan**: cek key, atau ubah `GEMINI_MODEL` di `.env`.
- **Binance HTTP 451/403**: jaringan/wilayah Anda diblokir; coba jaringan lain atau VPN.
- **Rate limit**: tunggu beberapa saat, lalu coba lagi.
- **Symbol tidak valid**: harus ada di Futures USDT-M (mis. `BTCUSDT`, `ETHUSDT`).
- **Chart tidak jelas**: gunakan screenshot resolusi baik dengan sumbu harga terlihat.
- Kecepatan tergantung API; model `flash` umumnya paling cepat.
