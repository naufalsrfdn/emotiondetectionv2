import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from datetime import datetime
import torch
import pandas as pd

from load_model import load_model
from db import get_connection
from ui_components import (
    apply_custom_styles,
    render_header,
    render_model_selector,
    get_emotion_badge,
    EMOTION_COLORS,
    preprocess_text
)

# ===============================
# PAGE CONFIG & STYLES
# ===============================
st.set_page_config(
    page_title="Scraping YouTube & Prediksi Emosi",
    page_icon="📥",
    layout="wide"
)

apply_custom_styles()

render_header(
    title="YouTube Scraper & Prediksi Emosi",
    subtitle="Pengambilan komentar otomatis dari YouTube dan klasifikasi emosi menggunakan model IndoBERT lokal.",
    icon="📥"
)

# ===============================
# MAIN PAGE MODEL SELECTOR
# ===============================
selected_version = render_model_selector(sidebar=False, key="scrape_model_select")

st.markdown('<hr style="border:0; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0 24px 0;">', unsafe_allow_html=True)

# ===============================
# LOAD MODEL BASED ON SELECTION
# ===============================
tokenizer, model, model_name = load_model(selected_version)

label_map = ["Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]

# ===============================
# PREDICT FUNCTION
# ===============================
def predict(text):
    clean_text = preprocess_text(text)
    inputs = tokenizer(
        clean_text if clean_text else text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()

    return label_map[idx], float(probs[0][idx])

# ===============================
# SIMPAN KE DATABASE
# ===============================
def save_to_db(text, label, conf):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO hasil_prediksi (komentar, emosi, confidence) VALUES (?, ?, ?)",
        (text, label, conf)
    )
    conn.commit()
    cursor.close()
    conn.close()

# ===============================
# SCRAPER YOUTUBE
# ===============================
def scrape(url, limit):
    dl = YoutubeCommentDownloader()
    gen = dl.get_comments_from_url(url, sort_by=0)

    data = []
    for i, c in enumerate(gen):
        if i >= limit:
            break
        data.append(c)

    return data

def parse_date(date_raw):
    if isinstance(date_raw, str):
        try:
            return datetime.strptime(date_raw, "%Y-%m-%d").date()
        except:
            return None
    elif isinstance(date_raw, (int, float)):
        try:
            return datetime.fromtimestamp(date_raw).date()
        except:
            return None
    return None

# ===============================
# UI CONFIGURATION CARD
# ===============================
st.markdown('<div class="custom-card"><h3 style="margin-top:0; font-size:1.15rem; color:#f8fafc;">⚙️ Pengaturan Scraping</h3></div>', unsafe_allow_html=True)

col_mode, col_num = st.columns([1, 1], gap="medium")

with col_mode:
    mode = st.radio("Pilih Mode Scraping:", ["Studi Kasus Preset", "Link YouTube Bebas"], horizontal=True)

with col_num:
    jumlah = st.number_input("Jumlah Komentar Maksimal:", min_value=1, max_value=5000, value=50, step=10)

col_d1, col_d2 = st.columns(2, gap="medium")
with col_d1:
    tanggal_mulai = st.date_input("Filter Tanggal Mulai (Opsional)", value=None)
with col_d2:
    tanggal_selesai = st.date_input("Filter Tanggal Selesai (Opsional)", value=None)

STUDI_KASUS_URL = "https://www.youtube.com/watch?v=gpHLhjkMR0E"

target_url = None
if mode == "Studi Kasus Preset":
    st.info(f"📌 **Video Studi Kasus**: `{STUDI_KASUS_URL}` (Diskusi Publik / Demo Rusuh)")
    target_url = STUDI_KASUS_URL
    btn_label = "🚀 Mulai Scraping Studi Kasus"
else:
    target_url = st.text_input("Link YouTube Video:", placeholder="https://www.youtube.com/watch?v=...")
    btn_label = "🚀 Mulai Scraping Link Ini"

if st.button(btn_label, type="primary"):
    if not target_url or not target_url.strip():
        st.error("⚠️ Link YouTube tidak boleh kosong!")
        st.stop()

    with st.spinner("📥 Menghubungi API YouTube & mengambil komentar..."):
        comments = scrape(target_url, jumlah)

    if not comments:
        st.warning("⚠️ Tidak ada komentar yang ditemukan pada video ini.")
        st.stop()

    st.success(f"✅ Berhasil mengambil **{len(comments)}** komentar. Memulai klasifikasi emosi...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    hasil_data = []

    for i, c in enumerate(comments):
        text = c.get("text", "")
        date_raw = c.get("time_parsed") or c.get("time")
        date_obj = parse_date(date_raw)

        if date_obj and tanggal_mulai and tanggal_selesai:
            if not (tanggal_mulai <= date_obj <= tanggal_selesai):
                progress_bar.progress((i + 1) / len(comments))
                continue

        label, conf = predict(text)
        save_to_db(text, label, conf)

        hasil_data.append({
            "Komentar": text,
            "Emosi Terdeteksi": label,
            "Confidence Score": f"{conf*100:.2f}%",
            "Tanggal": str(date_obj) if date_obj else "-"
        })

        status_text.text(f"Memproses {i+1}/{len(comments)} komentar...")
        progress_bar.progress((i + 1) / len(comments))

    status_text.empty()
    st.balloons()
    st.success(f"🎉 Selesai! **{len(hasil_data)}** komentar berhasil diklasifikasikan dan disimpan ke database.")

    if hasil_data:
        df_res = pd.DataFrame(hasil_data)
        st.dataframe(df_res, use_container_width=True)
