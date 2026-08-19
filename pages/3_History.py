import streamlit as st
import pandas as pd
from db import get_connection, init_db
from ui_components import (
    apply_custom_styles,
    render_header
)

# ======================================
# PAGE CONFIG & STYLES
# ======================================
st.set_page_config(
    page_title="Riwayat Komentar & Prediksi",
    page_icon="📚",
    layout="wide"
)

apply_custom_styles()

render_header(
    title="Riwayat Komentar & Prediksi Emosi",
    subtitle="Seluruh daftar komentar yang tersimpan di database beserta hasil prediksi model dan status validasi manusia.",
    icon="📚"
)

init_db()
conn = get_connection()
df = pd.read_sql("SELECT * FROM hasil_prediksi ORDER BY id DESC", conn)
conn.close()

if df.empty:
    st.info("ℹ️ Belum ada riwayat data di database.")
    st.stop()

# ======================================
# FILTER CONTROLS
# ======================================
st.markdown("""
    <div class="custom-card" style="padding: 16px 20px;">
        <h4 style="margin-top:0; font-size:1rem; color:#f8fafc;">🔍 Filter & Pencarian</h4>
    </div>
""", unsafe_allow_html=True)

col_f1, col_f2 = st.columns([1, 1], gap="medium")

with col_f1:
    search_keyword = st.text_input("Cari Kata Kunci dalam Komentar:", placeholder="Ketik kata yang dicari...")

with col_f2:
    all_emotions = ["Semua Emosi", "Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]
    selected_emotion_filter = st.selectbox("Filter Kategori Emosi:", all_emotions)

# Apply Filter
df_filtered = df.copy()

if search_keyword.strip():
    df_filtered = df_filtered[df_filtered["komentar"].str.contains(search_keyword, case=False, na=False)]

if selected_emotion_filter != "Semua Emosi":
    df_filtered = df_filtered[df_filtered["emosi"] == selected_emotion_filter]

st.markdown(f"**Menampilkan `{len(df_filtered)}` dari `{len(df)}` total data**")

# ======================================
# DATAFRAME DISPLAY
# ======================================
st.dataframe(
    df_filtered,
    use_container_width=True,
    column_config={
        "id": st.column_config.Column("ID", width="small"),
        "komentar": st.column_config.Column("Teks Komentar", width="large"),
        "emosi": st.column_config.Column("Emosi Prediksi", width="small"),
        "confidence": st.column_config.NumberColumn("Confidence Score", width="small", format="%.4f"),
        "created_at": st.column_config.DatetimeColumn("Waktu Dibuat", width="small", format="YYYY-MM-DD HH:mm:ss"),
        "label_benar": st.column_config.Column("Label Validasi Manusia", width="small"),
    }
)