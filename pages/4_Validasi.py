from auth import check_password
check_password()

import streamlit as st
import pandas as pd
from db import get_connection, init_db
from ui_components import (
    apply_custom_styles,
    render_header,
    get_emotion_badge
)

# ===============================
# PAGE CONFIG & STYLES
# ===============================
st.set_page_config(
    page_title="Validasi Label Emosi (Human Correction)",
    page_icon="📝",
    layout="wide"
)

apply_custom_styles()

render_header(
    title="Validasi Label Emosi (Human Correction)",
    subtitle="Interface Human-in-the-loop untuk mengverifikasi atau mengoreksi hasil prediksi model sebelum training ulang.",
    icon="📝"
)

label_list = ["Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]

# =====================
# LOAD DATA
# =====================
init_db()
conn = get_connection()
df = pd.read_sql("SELECT * FROM hasil_prediksi ORDER BY id DESC", conn)
conn.close()

df_belum = df[df["label_benar"].isnull()]

m1, m2 = st.columns(2)
with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏳ Komentar Belum Divalidasi</div>
            <div class="metric-value" style="color: #f59e0b;">{len(df_belum):,}</div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✅ Komentar Sudah Divalidasi</div>
            <div class="metric-value" style="color: #10b981;">{len(df) - len(df_belum):,}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if df_belum.empty:
    st.success("🎉 Luar biasa! Seluruh komentar dalam database telah selesai divalidasi.")
    st.stop()

st.markdown("### 🔍 Daftar Komentar yang Membutuhkan Validasi")

validasi_massal = {}

for _, row in df_belum.iterrows():
    c_id = row['id']
    c_text = row['komentar']
    c_pred = row['emosi']
    c_conf = row['confidence']

    badge_html = get_emotion_badge(c_pred, c_conf)

    st.markdown(f"""
        <div class="custom-card">
            <div style="display:flex; justify-shadow:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.85rem; color:#94a3b8; font-weight:700;">🆔 ID: #{c_id}</span>
            </div>
            <div style="font-size:1.05rem; color:#f8fafc; font-weight:500; margin-bottom:12px;">
                "{c_text}"
            </div>
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
                <span style="color:#94a3b8; font-size:0.85rem;">🤖 Prediksi Model:</span>
                {badge_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

    default_index = label_list.index(c_pred) if c_pred in label_list else 0

    col_sel, col_act = st.columns([0.7, 0.3], gap="small")
    with col_sel:
        pilihan = st.selectbox(
            "Tentukan Label Benar (Human Label):",
            label_list,
            index=default_index,
            key=f"select_{c_id}"
        )
        validasi_massal[c_id] = pilihan

    with col_act:
        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button(f"✔ Simpan ID #{c_id}", key=f"valid_{c_id}", type="secondary", use_container_width=True):
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE hasil_prediksi SET label_benar=? WHERE id=?", (pilihan, c_id))
            conn.commit()
            cursor.close()
            conn.close()

            st.toast(f"✅ Komentar ID #{c_id} berhasil divalidasi!")
            st.rerun()

st.markdown("---")
st.subheader("💾 Simpan Massal Semua Komentar")

if st.button("✔ Simpan Seluruh Validasi Halaman Ini", type="primary", use_container_width=True):
    conn = get_connection()
    cursor = conn.cursor()
    for id_row, label in validasi_massal.items():
        cursor.execute("UPDATE hasil_prediksi SET label_benar=? WHERE id=?", (label, id_row))
    conn.commit()
    cursor.close()
    conn.close()

    st.success("🎉 Seluruh validasi berhasil disimpan ke database!")
    st.rerun()
