import streamlit as st
import torch
import pandas as pd

from db import init_db, simpan_prediksi
from load_model import load_model
from ui_components import (
    apply_custom_styles,
    render_header,
    render_model_selector,
    get_emotion_badge,
    EMOTION_COLORS
)

# ===============================
# PAGE CONFIG & STYLES
# ===============================
st.set_page_config(
    page_title="Deteksi Emosi Komentar - IndoBERT",
    page_icon="🤫",
    layout="wide"
)

apply_custom_styles()
init_db()

label_map = ["Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]

# ===============================
# HEADER
# ===============================
render_header(
    title="Deteksi Emosi Komentar YouTube",
    subtitle="Analisis emosi berbasis Transformer IndoBERT dengan pemodelan 6 kategori emosi.",
    icon="🤫"
)

# ===============================
# MAIN PAGE MODEL SELECTOR
# ===============================
selected_version = render_model_selector(sidebar=False, key="app_model_select")

# Pembatas Garis Tipis yang Elegan
st.markdown('<hr style="border:0; border-top: 1px solid rgba(255,255,255,0.1); margin: 16px 0 24px 0;">', unsafe_allow_html=True)

# ===============================
# LOAD SELECTED MODEL
# ===============================
tokenizer, model, model_name = load_model(selected_version)

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_all_probs(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    idx = int(torch.argmax(probs))

    prob_dict = {label_map[i]: float(probs[i]) for i in range(len(label_map))}
    return label_map[idx], float(probs[idx]), prob_dict


# ===============================
# MAIN LAYOUT
# ===============================
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="custom-card"><h3 style="margin-top:0; margin-bottom:8px; font-size:1.2rem; color:#f8fafc;">📝 Masukkan Komentar</h3><p style="color:#94a3b8; font-size:0.9rem; margin-bottom:12px;">Tuliskan kalimat atau komentar bahasa Indonesia untuk dianalisis emosinya secara real-time.</p></div>', unsafe_allow_html=True)

    text = st.text_area(
        "Teks komentar:",
        placeholder="Contoh: Diskusi mantulll, penjelasannya sangat mendalam dan membantu banget!",
        height=140,
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 1])
    with col_btn:
        btn_predict = st.button("🔍 Analisis Emosi", type="primary", use_container_width=True)

with col_result:
    if btn_predict:
        if not text.strip():
            st.warning("⚠️ Komentar tidak boleh kosong.")
        else:
            with st.spinner("🧠 Menganalisis emosi dengan model IndoBERT..."):
                top_label, top_conf, all_probs = predict_all_probs(text)

            simpan_prediksi(text, top_label, top_conf)

            badge_html = get_emotion_badge(top_label, top_conf)
            border_c = EMOTION_COLORS.get(top_label, {}).get('hex', '#6366f1')

            st.markdown(f'<div class="custom-card" style="border-left: 4px solid {border_c};"><div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase; font-weight:700; letter-spacing:0.5px;">Hasil Prediksi Utama</div><div style="margin-top:10px; margin-bottom:12px;">{badge_html}</div><div style="font-size:0.9rem; color:#cbd5e1;">Model <b>{model_name}</b> yakin sebesar <b>{top_conf*100:.2f}%</b> bahwa komentar ini beremosi <b>{top_label}</b>.</div></div>', unsafe_allow_html=True)

            st.markdown('<h4 style="font-size:1rem; color:#e2e8f0; margin-bottom:12px;">📊 Rincian Probabilitas 6 Emosi:</h4>', unsafe_allow_html=True)

            for lbl in label_map:
                p_val = all_probs[lbl]
                c_info = EMOTION_COLORS.get(lbl, {})
                c_hex = c_info.get("hex", "#64748b")
                
                col_lbl, col_bar = st.columns([0.35, 0.65])
                with col_lbl:
                    st.markdown(f"<span style='color:{c_hex}; font-weight:600;'>{c_info.get('icon','')} {lbl}</span>", unsafe_allow_html=True)
                with col_bar:
                    st.progress(p_val)
                    st.caption(f"{p_val*100:.2f}%")
    else:
        st.markdown('<div class="custom-card" style="text-align: center; padding: 36px 20px;"><div style="font-size: 3rem; margin-bottom: 8px;">🎯</div><h4 style="color: #cbd5e1; margin-bottom: 6px;">Siap Melakukan Analisis</h4><p style="color: #64748b; font-size: 0.9rem;">Ketik komentar di sebelah kiri lalu klik tombol <b>Analisis Emosi</b> untuk melihat prediksi model.</p></div>', unsafe_allow_html=True)