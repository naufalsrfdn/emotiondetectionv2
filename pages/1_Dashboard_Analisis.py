import streamlit as st
import pandas as pd
import plotly.express as px
from db import get_connection, init_db
from ui_components import (
    apply_custom_styles,
    render_header,
    get_emotion_color_map,
    EMOTION_COLORS
)

# ======================================
# PAGE CONFIG & STYLES
# ======================================
st.set_page_config(
    page_title="Dashboard Analisis Emosi",
    page_icon="📊",
    layout="wide"
)

apply_custom_styles()

render_header(
    title="Dashboard Analisis Emosi",
    subtitle="Visualisasi statistik distribusi emosi komentar dan metrik validasi manusia.",
    icon="📊"
)

# ======================================
# URUTAN EMOSI (KONSISTEN)
# ======================================
EMOTION_ORDER = ["Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]
emotion_cat_type = pd.CategoricalDtype(categories=EMOTION_ORDER, ordered=True)
color_discrete_map = get_emotion_color_map()

# ======================================
# LOAD DATA
# ======================================
@st.cache_data(ttl=10, show_spinner="Memuat data dari database...")
def load_data():
    init_db()
    conn = get_connection()
    df_result = pd.read_sql("SELECT * FROM hasil_prediksi", conn)
    conn.close()
    return df_result

col_ref, _ = st.columns([0.2, 0.8])
with col_ref:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

df = load_data()

if df.empty:
    st.info("ℹ️ Database masih kosong. Silakan lakukan deteksi emosi atau scraping terlebih dahulu.")
    st.stop()

df = df.dropna(subset=["emosi"])
df["emosi"] = df["emosi"].astype(emotion_cat_type)

total_data = len(df)
df_val = df.dropna(subset=["label_benar"]).copy()
total_validated = len(df_val)

accuracy_val = None
if total_validated > 0:
    correct_count = (df_val["emosi"] == df_val["label_benar"]).sum()
    accuracy_val = (correct_count / total_validated) * 100

# ======================================
# KARTU METRIK UTAMA
# ======================================
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Total Komentar</div>
            <div class="metric-value">{total_data:,}</div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📝 Divalidasi Manusia</div>
            <div class="metric-value">{total_validated:,}</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    acc_text = f"{accuracy_val:.1f}%" if accuracy_val is not None else "N/A"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Akurasi Validasi</div>
            <div class="metric-value" style="color: #10b981;">{acc_text}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================
# GRAFIK DISTRIBUSI EMOSI
# ======================================
col_g1, col_g2 = st.columns(2, gap="medium")

with col_g1:
    st.markdown("### 🤖 Distribusi Emosi (Prediksi Model)")
    
    emotion_counts = (
        df["emosi"]
        .value_counts()
        .reindex(EMOTION_ORDER, fill_value=0)
        .reset_index()
    )
    emotion_counts.columns = ["emosi", "jumlah"]

    fig1 = px.bar(
        emotion_counts,
        x="emosi",
        y="jumlah",
        color="emosi",
        color_discrete_map=color_discrete_map,
        category_orders={"emosi": EMOTION_ORDER},
        text="jumlah"
    )

    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Emosi",
        yaxis_title="Jumlah Komentar",
        showlegend=False,
        height=380
    )
    fig1.update_traces(textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    st.markdown("### 🧑‍🏫 Distribusi Label Benar (Human)")
    
    if not df_val.empty:
        df_val["label_benar"] = df_val["label_benar"].astype(emotion_cat_type)
        label_counts = (
            df_val["label_benar"]
            .value_counts()
            .reindex(EMOTION_ORDER, fill_value=0)
            .reset_index()
        )
        label_counts.columns = ["label_benar", "jumlah"]

        fig2 = px.bar(
            label_counts,
            x="label_benar",
            y="jumlah",
            color="label_benar",
            color_discrete_map=color_discrete_map,
            category_orders={"label_benar": EMOTION_ORDER},
            text="jumlah"
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Label Benar",
            yaxis_title="Jumlah Komentar",
            showlegend=False,
            height=380
        )
        fig2.update_traces(textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("⚠️ Belum ada data yang divalidasi oleh pakar/manusia.")

# ======================================
# AKURASI VALIDASI PIE CHART
# ======================================
if not df_val.empty:
    st.markdown("### 🎯 Rasio Validasi Prediksi Model")
    df_val["status"] = df_val.apply(
        lambda r: "Benar (Sesuai)" if r["emosi"] == r["label_benar"] else "Salah (Dikoreksi)",
        axis=1
    )

    fig3 = px.pie(
        df_val,
        names="status",
        hole=0.4,
        color="status",
        color_discrete_map={"Benar (Sesuai)": "#10b981", "Salah (Dikoreksi)": "#ef4444"}
    )
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)