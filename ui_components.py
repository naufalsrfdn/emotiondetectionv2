import streamlit as st
import os
import load_model

# ==========================================
# EMOTION COLOR DEFINITIONS
# ==========================================
EMOTION_COLORS = {
    "Marah": {"bg": "rgba(239, 68, 68, 0.15)", "border": "#ef4444", "text": "#fca5a5", "hex": "#ef4444", "icon": "😡"},
    "Sedih": {"bg": "rgba(59, 130, 246, 0.15)", "border": "#3b82f6", "text": "#93c5fd", "hex": "#3b82f6", "icon": "😢"},
    "Senang": {"bg": "rgba(16, 185, 129, 0.15)", "border": "#10b981", "text": "#6ee7b7", "hex": "#10b981", "icon": "😃"},
    "Takut": {"bg": "rgba(139, 92, 246, 0.15)", "border": "#8b5cf6", "text": "#c4b5fd", "hex": "#8b5cf6", "icon": "😨"},
    "Terkejut": {"bg": "rgba(245, 158, 11, 0.15)", "border": "#f59e0b", "text": "#fcd34d", "hex": "#f59e0b", "icon": "😲"},
    "Netral": {"bg": "rgba(100, 116, 139, 0.15)", "border": "#64748b", "text": "#cbd5e1", "hex": "#64748b", "icon": "😐"}
}

def get_emotion_color_map():
    """Mengembalikan peta warna emosi untuk Plotly atau chart."""
    return {k: v["hex"] for k, v in EMOTION_COLORS.items()}

def get_emotion_badge(emotion_label, conf=None):
    """Mengembalikan HTML badge emosi (1-line murni tanpa indentasi untuk mencegah error div)."""
    info = EMOTION_COLORS.get(emotion_label, EMOTION_COLORS["Netral"])
    conf_str = f" ({conf*100:.1f}%)" if conf is not None else ""
    return f'<span style="display:inline-flex; align-items:center; gap:6px; background:{info["bg"]}; border:1px solid {info["border"]}; color:{info["text"]}; padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.95rem; box-shadow:0 2px 8px rgba(0,0,0,0.2);"><span>{info["icon"]}</span><span>{emotion_label}</span><span style="font-size:0.85em; opacity:0.85;">{conf_str}</span></span>'

def apply_custom_styles():
    """Menginjeksi CSS kustom untuk tampilan Glassmorphism & UI Modern."""
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.custom-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
}

.gradient-header {
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc) !important;
}
</style>""", unsafe_allow_html=True)

def render_header(title, subtitle=None, icon="🤖"):
    """Render header halaman yang elegan."""
    sub_html = f'<p style="color: #94a3b8; font-size: 1.05rem; margin-top: 6px;">{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div style="margin-bottom:20px;"><h1 style="margin:0; font-size:2.2rem; display:flex; align-items:center; gap:12px;"><span>{icon}</span><span class="gradient-header">{title}</span></h1>{sub_html}</div>', unsafe_allow_html=True)

def render_model_selector(sidebar=False, key="model_select"):
    """
    Render widget pemilih versi model di HALAMAN UTAMA (bukan sidebar).
    """
    available = load_model.get_available_models()
    _, active_ver = load_model.get_active_model_path()
    
    default_idx = 0
    if active_ver in available:
        default_idx = available.index(active_ver)

    c_sel, c_info = st.columns([0.65, 0.35])

    with c_sel:
        selected_ver = st.selectbox(
            "🎯 Pilih Versi Model IndoBERT (Lokal):",
            options=available,
            index=default_idx,
            help="Pilih versi model lokal yang ingin digunakan.",
            key=key
        )

    with c_info:
        st.markdown(f'<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2)); border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 12px; padding: 10px 16px; margin-top: 4px; text-align: center;"><span style="font-size:0.75rem; color:#a7f3d0; text-transform:uppercase; font-weight:700; letter-spacing:0.5px;">⚡ Model Aktif:</span><br><span style="font-size:1.1rem; font-weight:800; color:#ffffff;">{selected_ver}</span></div>', unsafe_allow_html=True)

    return selected_ver
