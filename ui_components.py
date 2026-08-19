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
    """Mengembalikan HTML badge emosi yang menarik."""
    info = EMOTION_COLORS.get(emotion_label, EMOTION_COLORS["Netral"])
    conf_str = f" ({conf*100:.1f}%)" if conf is not None else ""
    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {info['bg']};
        border: 1px solid {info['border']};
        color: {info['text']};
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    ">
        <span>{info['icon']}</span>
        <span>{emotion_label}</span>
        <span style="font-size: 0.85em; opacity: 0.85;">{conf_str}</span>
    </span>
    """

def apply_custom_styles():
    """Menginjeksi CSS kustom untuk tampilan Glassmorphism & UI Modern."""
    st.markdown("""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Container Card Styling */
        .custom-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .custom-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 14px 28px -4px rgba(0, 0, 0, 0.4);
        }

        /* Metric Box Styling */
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

        /* Gradient Text */
        .gradient-header {
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Model Badge di Sidebar */
        .model-active-badge {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }

        /* Button Styling Enhancement */
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

        /* Progress Bar Accent */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(title, subtitle=None, icon="🤖"):
    """Render header halaman yang elegan."""
    st.markdown(f"""
        <div style="margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.2rem; display: flex; align-items: center; gap: 12px;">
                <span>{icon}</span>
                <span class="gradient-header">{title}</span>
            </h1>
            {f'<p style="color: #94a3b8; font-size: 1.05rem; margin-top: 6px;">{subtitle}</p>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

def render_model_selector(sidebar=True, allow_all=True, key="model_select"):
    """
    Render widget pemilih versi model di Sidebar / Main Page.
    Mendukung pemilihan versi model LOKAL (misal: v1, v2, dll).
    Mengembalikan nama versi model yang dipilih.
    """
    available = load_model.get_available_models()
    _, active_ver = load_model.get_active_model_path()
    
    default_idx = 0
    if active_ver in available:
        default_idx = available.index(active_ver)

    container = st.sidebar if sidebar else st

    container.markdown("### ⚙️ Pengaturan Model Lokal")
    selected_ver = container.selectbox(
        "🎯 Pilih Versi Model:",
        options=available,
        index=default_idx,
        help="Pilih versi model IndoBERT lokal yang ingin digunakan.",
        key=key
    )

    container.markdown(f"""
        <div class="model-active-badge">
            <div style="font-size: 0.75rem; color: #a7f3d0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">
                ⚡ Model Lokal Digunakan
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-top: 2px; word-break: break-all;">
                {selected_ver}
            </div>
        </div>
    """, unsafe_allow_html=True)

    return selected_ver
