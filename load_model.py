import os
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_versions")
ACTIVE_MODEL_FILE = os.path.join(MODEL_BASE_DIR, "active.txt")

def get_available_models():
    """
    Mengembalikan daftar seluruh versi model LOKAL yang ada di folder model_versions/ (misal: v1, v2).
    Model dimuat 100% dari folder lokal.
    """
    models = []
    if os.path.exists(MODEL_BASE_DIR):
        local_versions = sorted([
            d for d in os.listdir(MODEL_BASE_DIR)
            if os.path.isdir(os.path.join(MODEL_BASE_DIR, d)) and not d.startswith(".")
        ], key=lambda x: (int(x[1:]) if x.startswith("v") and x[1:].isdigit() else 0, x))
        models.extend(local_versions)
    
    if not models:
        models = ["v1"]  # Fallback default local version
    return models

def get_active_model_path():
    """Mengembalikan path dan nama versi model LOKAL yang sedang diset aktif."""
    if os.path.exists(ACTIVE_MODEL_FILE):
        try:
            active_ver = open(ACTIVE_MODEL_FILE).read().strip()
            local_path = os.path.join(MODEL_BASE_DIR, active_ver)
            if os.path.isdir(local_path):
                return local_path, active_ver
        except Exception:
            pass
    
    available = get_available_models()
    fallback_ver = available[0]
    return os.path.join(MODEL_BASE_DIR, fallback_ver), fallback_ver

def get_latest_model_version():
    """
    Mengembalikan versi model lokal terbaru (misal: v2 jika ada v1 & v2).
    Digunakan khusus untuk fitur training ulang.
    """
    models = get_available_models()
    local_v = [m for m in models if m.startswith("v") and m[1:].isdigit()]
    if local_v:
        local_v.sort(key=lambda x: int(x[1:]))
        latest = local_v[-1]
        return latest, os.path.join(MODEL_BASE_DIR, latest)
    
    path, name = get_active_model_path()
    return name, path

@st.cache_resource(show_spinner="Memuat model IndoBERT lokal...")
def load_model(version_name=None):
    """
    Memuat tokenizer & model secara 100% LOKAL (local_files_only=True).
    Tidak mengontak atau mengunduh apapun dari huggingface.co.
    """
    if not version_name or version_name == "aktif":
        model_path, model_name = get_active_model_path()
    else:
        local_path = os.path.join(MODEL_BASE_DIR, version_name)
        if os.path.isdir(local_path):
            model_path = local_path
            model_name = version_name
        else:
            model_path, model_name = get_active_model_path()
    
    # 100% LOKAL tanpa koneksi internet ke huggingface.co
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()

    return tokenizer, model, model_name
