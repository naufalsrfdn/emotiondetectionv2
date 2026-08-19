from auth import check_password
check_password()

import streamlit as st
import torch
import os
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from db import get_connection, init_db
from load_model import (
    get_available_models,
    get_latest_model_version,
    MODEL_BASE_DIR,
    ACTIVE_MODEL_FILE
)
from ui_components import (
    apply_custom_styles,
    render_header
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PAGE CONFIG & STYLES
# ===============================
st.set_page_config(
    page_title="Training Ulang Model IndoBERT",
    page_icon="🔁",
    layout="wide"
)

apply_custom_styles()

render_header(
    title="Training Ulang Model IndoBERT (Incremental Learning)",
    subtitle="Pelatihan ulang (fine-tuning) model IndoBERT berbasis data validasi manusia terbaru secara 100% lokal.",
    icon="🔁"
)

label_map = ["Marah", "Sedih", "Senang", "Takut", "Terkejut", "Netral"]
label2id = {l: i for i, l in enumerate(label_map)}
id2label = {i: l for l, i in label2id.items()}

# ===============================
# GET LATEST MODEL VERSION (STRICT REQUIREMENT)
# ===============================
all_available_models = get_available_models()
latest_model_name, BASE_MODEL_PATH = get_latest_model_version()

def set_active_model(name):
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    with open(ACTIVE_MODEL_FILE, "w") as f:
        f.write(name)

# ===============================
# HEADER INFO METRICS
# ===============================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 Model Dasar Training (Lokal)</div>
            <div class="metric-value" style="color: #38bdf8;">{latest_model_name}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📦 Total Versi Model Lokal</div>
            <div class="metric-value">{len(all_available_models):,}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔒 Mode Training</div>
            <div class="metric-value" style="color: #a7f3d0; font-size:1.2rem; margin-top:4px;">100% Offline Lokal</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.info(f"🔒 **Skenario Offline**: Fitur Training Ulang secara otomatis menggunakan versi model lokal **terbaru (`{latest_model_name}`)** sebagai dasar pelatihan ulang (*incremental learning*).")

# ===============================
# LOAD DATA VALIDASI
# ===============================
def load_validation_data():
    init_db()
    conn = get_connection()
    query = """
        SELECT id, komentar, emosi, label_benar
        FROM hasil_prediksi
        WHERE label_benar IS NOT NULL
        ORDER BY id DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df_val = load_validation_data()

if df_val.empty:
    st.warning("⚠️ Belum ada data validasi manusia untuk pelatihan ulang model. Silakan lakukan validasi di menu **Validasi Label Emosi** terlebih dahulu.")
    st.stop()

# ===============================
# KONFIGURASI TRAINING
# ===============================
st.markdown("""
    <div class="custom-card">
        <h3 style="margin-top:0; font-size:1.15rem; color:#f8fafc;">⚙️ Konfigurasi Training</h3>
    </div>
""", unsafe_allow_html=True)

jumlah_data = st.slider(
    "Gunakan berapa data validasi TERBARU?",
    min_value=min(5, len(df_val)),
    max_value=len(df_val),
    value=min(50, len(df_val)),
    step=5
)

df_train = df_val.head(jumlah_data)

st.markdown(f"✏️ Data yang digunakan untuk training & evaluasi: **{len(df_train)}** sampel validasi")

with st.expander("📋 Lihat Preview Data Validasi"):
    st.dataframe(df_train, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# EXECUTE TRAINING
# ===============================
if st.button("🔥 Mulai Training Ulang Model", type="primary", use_container_width=True):

    progress = st.progress(0)
    status = st.empty()

    # 1. Prepare Dataset
    status.info("Menyiapkan dataset validasi...")
    texts = df_train["komentar"].tolist()
    labels = [label2id[l] for l in df_train["label_benar"]]

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })
    progress.progress(25)

    # 2. Load Base Model (100% Local)
    status.info(f"Memuat model dasar lokal: {latest_model_name}...")
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True
    )
    progress.progress(50)

    # 3. Output New Version Path
    v_nums = [int(m[1:]) for m in all_available_models if m.startswith("v") and m[1:].isdigit()]
    last_version = max(v_nums) if v_nums else 0
    new_version = f"v{last_version + 1}"
    output_dir = os.path.join(MODEL_BASE_DIR, new_version)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Training Config
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # 5. Train
    status.info("🔥 Fine-tuning model IndoBERT sedang berjalan...")
    trainer.train()
    progress.progress(80)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    progress.progress(100)
    status.success("🎉 Fine-tuning selesai dengan sukses!")

    st.success(f"📁 Versi model baru berhasil disimpan di folder lokal: **{new_version}** (`model_versions/{new_version}`)")

    # 6. Evaluation Metrics
    st.markdown("### 📊 Evaluasi Model Baru")
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            out = model(**inp)
            preds.append(torch.argmax(out.logits, dim=1).item())

    y_true = labels
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{acc*100:.2f}%")
    mc2.metric("Precision", f"{prec:.4f}")
    mc3.metric("Recall", f"{rec:.4f}")
    mc4.metric("F1-Score", f"{f1:.4f}")

    # Confusion Matrix
    st.markdown("### 📈 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_map))))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=label_map,
        yticklabels=label_map,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix ({new_version})")
    st.pyplot(fig)

    # Set Active Model Option
    if st.checkbox(f"Jadikan model versi {new_version} ini sebagai model aktif secara default", value=True):
        set_active_model(new_version)
        st.success(f"🟢 Model aktif sistem sekarang diset ke **{new_version}**.")
