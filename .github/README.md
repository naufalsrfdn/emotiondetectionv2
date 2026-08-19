# GitHub Repository Settings & Ignore Guidelines

Folder ini menyimpan konfigurasi repositori GitHub untuk projek **Emotion Detection IndoBERT**.

## 📌 Pengaturan Penanganan File Model Berat (*Large Files*)

Sesuai konfigurasi `.gitignore`, file bobot model berikut **TIDAK** diunggah ke repositori GitHub untuk menghemat kuota & mencegah error limit file GitHub (max 100MB):

- Folder `model_versions/v1/`, `model_versions/v2/`, dst.
- File bertipe `*.bin`, `*.safetensors`, `*.pt`, `*.pth`.
- Database SQLite lokal `emotion.db`.

### 🧠 Model Default
Saat aplikasi pertama kali di-clone dari GitHub, sistem secara otomatis akan mengunduh model default dari Hugging Face Hub: [`naufalsrfdn/indobert-emotion`](https://huggingface.co/naufalsrfdn/indobert-emotion).
