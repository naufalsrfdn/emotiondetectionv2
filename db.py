import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "emotion.db")

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hasil_prediksi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        komentar TEXT,
        emosi TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        label_benar TEXT
    )
    """)

    conn.commit()
    conn.close()


def simpan_prediksi(komentar, emosi, confidence):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO hasil_prediksi (komentar, emosi, confidence)
        VALUES (?, ?, ?)
    """, (komentar, emosi, confidence))

    conn.commit()
    conn.close()


def update_label(id_data, label_benar):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE hasil_prediksi
        SET label_benar = ?
        WHERE id = ?
    """, (label_benar, id_data))

    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, komentar, emosi, confidence, label_benar, created_at
        FROM hasil_prediksi
        ORDER BY id DESC
    """)

    data = cursor.fetchall()
    conn.close()

    return data


def delete_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM hasil_prediksi")

    conn.commit()
    conn.close()