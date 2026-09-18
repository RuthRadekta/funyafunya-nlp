import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

# Tentukan lokasi file database (di dalam folder database/)
DB_PATH = os.path.join(os.path.dirname(__file__), 'funyafunya.db')

@st.cache_resource
def get_sqlalchemy_engine():
    """Membuat koneksi ke file SQLite lokal"""
    # Akan otomatis membuat file funyafunya.db jika belum ada
    engine = create_engine(f'sqlite:///{DB_PATH}')
    return engine

def init_db():
    """Fungsi untuk membuat tabel dan data awal (Pengganti phpMyAdmin)"""
    engine = get_sqlalchemy_engine()
    
    with engine.begin() as conn:
        # Cek apakah tabel tb_projects sudah ada
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='tb_projects'")).fetchone()
        
        if not result:
            # 1. Buat Tabel Projects
            conn.execute(text("""
                CREATE TABLE tb_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT NOT NULL,
                    icon_name TEXT,
                    description TEXT,
                    status TEXT DEFAULT 'Active'
                )
            """))
            
            # 2. Buat Tabel Models
            conn.execute(text("""
                CREATE TABLE tb_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    model_name TEXT NOT NULL,
                    model_type TEXT,
                    FOREIGN KEY (project_id) REFERENCES tb_projects(id)
                )
            """))
            
            # 3. Buat Tabel Datasets
            conn.execute(text("""
                CREATE TABLE tb_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    description TEXT,
                    source_link TEXT
                )
            """))
            
            # 4. Insert Data Awal
            conn.execute(text("""
                INSERT INTO tb_projects (project_name, icon_name, description) VALUES 
                ('Topic Modelling', '📊', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Sentiment Analysis', '😃', 'Mengklasifikasikan polaritas emosi teks (positif, netral, negatif).'),
                ('Prediction', '📈', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Fraud Detection', '🛡️', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Content Summarization', '📝', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Agent / Chat Bot', '🤖', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Python Code Generator', '💻', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.'),
                ('Image Generator', '🎨', 'Mengekstraksi topik tersembunyi dari sekumpulan dokumen teks.')
            """))
            
            st.success("Database SQLite berhasil diinisialisasi!")

def fetch_projects():
    """Mengambil daftar proyek sebagai DataFrame Pandas"""
    engine = get_sqlalchemy_engine()
    query = "SELECT * FROM tb_projects WHERE status = 'Active'"
    df = pd.read_sql(query, engine)
    return df