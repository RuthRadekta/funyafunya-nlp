# File: database/db_connection.py

import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st

# Konfigurasi XAMPP Default
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = '' # Kosongkan jika password root XAMPP Anda default
DB_NAME = 'db_funyafunya'

@st.cache_resource
def get_mysql_connection():
    """Membuat koneksi mentah MySQL (bagus untuk Insert/Update data)"""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error saat menyambungkan ke MySQL: {e}")
        return None

@st.cache_resource
def get_sqlalchemy_engine():
    """Membuat SQLAlchemy engine (bagus untuk read data ke Pandas DataFrame)"""
    try:
        # Format URI: mysql+mysqlconnector://user:password@host/dbname
        engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
        return engine
    except Exception as e:
        st.error(f"Error saat membuat SQLAlchemy engine: {e}")
        return None

# Fungsi Bantuan untuk UI Streamlit
def fetch_projects():
    """Mengambil daftar proyek sebagai DataFrame Pandas"""
    engine = get_sqlalchemy_engine()
    if engine:
        query = "SELECT * FROM tb_projects WHERE status = 'Active'"
        df = pd.read_sql(query, engine)
        return df
    return pd.DataFrame()