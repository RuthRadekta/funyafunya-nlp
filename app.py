from database.db_connection import fetch_projects
from projects.topic_modelling import run_bertopic, run_lda
from database.db_connection import fetch_projects, init_db
from projects.text_preprocessing import clean_text
from projects.sentiment_analysis import run_local_sentiment, run_api_sentiment

import streamlit as st
import pandas as pd

# Konfigurasi dasar halaman (harus di baris pertama)
st.set_page_config(page_title="Funyafunya NLP", page_icon="🐱", layout="wide")
init_db()

# ==========================================
# 1. STATE MANAGEMENT (Fungsi Navigasi)
# ==========================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'
if 'active_project' not in st.session_state:
    st.session_state.active_project = ''

def go_to_project(project_name):
    st.session_state.current_page = 'Project'
    st.session_state.active_project = project_name

def go_home():
    st.session_state.current_page = 'Home'
    st.session_state.active_project = ''

# ==========================================
# 2. DIALOGS (POP-UP INFORMASI)
# ==========================================
@st.dialog("📚 Informasi Datasets")
def show_datasets_info():
    st.write("Kriteria dataset yang dapat digunakan:")
    st.markdown(f"**For Topic Modeling: CSV Format File**")
    st.caption("What is a CSV file? CSV (Comma-Separated Values) adalah format file teks yang digunakan untuk menyimpan data tabular, di mana setiap baris mewakili satu record dan setiap kolom dipisahkan oleh koma. Format ini sangat umum digunakan untuk pertukaran data antara aplikasi yang berbeda.")
    # Contoh konten panjang agar bisa di-scroll
    # for i in range(1):
    #     st.markdown(f"**Dataset {i}: CSV**")
        # st.caption("Berisi 10,000 ulasan produk untuk analisis sentimen.")
        # st.write("---")

@st.dialog("🧠 Informasi Models")
def show_models_info():
    st.write("Daftar algoritma dan LLM yang tersedia:")
    st.info("Local: BERTopic, Latent Dirichlet Allocation (LDA)")
    st.success("API: OpenAI GPT-4, Google Gemini Pro")

@st.dialog("🔌 Status API")
def show_api_info():
    st.write("Status penggunaan kredensial API LLM dalam sistem:")
    st.progress(60, text="OpenAI Quota (60%)")
    st.progress(20, text="Gemini Quota (20%)")

@st.dialog("❓ Help")
def show_help_info():
    st.write(f"**What is Funyafunya NLP?**")
    st.write("Funyafunya NLP adalah aplikasi untuk melakukan analisis teks menggunakan teknik Natural Language Processing (NLP).")

@st.dialog("❓ What is this?")
def show_what_info(project_name):
    st.write(f"Penjelasan teoritis mengenai **{project_name}**.")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    # Spacer untuk simulasi scroll
    st.markdown("<br><br><br><br><br><br><br><br>Akhir penjelasan.", unsafe_allow_html=True)

@st.dialog("🛠️ How to use")
def show_how_info():
    st.write("Langkah-langkah penggunaan:")
    st.markdown("""
    1. Pilih model pada dropdown di kiri atas.
    2. Upload file CSV.
    3. Pilih kolom yang berisi teks (bisa lebih dari satu).
    4. Tekan tombol **Run Project**.
    5. Tunggu hingga hasil dan visualisasi muncul.
    """)

# ==========================================
# 3. SIDEBAR COMPONENTS
# ==========================================
def render_home_sidebar():
    with st.sidebar:
        st.markdown("---") 
        st.markdown("### Information & Help")
        
        # Mengubah teks menjadi tombol yang memanggil pop-up
        if st.button("📚 Datasets", use_container_width=True):
            show_datasets_info()
            
        if st.button("🧠 Models", use_container_width=True):
            show_models_info()
            
        if st.button("🔌 API Status", use_container_width=True):
            show_api_info()
            
        if st.button("❓ Help", use_container_width=True):
            # st.info("Panduan aplikasi Funyafunya")
            show_help_info()
        
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 24px;'>
            🌐 &nbsp; 🐦 &nbsp; 💼
            <br>
            <span style='font-size: 16px;'>🐱 Funyafunya</span>
        </div>
        """, unsafe_allow_html=True)

def render_project_sidebar():
    with st.sidebar:
        st.markdown("### Menu Proyek")
        
        # Tombol Pop-up
        if st.button("❓ What?", use_container_width=True):
            show_what_info(st.session_state.active_project)
            
        if st.button("🛠️ How?", use_container_width=True):
            show_how_info()
            
        st.markdown("---")
        # Tombol kembali ke halaman utama (Start)
        st.button("🏠 Start (Home)", on_click=go_home, use_container_width=True)

# ==========================================
# 3. PAGE LAYOUTS
# ==========================================
def render_home_page():
    render_home_sidebar()
    
    # Baris Judul & Dark/Light Mode
    col_title, col_theme = st.columns([4, 1])
    with col_title:
        st.title("Funyafunya 🐱")
    with col_theme:
        st.write("") 
        st.button("🌓 Toggle Theme", use_container_width=True)

    st.markdown("### 🚀 Choose a NLP Project")
    st.write("")
    
    # Mengambil data dari Database MySQL
    df_projects = fetch_projects()
    
    if df_projects.empty:
        st.warning("⚠️ Belum ada proyek yang aktif. Silakan tambahkan data di phpMyAdmin.")
    else:
        # Logika Grid Dinamis (Maksimal 4 kolom per baris)
        chunk_size = 4
        
        # Looping setiap 4 item untuk membuat baris baru
        for i in range(0, len(df_projects), chunk_size):
            cols = st.columns(chunk_size)
            chunk = df_projects.iloc[i:i+chunk_size]
            
            # Looping item di dalam chunk tersebut untuk diisi ke dalam kolom
            for j, (index, row) in enumerate(chunk.iterrows()):
                with cols[j]:
                    # Tombol menggunakan data icon_name dan project_name dari DB
                    st.button(
                        f"{row['icon_name']} {row['project_name']}", 
                        key=f"btn_project_{row['id']}", # Unique ID agar Streamlit tidak error jika nama tombol sama
                        on_click=go_to_project, 
                        args=(row['project_name'],), 
                        use_container_width=True
                    )
            
            st.write("") # Spacer antar baris

    # Footer Halaman Utama
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>© 2026 Funyafunya NLP Web App by RuthRadekta. All rights reserved.</p>", unsafe_allow_html=True)

def render_project_page():
    render_project_sidebar()
    
    # Layout 2 Kolom (Kiri untuk Input, Kanan untuk Hasil)
    left_col, right_col = st.columns([1, 2], gap="large")
    
    with left_col:
        st.markdown("##### ⚙️ Konfigurasi Model")
        model_choice = st.selectbox("Jenis Model", ["BERTopic (Local)", "LDA (Local)", "RoBERTa (Local)", "GPT-4 (API)", "Gemini (API)"])
        
        # ==========================================
        # 1. UI KHUSUS PER PROYEK (Hanya memunculkan widget/input)
        # ==========================================
        if st.session_state.active_project == 'Topic Modelling':
            st.markdown("##### 🎛️ Hyperparameters")
            st.session_state.num_topics = st.slider("Jumlah Topik (K)", 2, 30, 5)
            st.session_state.random_seed = st.number_input("Random Seed", 0, 9999, 42)
            st.session_state.apply_cleaning = st.checkbox("Terapkan Text Cleaning", value=True)
            
        elif st.session_state.active_project == 'Sentiment Analysis':
            st.markdown("##### 🔑 Konfigurasi LLM")
            if "API" in model_choice:
                st.session_state.api_key = st.text_input("Masukkan API Key", type="password")
            else:
                st.info("💡 Anda menggunakan model Lokal. Tidak perlu API Key.")

        # ==========================================
        # 2. INPUT DATA UMUM (Harus di luar if-elif agar muncul di semua proyek)
        # ==========================================
        st.markdown("---")
        st.markdown("##### 📂 Input Data")
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
        text_input = st.text_area("Atau masukkan teks secara manual:")
        
        st.write("")
        run_button = st.button("🚀 Run Project", type="primary", use_container_width=True)

    with right_col:
        st.title(f"{st.session_state.active_project}")
        
        # ==========================================
        # 3. PREVIEW DATA (Berlaku untuk semua proyek)
        # ==========================================
        df = None
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Berhasil memuat: {uploaded_file.name}")
            st.dataframe(df.head(), use_container_width=True)
            st.session_state.target_columns = st.multiselect("Pilih kolom teks:", df.columns, default=[df.columns[0]])
            
        st.markdown("---")
        st.markdown("### Hasil Eksekusi")
        
        # ==========================================
        # 4. EKSEKUSI MODEL (Berjalan HANYA jika tombol ditekan)
        # ==========================================
        if run_button:
            if uploaded_file is None and text_input == "":
                st.warning("⚠️ Mohon upload dataset atau masukkan teks terlebih dahulu!")
            else:
                with st.spinner(f'Memproses data menggunakan {model_choice}...'):

                    # ---- CABANG LOGIKA EKSEKUSI PROYEK ----
                    if st.session_state.active_project == 'Topic Modelling':
                        if df is not None and len(st.session_state.target_columns) > 0:
                            # 1. Gabungkan kolom ke dalam Pandas Series
                            combined_series = df[st.session_state.target_columns].astype(str).agg(' '.join, axis=1)
                            
                            # 2. Proses Cleaning langsung di dalam Series
                            if st.session_state.apply_cleaning:
                                with st.spinner('Membersihkan teks (Hapus URL, Tanda Baca, & Stopwords)...'):
                                    # apply() akan menjalankan fungsi clean_text ke setiap baris
                                    cleaned_series = combined_series.apply(clean_text)
                            else:
                                cleaned_series = combined_series
                                
                            # 3. Buat "Mask" untuk mendeteksi baris mana yang tidak kosong
                            valid_mask = cleaned_series.str.strip() != ""
                            
                            # 4. Ambil teks yang valid saja sebagai list untuk dimasukkan ke model
                            docs = cleaned_series[valid_mask].tolist()
                            
                            if len(docs) == 0:
                                st.error("Semua teks menjadi kosong setelah di-preprocessing. Coba nonaktifkan opsi Text Cleaning.")
                                st.stop()
                            
                            # Cek model yang dipilih
                            if "BERTopic" in model_choice:
                                st.info(f"Menjalankan BERTopic dengan K={st.session_state.num_topics} dan Seed={st.session_state.random_seed}")
                                # Mengirim argumen dari UI ke script
                                result = run_bertopic(
                                    docs, 
                                    n_topics=st.session_state.num_topics, 
                                    seed=st.session_state.random_seed
                                )
                                
                                if result["status"] == "success":
                                    st.success("Proses BERTopic selesai!")
                                    st.dataframe(result["info_df"], use_container_width=True)
                                    st.plotly_chart(result["figure"], use_container_width=True)
                                else:
                                    st.error(f"Error: {result['message']}")
                                    
                            elif "LDA" in model_choice:
                                st.info(f"Menjalankan LDA dengan K={st.session_state.num_topics} dan Seed={st.session_state.random_seed}")
                                # Mengirim argumen dari UI ke script
                                result = run_lda(
                                    docs, 
                                    n_topics=st.session_state.num_topics, 
                                    seed=st.session_state.random_seed
                                )
                                
                                if result["status"] == "success":
                                    st.success(f"Proses {model_choice} selesai!")
                                    
                                    # 1. Menampilkan Metrik Evaluasi dengan st.metric
                                    st.markdown("#### 📊 Metrik Evaluasi")
                                    col_m1, col_m2 = st.columns(2)
                                    with col_m1:
                                        # NPMI biasanya berkisar -1 hingga 1. 
                                        st.metric(label="NPMI (Coherence)", value=f"{result['metrics']['npmi']:.4f}")
                                    with col_m2:
                                        st.metric(label="Topic Diversity", value=f"{result['metrics']['diversity']:.4f}")
                                    
                                    # 2. Menampilkan Tabel dan Grafik
                                    st.markdown("#### Detail Topik")
                                    st.dataframe(result["info_df"], use_container_width=True)
                                    
                                    if "figure" in result:
                                        st.plotly_chart(result["figure"], use_container_width=True)
                                    
                                    # 3. Fitur Download CSV (Menggabungkan topik ke dataframe asli)
                                    st.markdown("---")
                                    st.markdown("#### 💾 Ekspor Hasil")
                                    
                                    # Cek apakah panjang dokumen setelah cleaning sama dengan df asli 
                                    # (Jika Anda mendrop baris kosong di proses sebelumnya, lakukan filter df agar selaras)
                                    # Untuk penyederhanaan, kita asumsikan jumlah baris sama.
                                    df_result = df.copy()
                                    # Mengambil jumlah topik sebanyak dokumen yang diproses
                                    df_result['Assigned_Topic'] = -1
                                    # df_result['Assigned_Topic'] = result["assigned_topics"]
                                    df_result.loc[valid_mask, 'Assigned_Topic'] = result["assigned_topics"]
                                    
                                    # Konversi DataFrame ke format CSV
                                    csv_data = df_result.to_csv(index=False).encode('utf-8')
                                    
                                    st.download_button(
                                        label="📥 Download Dataset + Prediksi Topik (CSV)",
                                        data=csv_data,
                                        file_name=f"topic_modelling_result_{model_choice}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                else:
                                    st.error(f"Error: {result['message']}")

                    # --- Sentiment Analysis Logic ---
                    elif st.session_state.active_project == 'Sentiment Analysis':
                        if df is not None and len(st.session_state.target_columns) > 0:
                            combined_series = df[st.session_state.target_columns].astype(str).agg(' '.join, axis=1)
                            valid_mask = combined_series.str.strip() != ""
                            docs = combined_series[valid_mask].tolist()
                            
                            if "Local" in model_choice:
                                result = run_local_sentiment(docs)
                            else:
                                result = run_api_sentiment(docs, model_choice, st.session_state.get('api_key', ''))
                                
                            if result["status"] == "success":
                                st.success("Analisis Sentimen selesai!")
                                st.plotly_chart(result["figure"], use_container_width=True)
                                
                                df_result = df.copy()
                                df_result['Sentiment_Label'] = "Unknown"
                                df_result.loc[valid_mask, 'Sentiment_Label'] = result["labels"]
                                
                                st.dataframe(df_result, use_container_width=True)
                            else:
                                st.error(result["message"])
# ==========================================
# 4. MAIN ROUTER
# ==========================================
if st.session_state.current_page == 'Home':
    render_home_page()
else:
    render_project_page()