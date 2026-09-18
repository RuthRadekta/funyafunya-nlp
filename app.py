from database.db_connection import fetch_projects
from projects.topic_modelling import run_bertopic, run_lda

import streamlit as st
import pandas as pd

# Konfigurasi dasar halaman (harus di baris pertama)
st.set_page_config(page_title="Funyafunya NLP", page_icon="🐱", layout="wide")

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
        model_choice = st.selectbox("Jenis Model", ["BERTopic (Local)", "LDA (Local)", "GPT-4 (API)", "Gemini (API)"])
        
        st.markdown("---")
        
        st.markdown("##### 📂 Input Data")
        # 1. Menambahkan widget file uploader dengan filter khusus file CSV
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
        text_input = st.text_area("Atau masukkan teks secara manual:")
        
        st.write("")
        run_button = st.button("🚀 Run Project", type="primary", use_container_width=True)

    with right_col:
        st.title(f"{st.session_state.active_project}")
        
        # 2. Logika untuk memproses file yang diunggah
        df = None # Inisialisasi variabel df (DataFrame)
        
        if uploaded_file is not None:
            try:
                # Membaca file CSV yang disimpan sementara di memori Streamlit
                df = pd.read_csv(uploaded_file)
                
                st.success(f"Berhasil memuat: {uploaded_file.name}")
                st.markdown("##### Preview Dataset (5 Baris Pertama)")
                # Menampilkan DataFrame di UI secara interaktif
                st.dataframe(df.head(), use_container_width=True)
                
                # Opsi tambahan: Meminta user memilih kolom mana yang berisi teks untuk di-NLP-kan
                st.session_state.target_columns = st.multiselect(
                    "Pilih kolom teks yang akan dianalisis (bisa lebih dari satu):", 
                    options=df.columns,
                    default=[df.columns[0]] # Otomatis memilih kolom pertama sebagai default
                )
                
            except Exception as e:
                st.error(f"Gagal membaca file CSV. Pastikan formatnya benar. Detail error: {e}")

        st.markdown("---")
        st.markdown("### Hasil Eksekusi")
        
        # Logika ketika tombol Run ditekan
        if run_button:
            if uploaded_file is None and text_input == "":
                st.warning("⚠️ Mohon upload dataset atau masukkan teks terlebih dahulu!")
            else:
                with st.spinner(f'Memproses data menggunakan {model_choice}...'):
                    
                    # Eksekusi khusus jika proyek yang aktif adalah Topic Modelling
                    if st.session_state.active_project == 'Topic Modelling':
                        if df is not None:
                            # Cek apakah user sudah memilih minimal 1 kolom
                            if len(st.session_state.target_columns) == 0:
                                st.error("⚠️ Silakan pilih minimal satu kolom teks untuk dianalisis!")
                            else:
                                # Menggabungkan teks dari kolom yang dipilih dengan spasi antar teks
                                combined_text = df[st.session_state.target_columns].astype(str).agg(' '.join, axis=1)
                                
                                # Hilangkan baris kosong dan konversi ke list
                                docs = combined_text.dropna().tolist()
                            
                            # Cek model apa yang dipilih di dropdown UI
                            if "BERTopic" in model_choice:
                                result = run_bertopic(docs)
                                
                                if result["status"] == "success":
                                    st.success("Proses BERTopic selesai!")
                                    st.markdown("#### Detail Topik")
                                    st.dataframe(result["info_df"], use_container_width=True)
                                    
                                    st.markdown("#### Visualisasi Topik")
                                    st.plotly_chart(result["figure"], use_container_width=True)
                                else:
                                    st.error(f"Terjadi kesalahan saat memproses BERTopic: {result['message']}")
                                    
                            elif "LDA" in model_choice:
                                result = run_lda(docs, n_topics=5)
                                
                                if result["status"] == "success":
                                    st.success("Proses LDA selesai!")
                                    st.markdown("#### Top Words per Topic")
                                    st.dataframe(result["info_df"], use_container_width=True)
                                else:
                                    st.error(f"Terjadi kesalahan saat memproses LDA: {result['message']}")

                    # elif st.session_state.active_project == 'Sentiment Analysis':
                    #     ...

# ==========================================
# 4. MAIN ROUTER
# ==========================================
if st.session_state.current_page == 'Home':
    render_home_page()
else:
    render_project_page()