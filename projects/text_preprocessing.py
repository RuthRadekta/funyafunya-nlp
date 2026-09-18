# File: projects/text_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords

# Mengunduh corpus stopwords Indonesia secara otomatis jika belum ada di lokal
try:
    stop_words_id = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_id = set(stopwords.words('indonesian'))

# Anda juga bisa menambahkan custom stopwords spesifik ke dalam set ini
# stop_words_id.update(['nya', 'yg', 'kalo', 'biar'])

def clean_text(text):
    """
    Membersihkan teks dari URL, karakter khusus, spasi berlebih, dan stopwords.
    """
    # Pastikan data berupa string
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing (Ubah ke huruf kecil)
    text = text.lower()
    
    # 2. Hapus URL / Link
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Hapus Email, mention (@), dan hashtag (#) opsional
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) 
    
    # 4. Hapus Karakter Khusus dan Angka (Hanya menyisakan alfabet a-z dan spasi)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 5. Hapus spasi berlebih (multiple whitespaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Hapus Stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words_id]
    
    return " ".join(cleaned_words)