# File: projects/sentiment_analysis.py

import pandas as pd
import plotly.express as px
from transformers import pipeline

def run_local_sentiment(docs, model_name="w11wo/indonesian-roberta-base-sentiment-classifier"):
    """
    Menjalankan klasifikasi sentimen menggunakan model RoBERTa lokal dari Hugging Face.
    """
    try:
        # Inisialisasi pipeline Hugging Face
        # (Akan mengunduh model ~500MB pada eksekusi pertama)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
        
        # Eksekusi prediksi (batasi panjang teks max 512 token untuk RoBERTa)
        truncated_docs = [str(doc)[:512] for doc in docs]
        results = sentiment_pipeline(truncated_docs)
        
        # Ekstrak label dan skor probabilitas (confidence)
        labels = [res['label'] for res in results]
        scores = [res['score'] for res in results]
        
        # Buat visualisasi distribusi sentimen (Pie Chart)
        df_dist = pd.DataFrame({'Sentimen': labels})
        dist_count = df_dist['Sentimen'].value_counts().reset_index()
        dist_count.columns = ['Sentimen', 'Jumlah']
        
        # Warna custom agar intuitif: Positif (Hijau), Negatif (Merah), Netral (Abu-abu)
        color_map = {'positive': '#2Ecb71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        fig_pie = px.pie(
            dist_count, 
            names='Sentimen', 
            values='Jumlah', 
            color='Sentimen',
            color_discrete_map=color_map,
            hole=0.4,
            title="Distribusi Sentimen"
        )
        
        return {
            "status": "success",
            "labels": labels,
            "scores": scores,
            "figure": fig_pie
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_api_sentiment(docs, api_provider, api_key):
    """
    Kerangka untuk menjalankan sentimen menggunakan LLM (OpenAI / Gemini).
    Ini sangat berguna untuk teks panjang, idiom yang kompleks, atau sarkasme.
    """
    if not api_key:
        return {"status": "error", "message": "API Key tidak boleh kosong!"}
    
    # TODO: Implementasi pemanggilan API OpenAI atau Google Gemini (Batch processing)
    # Untuk sementara kita kembalikan mock data
    
    return {
        "status": "error", 
        "message": f"Modul API {api_provider} sedang dalam tahap pengembangan."
    }