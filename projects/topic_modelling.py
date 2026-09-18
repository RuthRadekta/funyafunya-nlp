# File: projects/topic_modelling.py

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_bertopic(docs):
    """
    Menjalankan algoritma BERTopic pada sekumpulan dokumen teks.
    """
    try:
        # Inisialisasi model BERTopic
        topic_model = BERTopic(language="indonesian", calculate_probabilities=False)
        
        # Fitting model ke data
        topics, probs = topic_model.fit_transform(docs)
        
        # Mengambil informasi topik
        topic_info = topic_model.get_topic_info()
        
        # Mengambil visualisasi (Barchart) untuk ditampilkan di Streamlit
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        
        # (Opsional) Tempat untuk menambahkan perhitungan Coherence Score / NPMI 
        # menggunakan Gensim / octis di sini jika diperlukan untuk evaluasi.
        
        return {
            "status": "success",
            "info_df": topic_info,
            "figure": fig_barchart
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_lda(docs, n_topics=5):
    """
    Menjalankan algoritma Latent Dirichlet Allocation (LDA) menggunakan scikit-learn.
    """
    try:
        # Preprocessing sederhana: tokenisasi dan count vectorizer
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(docs)
        
        # Inisialisasi dan fitting LDA
        lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=42)
        lda_model.fit(tf)
        
        # Mengekstrak top words per topik
        feature_names = tf_vectorizer.get_feature_names_out()
        topic_results = []
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[: -10 - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_results.append({
                "Topic": f"Topic {topic_idx}",
                "Top Words": ", ".join(top_features)
            })
            
        return {
            "status": "success",
            "info_df": pd.DataFrame(topic_results)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}