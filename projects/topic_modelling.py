# File: projects/topic_modelling.py

import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import untuk evaluasi
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def calculate_topic_diversity(topic_words):
    """
    Menghitung proporsi kata unik di seluruh topik (Nilai 0 sampai 1).
    Semakin mendekati 1, semakin sedikit topik yang tumpang tindih.
    """
    if not topic_words:
        return 0
    unique_words = set(word for topic in topic_words for word in topic)
    total_words = sum(len(topic) for topic in topic_words)
    return len(unique_words) / total_words if total_words > 0 else 0

def get_npmi_score(topic_words, tokenized_docs):
    """Menghitung metrik NPMI menggunakan Gensim"""
    dictionary = Dictionary(tokenized_docs)
    cm = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_npmi')
    return cm.get_coherence()

def run_bertopic(docs, n_topics=None, seed=42):
    try:
        # 1. Eksplisit menggunakan SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            nr_topics=n_topics
        )
        
        # Prediksi Topik per Dokumen
        topics, probs = topic_model.fit_transform(docs)
        topic_info = topic_model.get_topic_info()
        
        # Ekstrak kata-kata untuk evaluasi metrik (Top 10 kata per topik, abaikan topik -1 outlier)
        topic_words = []
        for topic_id in topic_info['Topic']:
            if topic_id != -1: 
                words = [word[0] for word in topic_model.get_topic(topic_id)[:10]]
                topic_words.append(words)
        
        # Hitung Metrik
        tokenized_docs = [doc.split() for doc in docs]
        npmi_score = get_npmi_score(topic_words, tokenized_docs) if topic_words else 0
        diversity_score = calculate_topic_diversity(topic_words)
        
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        
        return {
            "status": "success",
            "info_df": topic_info,
            "figure": fig_barchart,
            "assigned_topics": topics, # Array nomor topik untuk digabung ke DataFrame
            "metrics": {"npmi": npmi_score, "diversity": diversity_score}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def run_lda(docs, n_topics=5, seed=42):
    try:
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(docs)
        
        lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=seed)
        
        # Distribusi probabilitas topik per dokumen
        document_topic_probs = lda_model.fit_transform(tf)
        # Ambil topik dominan per baris dokumen
        assigned_topics = document_topic_probs.argmax(axis=1).tolist()
        
        feature_names = tf_vectorizer.get_feature_names_out()
        topic_results = []
        topic_words = []
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_features_ind = topic.argsort()[: -10 - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words.append(top_features)
            topic_results.append({
                "Topic": topic_idx,
                "Top Words": ", ".join(top_features)
            })
            
        tokenized_docs = [doc.split() for doc in docs]
        npmi_score = get_npmi_score(topic_words, tokenized_docs)
        diversity_score = calculate_topic_diversity(topic_words)
            
        return {
            "status": "success",
            "info_df": pd.DataFrame(topic_results),
            "assigned_topics": assigned_topics,
            "metrics": {"npmi": npmi_score, "diversity": diversity_score}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}