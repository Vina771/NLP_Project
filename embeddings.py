import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'

USE_SUBSET = True
SUBSET_SIZE = 20000


def run():
    print("Chargement des données nettoyées...")
    train = pd.read_csv('train_clean.csv')
    test  = pd.read_csv('test_clean.csv')

    if USE_SUBSET:
        samples = []
        for _, group in train.groupby('category'):
            samples.append(group.sample(min(len(group), SUBSET_SIZE // 4), random_state=42))
        train = pd.concat(samples).reset_index(drop=True)
        print(f"Sous-ensemble stratifié : {len(train)} documents")
        print(train['category'].value_counts())
    else:
        print(f"Corpus complet : {len(train)} documents")

    print(f"\nChargement du modèle '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    texts = (train['title'].fillna('') + ' ' + train['description'].fillna('')).tolist()

    print("Encodage en cours (peut prendre plusieurs minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"\nShape des embeddings : {embeddings.shape}") 

    embedding_data = {
        'train'     : train.reset_index(drop=True),
        'embeddings': embeddings,
        'model_name': MODEL_NAME,
        'subset'    : USE_SUBSET
    }

    with open('embedding_data.pkl', 'wb') as f:
        pickle.dump(embedding_data, f)

    print("Sauvegardé : embedding_data.pkl")

    # Test rapide
    from sklearn.metrics.pairwise import cosine_similarity

    test_queries = [
        "stock market crash financial crisis",
        "NASA space mission astronaut",
        "football world cup championship",
    ]

    print("\n--- Test de recherche sémantique ---")
    for query in test_queries:
        q_vec   = model.encode([query], convert_to_numpy=True)
        scores  = cosine_similarity(q_vec, embeddings).flatten()
        top_idx = scores.argsort()[-5:][::-1]
        print(f"\nRequête : '{query}'")
        for i in top_idx:
            print(f"  [{train['category'].iloc[i]:8s}] {scores[i]:.4f} — {train['title'].iloc[i]}")


if __name__ == '__main__':
    run()
