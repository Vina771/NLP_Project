import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

TEST_QUERIES = [
    ("NASA space shuttle mission astronaut",   "Sci/Tech"),
    ("football world cup championship soccer", "Sports"),
    ("stock market wall street oil price",     "Business"),
    ("election president vote democracy",      "World"),
    ("software programming computer internet", "Sci/Tech"),
    ("tennis grand slam wimbledon",            "Sports"),
    ("bank merger acquisition finance",        "Business"),
    ("war military troops conflict",           "World"),
    ("mobile phone chip processor technology", "Sci/Tech"),
    ("olympic gold medal athlete competition", "Sports"),
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def search(query, train, tfidf_train, vectorizer, top_n=10):
    clean_query  = clean_text(query)
    query_vector = vectorizer.transform([clean_query])
    scores       = cosine_similarity(query_vector, tfidf_train).flatten()
    top_indices  = scores.argsort()[-top_n:][::-1]
    results      = train.iloc[top_indices][['title', 'category']].copy()
    results['score'] = scores[top_indices]
    return results.reset_index(drop=True)


def precision_at_k(query, expected_category, train, tfidf_train, vectorizer, k=10):
    results = search(query, train, tfidf_train, vectorizer, top_n=k)
    return (results['category'] == expected_category).sum() / k


def run():
    print("Chargement des données...")
    train       = pd.read_csv('train_clean.csv')
    tfidf_train = sp.load_npz('tfidf_train.npz')

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print("\n--- Precision@10 par requête ---")
    scores_list = []
    for query, cat in TEST_QUERIES:
        p = precision_at_k(query, cat, train, tfidf_train, vectorizer, k=10)
        scores_list.append({'query': query, 'expected': cat, 'precision@10': round(p, 2)})
        print(f"P@10={p:.2f}  [{cat:10s}]  '{query}'")

    df_eval = pd.DataFrame(scores_list)
    print(f"\nPrecision@10 moyenne : {df_eval['precision@10'].mean():.3f}")

    k_values      = [1, 3, 5, 10, 20]
    avg_prec      = []
    for k in k_values:
        p_list = [precision_at_k(q, c, train, tfidf_train, vectorizer, k) for q, c in TEST_QUERIES]
        avg_prec.append(np.mean(p_list))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_values, avg_prec, marker='o', linewidth=2)
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Precision@K moyenne')
    axes[0].set_title('Precision@K du moteur TF-IDF')
    axes[0].grid(True)

    query  = "NASA space mission"
    qv     = vectorizer.transform([clean_text(query)])
    all_sc = cosine_similarity(qv, tfidf_train).flatten()
    axes[1].hist(all_sc[all_sc > 0], bins=50, edgecolor='black')
    axes[1].set_xlabel('Score cosine')
    axes[1].set_ylabel('Nb documents')
    axes[1].set_title(f"Distribution des scores — '{query}'")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=150)
    plt.show()

    print(f"\nDocuments avec score > 0   : {(all_sc > 0).sum()}")
    print(f"Documents avec score > 0.1 : {(all_sc > 0.1).sum()}")
    print("Graphique sauvegardé : evaluation_plots.png")


if __name__ == '__main__':
    run()
