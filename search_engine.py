import pandas as pd
import scipy.sparse as sp
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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
    results      = train.iloc[top_indices][['title', 'description', 'category']].copy()
    results['score'] = scores[top_indices].round(4)
    return results.reset_index(drop=True)


def run():
    print("Chargement des données...")
    train       = pd.read_csv('train_clean.csv')
    tfidf_train = sp.load_npz('tfidf_train.npz')

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    test_queries = [
        "NASA space exploration mission",
        "stock market crash financial crisis",
        "football world cup championship",
        "election president democracy",
        "apple",
    ]

    for query in test_queries:
        print(f"\n--- Requête : '{query}' ---")
        results = search(query, train, tfidf_train, vectorizer, top_n=5)
        print(results[['title', 'category', 'score']].to_string())

    print("\n--- Distribution catégories par requête (top 10) ---")
    for query in test_queries[:4]:
        res = search(query, train, tfidf_train, vectorizer, top_n=10)
        print(f"'{query}' → {res['category'].value_counts().to_dict()}")


if __name__ == '__main__':
    run()
