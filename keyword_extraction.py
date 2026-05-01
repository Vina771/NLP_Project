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


def extract_keywords(doc_index, tfidf_train, feature_names, top_n=8):
    row     = tfidf_train[doc_index]
    scores  = row.toarray()[0]
    top_idx = scores.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(scores[i], 4)) for i in top_idx if scores[i] > 0]


def search_with_keywords(query, train, tfidf_train, vectorizer, feature_names, top_n=10):
    clean_query  = clean_text(query)
    query_vector = vectorizer.transform([clean_query])
    scores       = cosine_similarity(query_vector, tfidf_train).flatten()
    top_indices  = scores.argsort()[-top_n:][::-1]

    rows = []
    for i in top_indices:
        kw = [k for k, _ in extract_keywords(i, tfidf_train, feature_names, top_n=5)]
        rows.append({
            'title'      : train['title'].iloc[i],
            'description': train['description'].iloc[i],
            'category'   : train['category'].iloc[i],
            'score'      : round(float(scores[i]), 4),
            'keywords'   : ', '.join(kw)
        })
    return pd.DataFrame(rows)


def run():
    print("Chargement des données...")
    train       = pd.read_csv('train_clean.csv')
    tfidf_train = sp.load_npz('tfidf_train.npz')

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()

    print("\nExtraction de mots-clés sur quelques documents :")
    for idx in [0, 100, 500, 1000]:
        print(f"\nDoc #{idx} [{train['category'].iloc[idx]}] — {train['title'].iloc[idx]}")
        print("Mots-clés :", extract_keywords(idx, tfidf_train, feature_names))

    print("\nTest de recherche avec mots-clés :")
    results = search_with_keywords("NASA space exploration", train, tfidf_train, vectorizer, feature_names)
    print(results[['title', 'category', 'score', 'keywords']].to_string())

    print("\nSauvegarde de search_data.pkl...")
    search_data = {
        'train'        : train,
        'tfidf_train'  : tfidf_train,
        'vectorizer'   : vectorizer,
        'feature_names': feature_names
    }
    with open('search_data.pkl', 'wb') as f:
        pickle.dump(search_data, f)

    print("Sauvegardé : search_data.pkl")


if __name__ == '__main__':
    run()
