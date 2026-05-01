import pandas as pd
import scipy.sparse as sp
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def run():
    print("Chargement des données nettoyées...")
    train = pd.read_csv('train_clean.csv')
    test  = pd.read_csv('test_clean.csv')

    print(f"Train : {train.shape} | Test : {test.shape}")

    print("Vectorisation TF-IDF (peut prendre ~2 min)...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    tfidf_train = vectorizer.fit_transform(train['clean_text'])
    tfidf_test  = vectorizer.transform(test['clean_text'])

    print(f"Matrice train : {tfidf_train.shape}")
    print(f"Matrice test  : {tfidf_test.shape}")

    feature_names = vectorizer.get_feature_names_out()
    sample_idx    = 42
    row           = tfidf_train[sample_idx]
    top_indices   = row.toarray()[0].argsort()[-10:][::-1]

    print(f"\nExemple doc #{sample_idx} [{train['category'].iloc[sample_idx]}]")
    print("Titre :", train['title'].iloc[sample_idx])
    print("Top 10 termes TF-IDF :", [(feature_names[i], round(row.toarray()[0][i], 4)) for i in top_indices])

    sp.save_npz('tfidf_train.npz', tfidf_train)
    sp.save_npz('tfidf_test.npz',  tfidf_test)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nSauvegardés : tfidf_train.npz, tfidf_test.npz, vectorizer.pkl")


if __name__ == '__main__':
    run()
