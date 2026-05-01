import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

label_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def run():
    print("Chargement des données...")
    train = pd.read_csv('train.csv', header=0, names=['label', 'title', 'description'])
    test  = pd.read_csv('test.csv',  header=0, names=['label', 'title', 'description'])

    print(f"Train : {train.shape} | Test : {test.shape}")

    train['category'] = train['label'].map(label_map)
    test['category']  = test['label'].map(label_map)

    train['text'] = train['title'].fillna('') + ' ' + train['description'].fillna('')
    test['text']  = test['title'].fillna('')  + ' ' + test['description'].fillna('')

    print("Nettoyage du texte (peut prendre ~1 min)...")
    train['clean_text'] = train['text'].apply(clean_text)
    test['clean_text']  = test['text'].apply(clean_text)

    train = train.dropna(subset=['clean_text'])
    test  = test.dropna(subset=['clean_text'])

    train.to_csv('train_clean.csv', index=False)
    test.to_csv('test_clean.csv',   index=False)

    print("Exemple avant :", train['text'].iloc[0][:150])
    print("Exemple après :", train['clean_text'].iloc[0][:150])
    print("\nDistribution des catégories (train) :")
    print(train['category'].value_counts())
    print("\nFichiers sauvegardés : train_clean.csv, test_clean.csv")


if __name__ == '__main__':
    run()
