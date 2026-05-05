import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

st.set_page_config(
    page_title="NLP Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e8e0d5; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; color: #e8e0d5; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: #e8e0d5;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #6b6560;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.accent { color: #c9a96e; }

.result-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c9a96e;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.result-card:hover { border-left-color: #e8e0d5; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: #e8e0d5;
    margin-bottom: 0.3rem;
}
.result-desc {
    font-size: 0.85rem;
    color: #8a8278;
    margin-bottom: 0.6rem;
    line-height: 1.5;
}
.result-meta {
    display: flex;
    gap: 0.8rem;
    align-items: center;
    flex-wrap: wrap;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-world    { background: #1a2535; color: #6b9fd4; }
.badge-sports   { background: #1a2a1a; color: #6db86d; }
.badge-business { background: #2a1f10; color: #c9a96e; }
.badge-scitech  { background: #1f1a2a; color: #a88cd4; }

.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6b6560;
}
.score-bar-bg {
    width: 80px; height: 4px;
    background: #2a2a2a;
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill { height: 100%; background: #c9a96e; border-radius: 2px; }

.kw-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #6b6560;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 0.1rem 0.4rem;
}
.stat-box {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #c9a96e;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6b6560;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.mode-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.mode-tfidf { background: #1a2535; color: #6b9fd4; }
.mode-embed { background: #1f1a2a; color: #a88cd4; }

.info-box {
    background: #1a1a2a;
    border: 1px solid #2a2a3a;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8a84a8;
    margin-bottom: 1rem;
}

.stTextInput > div > div > input {
    background-color: #161616 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 4px !important;
    color: #e8e0d5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #c9a96e !important;
    box-shadow: none !important;
}
.stSelectbox > div > div {
    background-color: #161616 !important;
    border: 1px solid #3a3a3a !important;
    color: #e8e0d5 !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stSidebar"] {
    background-color: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
.no-results {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #6b6560;
    padding: 2rem;
    text-align: center;
    border: 1px dashed #2a2a2a;
    border-radius: 4px;
}
hr { border-color: #1e1e1e !important; }
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {
    'World'   : 'badge-world',
    'Sports'  : 'badge-sports',
    'Business': 'badge-business',
    'Sci/Tech': 'badge-scitech',
}


@st.cache_resource(show_spinner="Chargement TF-IDF...")
def load_tfidf():
    with open('search_data.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Chargement embeddings...")
def load_embeddings():
    with open('embedding_data.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Chargement du modèle sémantique...")
def load_model(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', ' ', text)
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def extract_keywords(doc_index, tfidf_matrix, feature_names, top_n=5):
    row     = tfidf_matrix[doc_index]
    scores  = row.toarray()[0]
    top_idx = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_idx if scores[i] > 0]


def search_tfidf(query, data, top_n=10, category_filter=None):
    train         = data['train']
    tfidf_train   = data['tfidf_train']
    vectorizer    = data['vectorizer']
    feature_names = data['feature_names']

    clean_query = clean_text(query)
    if not clean_query:
        return pd.DataFrame()

    query_vector = vectorizer.transform([clean_query])
    scores       = cosine_similarity(query_vector, tfidf_train).flatten()

    if category_filter and category_filter != 'Toutes':
        mask   = (train['category'] == category_filter).values
        scores = scores * mask

    top_indices = scores.argsort()[-top_n:][::-1]
    top_indices = [i for i in top_indices if scores[i] > 0]

    rows = []
    for i in top_indices:
        kw = extract_keywords(i, tfidf_train, feature_names)
        rows.append({
            'title'      : train['title'].iloc[i],
            'description': str(train['description'].iloc[i])[:300],
            'category'   : train['category'].iloc[i],
            'score'      : round(float(scores[i]), 4),
            'keywords'   : kw,
        })
    return pd.DataFrame(rows)


def search_embeddings(query, emb_data, model, top_n=10, category_filter=None):
    train      = emb_data['train']
    embeddings = emb_data['embeddings']

    q_vec  = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(q_vec, embeddings).flatten()

    if category_filter and category_filter != 'Toutes':
        mask   = (train['category'] == category_filter).values
        scores = scores * mask

    top_indices = scores.argsort()[-top_n:][::-1]
    top_indices = [i for i in top_indices if scores[i] > 0]

    rows = []
    for i in top_indices:
        rows.append({
            'title'      : train['title'].iloc[i],
            'description': str(train['description'].iloc[i])[:300],
            'category'   : train['category'].iloc[i],
            'score'      : round(float(scores[i]), 4),
            'keywords'   : [],
        })
    return pd.DataFrame(rows)


# --- Chargement des données ---
tfidf_ok = False
emb_ok   = False

try:
    tfidf_data = load_tfidf()
    tfidf_ok   = True
except FileNotFoundError:
    pass

try:
    emb_data  = load_embeddings()
    emb_model = load_model(emb_data['model_name'])
    emb_ok    = True
except FileNotFoundError:
    pass

if not tfidf_ok and not emb_ok:
    st.error("Aucun fichier de données trouvé. Lance d'abord les scripts de préparation.")
    st.stop()

# --- Header ---
st.markdown("""
<div style="margin-bottom:0.2rem">
    <span class="hero-title">NLP <span class="accent">Search</span></span>
</div>
<div class="hero-sub">Moteur de recherche intelligent basé sur NLP · AG News · 120 000 documents</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Paramètres")

    engine_options = []
    if tfidf_ok:
        engine_options.append("TF-IDF")
    if emb_ok:
        engine_options.append("Embeddings")

    if len(engine_options) > 1:
        engine = st.radio("Moteur de recherche", engine_options)
    else:
        engine = engine_options[0]
        st.markdown(f"**Moteur :** {engine}")

    top_n = st.slider("Nombre de résultats", min_value=3, max_value=30, value=10, step=1)
    category_filter = st.selectbox(
        "Filtrer par catégorie",
        ['Toutes', 'World', 'Sports', 'Business', 'Sci/Tech']
    )

    st.markdown("---")

    if engine == "TF-IDF":
        st.markdown("""
        <div class="info-box">
            <strong style="color:#6b9fd4">TF-IDF</strong><br>
            Correspondance par termes pondérés.<br>
            Rapide · corpus complet (120k docs)<br>
            Sensible aux mots exacts.
        </div>""", unsafe_allow_html=True)
    elif engine == "Embeddings" and emb_ok:
        n_docs = len(emb_data['train'])
        st.markdown(f"""
        <div class="info-box">
            <strong style="color:#a88cd4">Embeddings</strong><br>
            Similarité sémantique contextuelle.<br>
            Modèle : {emb_data['model_name']}<br>
            Corpus : {n_docs:,} docs
        </div>""", unsafe_allow_html=True)

    st.markdown("### Stats du corpus")
    ref_train = tfidf_data['train'] if tfidf_ok else emb_data['train']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-num">120k</div>
            <div class="stat-label">documents</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-num">50k</div>
            <div class="stat-label">termes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    cat_counts = ref_train['category'].value_counts()
    for cat, count in cat_counts.items():
        badge = BADGE_CLASS.get(cat, '')
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:0.4rem;font-family:'DM Mono',monospace;font-size:0.78rem;">
            <span class="badge {badge}">{cat}</span>
            <span style="color:#6b6560">{count:,}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#6b6560;line-height:1.7">
    TF-IDF · similarité cosine<br>
    unigrammes + bigrammes<br>
    sentence-transformers (bonus)
    </div>""", unsafe_allow_html=True)

# --- Recherche ---
query = st.text_input("", placeholder="Entrez votre requête en anglais...")

if query:
    with st.spinner("Recherche en cours..."):
        if engine == "TF-IDF":
            results  = search_tfidf(query, tfidf_data, top_n=top_n, category_filter=category_filter)
            mode_tag = '<span class="mode-tag mode-tfidf">TF-IDF</span>'
        else:
            results  = search_embeddings(query, emb_data, emb_model, top_n=top_n, category_filter=category_filter)
            mode_tag = '<span class="mode-tag mode-embed">Embeddings</span>'

    if results.empty:
        st.markdown('<div class="no-results">Aucun résultat trouvé pour cette requête.</div>', unsafe_allow_html=True)
    else:
        n        = len(results)
        cat_dist = results['category'].value_counts().to_dict()
        dist_str = ' · '.join([f"{k} {v}" for k, v in cat_dist.items()])

        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#6b6560;
                    margin-bottom:1.2rem;padding-bottom:0.8rem;border-bottom:1px solid #1e1e1e;
                    display:flex;gap:1rem;align-items:center;">
            {mode_tag}
            <span>{n} résultat{'s' if n > 1 else ''} &nbsp;·&nbsp; {dist_str}</span>
        </div>""", unsafe_allow_html=True)

        import html as html_lib
        cards_html = []
        for _, row in results.iterrows():
            badge_cls = BADGE_CLASS.get(row['category'], '')
            score_pct = min(int(row['score'] * 400), 100)
            title     = html_lib.escape(str(row['title']))
            desc      = html_lib.escape(str(row['description']))
            kw_list   = row['keywords'] if isinstance(row['keywords'], list) else []
            kw_pills  = ''.join([f'<span class="kw-pill">{kw}</span>' for kw in kw_list])
            kw_block  = f'<div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-top:0.3rem">{kw_pills}</div>' if kw_pills else ''

            cards_html.append(f'''
            <div class="result-card">
                <div class="result-title">{title}</div>
                <div class="result-desc">{desc}</div>
                <div class="result-meta">
                    <span class="badge {badge_cls}">{row['category']}</span>
                    <div class="score-bar-wrap">
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{score_pct}%"></div>
                        </div>
                        {row['score']}
                    </div>
                </div>
                {kw_block}
            </div>''')

        st.markdown('\n'.join(cards_html), unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="no-results" style="margin-top:2rem">
        Tape une requête pour commencer · exemples :
        <em>space mission</em>, <em>stock market crash</em>, <em>olympic games</em>
    </div>""", unsafe_allow_html=True)
