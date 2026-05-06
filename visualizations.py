import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
from collections import Counter


def graphique_1_distribution_categories(train):
    """Bar chart de la distribution des catégories."""
    counts = train['category'].value_counts()
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=11)

    ax.set_title('Distribution des catégories — AG News (train)', fontsize=13)
    ax.set_xlabel('Catégorie')
    ax.set_ylabel('Nombre de documents')
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('graph_distribution_categories.png', dpi=150)
    plt.show()
    print("Sauvegardé : graph_distribution_categories.png")


def graphique_2_top_mots_par_categorie(train, tfidf_train, vectorizer):
    """Top 15 mots TF-IDF moyens par catégorie."""
    feature_names = vectorizer.get_feature_names_out()
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        mask = (train['category'] == cat).values
        cat_matrix = tfidf_train[mask]

        # Score TF-IDF moyen par terme pour cette catégorie
        mean_scores = cat_matrix.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-15:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        top_scores = [mean_scores[j] for j in top_idx]

        axes[i].barh(top_words[::-1], top_scores[::-1],
                     color=colors[i], edgecolor='black', linewidth=0.4)
        axes[i].set_title(f'Top 15 termes — {cat}', fontsize=12)
        axes[i].set_xlabel('Score TF-IDF moyen')
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle('Termes les plus représentatifs par catégorie', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('graph_top_mots_categories.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sauvegardé : graph_top_mots_categories.png")


def graphique_3_longueur_articles(train):
    """Longueur moyenne des articles par catégorie."""
    train = train.copy()
    train['nb_mots'] = train['clean_text'].fillna('').apply(lambda x: len(x.split()))

    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Graphique gauche : moyenne + écart type
    means = [train[train['category'] == cat]['nb_mots'].mean() for cat in categories]
    stds  = [train[train['category'] == cat]['nb_mots'].std()  for cat in categories]

    bars = axes[0].bar(categories, means, yerr=stds, color=colors,
                       edgecolor='black', linewidth=0.5, capsize=5)
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    axes[0].set_title('Longueur moyenne des articles (après nettoyage)', fontsize=12)
    axes[0].set_ylabel('Nombre de mots')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Graphique droite : boxplot
    data = [train[train['category'] == cat]['nb_mots'].values for cat in categories]
    bp = axes[1].boxplot(data, labels=categories, patch_artist=True,
                         medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title('Distribution de la longueur par catégorie', fontsize=12)
    axes[1].set_ylabel('Nombre de mots')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('graph_longueur_articles.png', dpi=150)
    plt.show()
    print("Sauvegardé : graph_longueur_articles.png")


def run():
    print("Chargement des données...")
    train       = pd.read_csv('train_clean.csv')
    tfidf_train = sp.load_npz('tfidf_train.npz')

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print(f"Train : {len(train)} documents\n")

    print("Graphique 1 — Distribution des catégories...")
    graphique_1_distribution_categories(train)

    print("\nGraphique 2 — Top mots par catégorie (peut prendre ~30s)...")
    graphique_2_top_mots_par_categorie(train, tfidf_train, vectorizer)

    print("\nGraphique 3 — Longueur des articles...")
    graphique_3_longueur_articles(train)

    print("\nTous les graphiques sont générés !")


if __name__ == '__main__':
    run()
