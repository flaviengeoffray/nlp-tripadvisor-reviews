import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Chargement des données
from datasets import load_dataset
ds = load_dataset("jniimi/tripadvisor-review-rating")
raw_data = pd.DataFrame(ds['train'])

# Préparation des données
# On utilisera la colonne 'review' comme entrée et 'overall' comme cible
def prepare_data(df):
    df = df.copy()
    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna(subset=['review', 'overall'])
    # Conversion de 'overall' en entier pour la classification
    df['overall'] = df['overall'].astype(int)
    return df

def char_tokenize(text):
    """Fonction de tokenisation personnalisée pour les caractères."""
    # Tokenisation par caractères
    tokens = list(text)
    # Filtrage des stop words et des tokens courts
    tokens = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]
    return tokens

def byte_tokenize(text):
    """Fonction de tokenisation personnalisée pour les octets."""
    # Tokenisation par octets
    tokens = list(text.encode('utf-8'))
    # Filtrage des stop words et des tokens courts
    tokens = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]
    return tokens    

def custom_tokenizer(tokenizer_type):
    """Fonction pour créer un tokenizer personnalisé avec différentes méthodes."""
    stop_words = set(stopwords.words('english'))
    
    def tokenize(text):
        if tokenizer_type == 'word_tokenize':
            tokens = word_tokenize(text.lower())
        elif tokenizer_type == 'char':
            tokens = char_tokenize(text)
        elif tokenizer_type == 'byte':
            tokens = byte_tokenize(text)
        else:
            tokens = text.lower().split()  # Tokenisation simple par espace
        
        # Filtrage des stop words et des tokens courts
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return tokens
    
    return tokenize

def evaluate_tokenizer(X_train, X_test, y_train, y_test, tokenizer_name, tokenizer_func=None, vectorizer_type='count'):
    """Évalue un tokenizer spécifique avec Naive Bayes."""
    start_time = time.time()
    
    # Configuration du vectoriseur en fonction du type
    if vectorizer_type == 'count':
        if tokenizer_func:
            vectorizer = CountVectorizer(tokenizer=tokenizer_func, min_df=5)
        else:
            vectorizer = CountVectorizer(min_df=5)
    else:  # tfidf
        if tokenizer_func:
            vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, min_df=5)
        else:
            vectorizer = TfidfVectorizer(min_df=5)
    
    # Création du pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', MultinomialNB())
    ])
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Prédiction
    y_pred = pipeline.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Résumé des résultats
    results = {
        'tokenizer': tokenizer_name,
        'vectorizer': vectorizer_type,
        'accuracy': accuracy,
        'processing_time': processing_time,
        'report': report
    }
    
    return results, y_pred

def benchmark_tokenizers(df):
    """Benchmark de différents tokenizers."""
    # Préparation des données
    X = df['review'].values
    y = df['overall'].values
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configuration des tokenizers à tester
    tokenizers = {
        'Default (CountVectorizer)': None,
        'Default (TF-IDF)': None,
        'Word Tokenize': custom_tokenizer('word_tokenize'),
        'WordPunct Tokenize': custom_tokenizer('wordpunct'),
        'Tweet Tokenize': custom_tokenizer('tweet'),
        'Simple Split': custom_tokenizer('split')
    }
    
    results = []
    predictions = {}
    
    # Évaluation de chaque tokenizer
    for name, tokenizer in tokenizers.items():
        if 'TF-IDF' in name:
            result, y_pred = evaluate_tokenizer(X_train, X_test, y_train, y_test, name, tokenizer, 'tfidf')
        else:
            if 'Default (CountVectorizer)' == name:
                result, y_pred = evaluate_tokenizer(X_train, X_test, y_train, y_test, name, None, 'count')
            else:
                result, y_pred = evaluate_tokenizer(X_train, X_test, y_train, y_test, name, tokenizer, 'count')
        
        results.append(result)
        predictions[name] = y_pred
    
    return results, predictions, y_test

def plot_results(results):
    """Visualisation des résultats."""
    # Extraction des données pour les graphiques
    tokenizers = [r['tokenizer'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['processing_time'] for r in results]
    
    # Création des graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique d'exactitude
    ax1.bar(tokenizers, accuracies, color='skyblue')
    ax1.set_title('Exactitude par tokenizer')
    ax1.set_xlabel('Tokenizer')
    ax1.set_ylabel('Exactitude')
    ax1.set_ylim(0.7, 1.0)  # Ajustez en fonction de vos résultats
    ax1.tick_params(axis='x', rotation=45)
    
    # Graphique de temps de traitement
    ax2.bar(tokenizers, times, color='salmon')
    ax2.set_title('Temps de traitement par tokenizer')
    ax2.set_xlabel('Tokenizer')
    ax2.set_ylabel('Temps (secondes)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Affichage d'un tableau récapitulatif
    summary_data = []
    for r in results:
        weighted_f1 = r['report']['weighted avg']['f1-score']
        summary_data.append([r['tokenizer'], r['accuracy'], weighted_f1, r['processing_time']])
    
    summary_df = pd.DataFrame(summary_data, columns=['Tokenizer', 'Accuracy', 'Weighted F1', 'Processing Time'])
    return summary_df

def plot_confusion_matrix(y_test, predictions, tokenizer_name):
    """Affiche la matrice de confusion pour un tokenizer spécifique."""
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion pour {tokenizer_name}')
    plt.xlabel('Prédiction')
    plt.ylabel('Valeur réelle')
    plt.show()

def main():
    # Préparation des données (ajustez en fonction de votre variable)
    # Si vous avez déjà chargé 'raw_data' comme indiqué plus haut
    data = prepare_data(raw_data)  # Utilisez raw_data si déjà chargé
    
    # Aperçu des données
    print("Aperçu des données préparées:")
    print(data.head())
    print(f"Nombre total d'échantillons: {len(data)}")
    print(f"Distribution des notes: \n{data['overall'].value_counts().sort_index()}")
    
    # Benchmark des tokenizers
    print("\nBenchmark des tokenizers en cours...")
    results, predictions, y_test = benchmark_tokenizers(data)
    
    # Affichage des résultats
    summary_df = plot_results(results)
    print("\nRésumé des performances:")
    print(summary_df.to_string(index=False))
    
    # Affichage de la matrice de confusion pour le meilleur tokenizer
    best_tokenizer = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Tokenizer']
    print(f"\nMeilleur tokenizer: {best_tokenizer}")
    plot_confusion_matrix(y_test, predictions[best_tokenizer], best_tokenizer)
    
    # Analyse détaillée pour le meilleur tokenizer
    best_idx = next(i for i, r in enumerate(results) if r['tokenizer'] == best_tokenizer)
    print("\nRapport de classification détaillé pour le meilleur tokenizer:")
    for label, metrics in results[best_idx]['report'].items():
        if label.isdigit():
            print(f"Note {label}: Précision={metrics['precision']:.3f}, Rappel={metrics['recall']:.3f}, F1-score={metrics['f1-score']:.3f}")

if __name__ == "__main__":
    main()
