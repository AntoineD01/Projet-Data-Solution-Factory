import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk

# Télécharger les modèles de tokenisation NLTK
nltk.download('punkt')

# Fonction pour nettoyer le texte
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Enlever les balises HTML
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Enlever les caractères spéciaux
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Lire le fichier CSV
df = pd.read_csv('EFREI - LIPSTIP - 50k elements EPO.csv', nrows=2000)  # Lire plus de lignes pour un meilleur modèle

# Nettoyer les descriptions
df['Cleaned_Description'] = df['description'].apply(clean_text)

# Vectoriser les descriptions nettoyées avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limite à 5000 mots les plus fréquents
X = tfidf_vectorizer.fit_transform(df['Cleaned_Description'])

# Pour simplifier, nous utiliserons uniquement le premier code IPC de chaque entrée
df['Primary_CPC'] = df['CPC'].apply(lambda x: x.split(';')[0])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, df['Primary_CPC'], test_size=0.2, random_state=42)

# Entraîner un modèle de régression logistique
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Obtenir les coefficients du modèle
coefficients = model.coef_

# Obtenir les mots les plus importants pour chaque code IPC
feature_names = tfidf_vectorizer.get_feature_names_out()

# Afficher les mots les plus importants pour les premières classes IPC
for idx, ipc_code in enumerate(model.classes_[:10]):  
    top_features = coefficients[idx].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_features]
    print(f"Mots-clés pour CPC {ipc_code}: {', '.join(top_words)}")