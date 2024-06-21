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
df = pd.read_csv('dataset.csv', nrows=10000)  

if 'IPC' in df.columns:
    df.drop(columns=['IPC'], inplace=True)
    print("La colonne 'IPC' a été supprimée avec succès.")
else:
    print("La colonne 'IPC' n'existe pas dans le DataFrame.")

# Nettoyer les descriptions PAS NECESSAIRE
df['Cleaned_Description'] = df['description'].apply(clean_text)

# Vectoriser les descriptions nettoyées avec TF-IDF POURQUOI ??? 
tfidf_vectorizer = TfidfVectorizer(max_features=9000)  # Limite à 5000 mots les plus fréquents
X = tfidf_vectorizer.fit_transform(df['Cleaned_Description'])

# Simplifier les labels CPC en utilisant uniquement la première lettre et les deux premiers chiffres
df['Primary_CPC_Simplified'] = df['CPC'].apply(lambda x: x.split(';')[0][:3])

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, df['Primary_CPC_Simplified'], test_size=0.2, random_state=42)

# Entraîner un modèle de régression logistique
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Obtenir les coefficients du modèle
coefficients = model.coef_

# Obtenir les mots les plus importants pour chaque classe CPC simplifiée
feature_names = tfidf_vectorizer.get_feature_names_out()

# Afficher les mots les plus importants pour les premières classes CPC
for idx, cpc_code in enumerate(model.classes_[:20]):  
    top_features = coefficients[idx].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_features]
    print(f"Mots-clés pour CPC {cpc_code} : {', '.join(top_words)}")


#idées
## Vérifier les valeurs manquantes
#print(df.isnull().sum())

## Par exemple, supprimer les lignes avec des valeurs manquantes dans 'description' et 'CPC'
#df.dropna(subset=['description', 'CPC'], inplace=True)

