import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Télécharger les modèles de tokenisation NLTK
nltk.download('punkt')

# Fonction pour nettoyer le texte
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Enlever les balises HTML
    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Enlever les caractères spéciaux
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

# Lire le fichier CSV
df = pd.read_csv('dataset.csv', nrows=100)  

if 'IPC' in df.columns:
    df.drop(columns=['IPC'], inplace=True)
    print("La colonne 'IPC' a été supprimée avec succès.")
else:
    print("La colonne 'IPC' n'existe pas dans le DataFrame.")

# Nettoyer les champs textuels
df['Cleaned_Claim'] = df['claim'].apply(clean_text)
df['Cleaned_Description'] = df['description'].apply(clean_text)

# Charger le modèle et le tokenizer DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=510)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Générer les embeddings pour chaque champ textuel par lots
batch_size = 8  
claim_embeddings = []
description_embeddings = []

for i in range(0, len(df), batch_size):
    batch_claims = df['Cleaned_Claim'][i:i+batch_size].tolist()
    batch_descriptions = df['Cleaned_Description'][i:i+batch_size].tolist()
    claim_embeddings.append(get_embeddings(batch_claims))
    description_embeddings.append(get_embeddings(batch_descriptions))

# Concaténer les embeddings
claim_embeddings = torch.cat(claim_embeddings)
description_embeddings = torch.cat(description_embeddings)

# Agrégation par sommation vectorielle
aggregated_embeddings = claim_embeddings + description_embeddings

# Simplifier les labels CPC en utilisant uniquement la première lettre et les deux premiers chiffres
df['Primary_CPC_Simplified'] = df['CPC'].apply(lambda x: x.split(';')[0][:3])

# Préparer les données pour l'entraînement
X = aggregated_embeddings.numpy()
y = df['Primary_CPC_Simplified']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplatir les données d'entrée pour qu'elles soient 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Entraîner un modèle de régression logistique
model = LogisticRegression(max_iter=1000)
model.fit(X_train_flat, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_flat)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Vectorisation des mots-clés
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(df['Cleaned_Claim'])

# Obtenir les noms des caractéristiques
feature_names = tfidf_vectorizer.get_feature_names_out()

# Afficher les mots les plus importants pour les premières classes CPC
coefficients = model.coef_
for idx, cpc_code in enumerate(model.classes_[:20]):  
    top_features = coefficients[idx].argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_features]
    print(f"Mots-clés pour CPC {cpc_code} : {', '.join(top_words)}")
