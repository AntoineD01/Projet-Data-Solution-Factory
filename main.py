import pandas as pd

# Lire les 10 premières lignes du fichier CSV
df = pd.read_csv('EFREI - LIPSTIP - 50k elements EPO.csv', nrows=100)

labels = pd.read_csv('labels.txt')

cpc_dict = dict(zip(labels['Code CPC'], labels['Title']))
print(cpc_dict)

print(df['CPC'][0])
tags = df['CPC'][0]

for tag in tags:
    # Extraire la première lettre du tag
    first_letter = tag[0]
    
    # Vérifier si la première lettre est présente dans le dictionnaire
    if first_letter in cpc_dict:
        # Imprimer la signification correspondante
        print(f"Tag: {tag}, Signification: {cpc_dict[first_letter]}")
    



