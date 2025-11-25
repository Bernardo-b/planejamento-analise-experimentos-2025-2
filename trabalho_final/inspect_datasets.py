"""
Script para inspecionar os 5 datasets binários baixados
"""

import pandas as pd
import os

os.chdir('data')

datasets = [
    {'file': 'stroke.csv', 'name': 'Stroke Prediction'},
    {'file': 'titanic.csv', 'name': 'Titanic Survival'},
    {'file': 'weather.csv', 'name': 'Australia Rain'},
    {'file': 'breast_cancer.csv', 'name': 'Breast Cancer'},
    {'file': 'water_potability.csv', 'name': 'Water Potability'}
]

print("=" * 80)
print("INSPEÇÃO DOS DATASETS")
print("=" * 80)

for ds in datasets:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {ds['name']}")
    print(f"Arquivo: {ds['file']}")
    print("=" * 80)

    df = pd.read_csv(ds['file'])

    print(f"Shape: {df.shape} (linhas, colunas)")
    print(f"\nColunas ({len(df.columns)}):")
    print(df.columns.tolist())

    print(f"\nTipos de dados:")
    print(df.dtypes.value_counts())

    print(f"\nValores nulos:")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(null_counts[null_counts > 0])
    else:
        print("Nenhum valor nulo")

    print(f"\nPrimeiras 3 linhas:")
    print(df.head(3))

print("\n" + "=" * 80)
print("INSPEÇÃO CONCLUÍDA!")
print("=" * 80)
