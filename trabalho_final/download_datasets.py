"""
Script para download dos 5 datasets binários do Kaggle
Trabalho Final - EEE933 - Planejamento e Análise de Experimentos
Equipe F: Bernardo, Gustavo, Marília
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os
from pathlib import Path
import ssl
import warnings

# Ignorar avisos de SSL (apenas para desenvolvimento)
warnings.filterwarnings('ignore')

# Desabilitar verificação SSL
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KAGGLE_PROXY_VERIFY'] = 'False'

# Criar pasta data se não existir
os.makedirs('data', exist_ok=True)

# Inicializar API do Kaggle
api = KaggleApi()
api.authenticate()

print("=" * 70)
print("DOWNLOAD DOS DATASETS DO KAGGLE")
print("=" * 70)

# Definir os datasets e seus nomes de saída
datasets = [
    {
        'name': 'fedesoriano/stroke-prediction-dataset',
        'output': 'data/stroke.csv',
        'description': 'Stroke Prediction Dataset'
    },
    {
        'name': 'yasserh/titanic-dataset',
        'output': 'data/titanic.csv',
        'description': 'Titanic Dataset'
    },
    {
        'name': 'jsphyg/weather-dataset-rattle-package',
        'output': 'data/weather.csv',
        'description': 'Weather Dataset (Rain Prediction)'
    },
    {
        'name': 'yasserh/breast-cancer-dataset',
        'output': 'data/breast_cancer.csv',
        'description': 'Breast Cancer Dataset'
    },
    {
        'name': 'adityakadiwal/water-potability',
        'output': 'data/water_potability.csv',
        'description': 'Water Potability Dataset'
    }
]

# Download de cada dataset
for idx, dataset in enumerate(datasets, 1):
    print(f"\n[{idx}/5] Baixando: {dataset['description']}")
    print(f"    Kaggle: {dataset['name']}")

    try:
        # Separar owner/dataset-slug
        owner, dataset_slug = dataset['name'].split('/')

        # Baixar dataset para pasta temporária
        temp_dir = 'temp_kaggle'
        os.makedirs(temp_dir, exist_ok=True)

        api.dataset_download_files(dataset['name'], path=temp_dir, unzip=True)
        print(f"    Baixado em: {temp_dir}")

        # Encontrar o primeiro arquivo CSV no diretório
        csv_files = list(Path(temp_dir).glob('*.csv'))

        if not csv_files:
            print(f"    ✗ Nenhum arquivo CSV encontrado")
            continue

        # Ler o primeiro CSV encontrado
        csv_file = csv_files[0]
        print(f"    Lendo arquivo: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Salvar na pasta data/
        df.to_csv(dataset['output'], index=False)

        print(f"    ✓ Salvo em: {dataset['output']}")
        print(f"    Shape: {df.shape} (linhas, colunas)")

        # Limpar arquivos temporários
        for f in csv_files:
            f.unlink()

    except Exception as e:
        print(f"    ✗ Erro ao processar: {str(e)}")

print("\n" + "=" * 70)
print("DOWNLOAD CONCLUÍDO!")
print("=" * 70)
