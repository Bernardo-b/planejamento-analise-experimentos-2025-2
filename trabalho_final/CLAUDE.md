# CLAUDE.md - Contexto do Projeto

**Última Atualização:** 25/11/2025 19:30

---

## Visão Geral do Projeto

**Disciplina:** EEE933 - Planejamento e Análise de Experimentos (2025/2)
**Professor:** Michel Bessani
**Equipe F:** Bernardo Bacha de Resende, Gustavo Augusto Faria dos Reis, Marília Macêdo de Melo

### Objetivo Principal
Experimento planejado usando **RCBD (Randomized Complete Block Design)** para comparar diferentes tratamentos de algoritmos de classificação ML, controlando variabilidade através de datasets como blocos.

---

## Estrutura do Experimento

### Delineamento: RCBD
- **Blocos:** 5 datasets de **classificação binária** (diferentes complexidades)
- **Tratamentos:** A DEFINIR (kernels SVM, algoritmos, hiperparâmetros)
- **Variável Resposta:** Métrica de desempenho (acurácia, F1-score, AUC-ROC, etc.)
- **Objetivo:** Comparar tratamentos controlando variabilidade entre datasets

### Os 5 Datasets (Blocos)

| Dataset | Amostras | Features | Target | Valores Nulos | Dificuldade | Observações |
|---------|----------|----------|--------|---------------|-------------|-------------|
| **Breast Cancer** | 569 | 31 | diagnosis (M/B) | Nenhum | Baixa | Diagnóstico câncer - dataset limpo |
| **Titanic** | 891 | 11 | Survived (0/1) | Age, Cabin, Embarked | Baixa-Média | Sobrevivência - requer feature engineering |
| **Water Potability** | 3,276 | 9 | Potability (0/1) | pH, Sulfate, Trihalomethanes | Média | Qualidade da água - dados faltantes |
| **Employee** | 4,653 | 8 | LeaveOrNot (0/1) | Nenhum | Média | Rotatividade funcionários - classes balanceadas (~34%) |
| **Australia Rain** | 145,460 | 22 | RainTomorrow (Yes/No) | Muitos (~40%) | Alta | Previsão de chuva - dataset grande com muitos nulos |

**Justificativa dos Blocos:**
- **Todos são classificação binária** (simplifica análise e permite métricas consistentes)
- **Diversidade em tamanho**: de 569 a 145,460 amostras
- **Diversidade em features**: de 8 a 31 features
- **Diferentes desafios**: dados limpos vs. muitos nulos, classes balanceadas vs. desbalanceadas
- **Variedade de domínios**: saúde, transporte, meio ambiente, qualidade da água, recursos humanos

---

## Arquivos e Estrutura

```
trabalho_final/
├── CLAUDE.md                           # Este arquivo (contexto para Claude)
├── data/                               # Datasets em CSV
│   ├── breast_cancer.csv              # ✅ 569 × 32
│   ├── titanic.csv                    # ✅ 891 × 12
│   ├── water_potability.csv           # ✅ 3,276 × 10
│   ├── Employee.csv                   # ✅ 4,653 × 9
│   └── weather.csv                    # ✅ 145,460 × 23
├── download_datasets.py                # Script de download (não usado - baixado manualmente)
├── inspect_datasets.py                 # ✅ Script de inspeção dos datasets
├── notebooks/
│   ├── data_import.ipynb              # (antigo - datasets multiclasse)
│   └── data_preprocessing.ipynb        # ✅ Pré-processamento dos 5 datasets binários
├── TrabalhoFinal (1).pdf              # Instruções oficiais do trabalho
└── Proposta de Trabalho....pdf        # Proposta apresentada
```

### Datasets Disponíveis (CSV na pasta data/)

Todos os 5 datasets estão disponíveis em formato CSV na pasta `data/`:

```python
# Dataset 1: Breast Cancer
# Arquivo: data/breast_cancer.csv
# Shape: (569, 32) - 31 features + target 'diagnosis'
# Target: 'diagnosis' (M=Malignant, B=Benign)
# Valores nulos: Nenhum

# Dataset 2: Titanic
# Arquivo: data/titanic.csv
# Shape: (891, 12) - 11 features + target 'Survived'
# Target: 'Survived' (0=No, 1=Yes)
# Valores nulos: Age (177), Cabin (687), Embarked (2)

# Dataset 3: Water Potability
# Arquivo: data/water_potability.csv
# Shape: (3276, 10) - 9 features + target 'Potability'
# Target: 'Potability' (0=Not potable, 1=Potable)
# Valores nulos: pH (491), Sulfate (781), Trihalomethanes (162)

# Dataset 4: Employee Attrition
# Arquivo: data/Employee.csv
# Shape: (4653, 9) - 8 features + target 'LeaveOrNot'
# Target: 'LeaveOrNot' (0=Ficou, 1=Saiu do emprego)
# Valores nulos: Nenhum

# Dataset 5: Australia Rain (Weather)
# Arquivo: data/weather.csv
# Shape: (145460, 23) - 22 features + target 'RainTomorrow'
# Target: 'RainTomorrow' (Yes/No)
# Valores nulos: Muitos (~40% das features)
```

---

## Pré-processamento Necessário

### Breast Cancer
- ✅ Dataset limpo - sem valores nulos
- Target 'diagnosis': M (Malignant) → 1, B (Benign) → 0
- Remover coluna 'id' (não informativa)
- **Pré-processamento**: Apenas normalização/padronização das features

### Titanic
- ⚠️ Valores nulos em Age, Cabin, Embarked
- Features categóricas: Sex, Embarked, etc.
- **Pré-processamento necessário:**
  - Imputação de valores nulos (Age: mediana, Embarked: moda)
  - Remover ou feature engineering em Cabin (muitos nulos)
  - One-hot encoding para categóricas
  - Remover colunas não informativas (PassengerId, Name, Ticket)

### Water Potability
- ⚠️ Valores nulos em pH (15%), Sulfate (24%), Trihalomethanes (5%)
- **Pré-processamento necessário:**
  - Imputação de valores nulos (mediana ou KNN imputer)
  - Normalização/padronização

### Employee Attrition
- ✅ Dataset limpo - sem valores nulos
- Target: LeaveOrNot (0=Ficou, 1=Saiu do emprego)
- Features categóricas: Education, City, Gender, EverBenched
- **Pré-processamento aplicado:**
  - One-hot encoding para features categóricas
  - Normalização/padronização
  - **Vantagem**: Classes razoavelmente balanceadas (~34% saídas vs 5% do Stroke anterior)

### Australia Rain (Weather)
- ⚠️ MUITOS valores nulos (~40% em várias features)
- Features categóricas: Location, WindGustDir, RainToday, etc.
- **Pré-processamento necessário:**
  - Decisão: remover linhas com muitos nulos OU imputação agressiva
  - One-hot encoding para categóricas
  - Conversão de RainTomorrow (Yes/No → 1/0)
  - Remover coluna Date (ou extrair features temporais)
  - Normalização/padronização
  - **Atenção**: Dataset muito grande - considerar amostragem

---

## Pré-processamento Aplicado (data_preprocessing.ipynb)

### Notebook: `notebooks/data_preprocessing.ipynb`

Pipeline completo de pré-processamento implementado para os 5 datasets:

**Etapas Gerais:**
1. Carregamento dos CSVs
2. Remoção de colunas não informativas (IDs, nomes, datas, colunas com alta cardinalidade)
3. Separação de target (y) e features (X)
4. Tratamento de valores nulos:
   - Features numéricas: imputação com mediana
   - Features categóricas: imputação com moda
5. One-hot encoding para features categóricas (com `drop_first=True`)
6. Normalização com StandardScaler (z-score) para todas as features
7. Validação final (verificar nulos, tipos, distribuição de classes)

### Resultados do Pré-processamento:

| Dataset | Amostras Final | Features Final | Nulos | Classes (0/1) | Proporção |
|---------|---------------|----------------|-------|---------------|-----------|
| **Breast Cancer** | 569 | 30 | 0 | Balanceado | ~37% malignant |
| **Titanic** | 891 | 10 | 0 | Desbalanceado | ~38% survived |
| **Water Potability** | 3,276 | 9 | 0 | Balanceado | ~39% potable |
| **Employee** | 4,653 | 12 | 0 | Balanceado | ~34% saiu |
| **Weather** | ~10,000 | 62 | 0 | Desbalanceado | ~22% rain |

**Observações Importantes:**
- **Employee**: Substituiu Stroke. Classes bem balanceadas (~34% saídas) - excelente para treinamento!
- **Weather**: Reduzido de 145k para ~10k amostras via amostragem estratificada para balancear com outros datasets.
- **Todas as features normalizadas** com StandardScaler (média=0, std=1).
- **Pronto para uso** em classificadores de ML (SVM, Random Forest, etc.).

### Variáveis Disponíveis:

Após executar o notebook, as seguintes variáveis estarão disponíveis:

```python
# Dataset 1: Breast Cancer
X_breast_cancer  # DataFrame normalizado (569, 30)
y_breast_cancer  # Series (569,)

# Dataset 2: Titanic
X_titanic  # DataFrame normalizado (891, ~10-12)
y_titanic  # Series (891,)

# Dataset 3: Water Potability
X_water_potability  # DataFrame normalizado (3276, 9)
y_water_potability  # Series (3276,)

# Dataset 4: Employee
X_employee  # DataFrame normalizado (4653, 12)
y_employee  # Series (4653,)

# Dataset 5: Weather
X_weather  # DataFrame normalizado (~10000, ~20-30)
y_weather  # Series (~10000,)
```

---

## Baseline - SVM com Kernel RBF

### Seção 8 do Notebook

Para validar os dados e obter métricas de referência, foi implementado um baseline simples:

**Configuração:**
- Train/Test Split: 80/20 (stratified)
- Modelo: SVM com kernel RBF (padrão sklearn)
- Métricas calculadas: Acurácia, Precisão, Recall, F1-Score

**Objetivo:**
1. Validar que todos os datasets estão funcionando corretamente
2. Obter métricas baseline para comparação futura no experimento RCBD

**Resultados Esperados (exemplo):**

| Dataset | Treino | Teste | Acurácia | Precisão | Recall | F1-Score |
|---------|--------|-------|----------|----------|--------|----------|
| Breast Cancer | 455 | 114 | ~95% | ~93% | ~93% | ~93% |
| Titanic | 712 | 179 | ~86% | ~94% | ~68% | ~79% |
| Water Potability | 2620 | 656 | ~68% | ~72% | ~30% | ~42% |
| Employee | 3722 | 931 | ~75% | ~65% | ~55% | ~60% |
| Weather | 8000 | 2000 | ~85% | ~75% | ~51% | ~61% |

**Observações:**
- **Employee**: Substituiu Stroke. Métricas moderadas esperadas (~75% acurácia) - dataset balanceado com bom desempenho.
- **Water Potability**: Métricas medianas esperadas - problema mais difícil.
- **Breast Cancer**: Métricas altas esperadas - dataset limpo e bem comportado.
- Estes resultados servem como baseline para comparação com outros algoritmos/configurações no experimento RCBD.

---

## Status do Trabalho

### ✅ Concluído
- [x] Definição da questão experimental
- [x] Seleção dos 5 datasets binários (blocos)
- [x] Download dos datasets do Kaggle
- [x] Inspeção inicial dos datasets (shape, colunas, nulos)
- [x] Identificação de necessidades de pré-processamento
- [x] **Criação do notebook de pré-processamento unificado** (`data_preprocessing.ipynb`)
- [x] **Pré-processamento completo dos 5 datasets:**
  - [x] Breast Cancer (569 amostras, 30 features)
  - [x] Titanic (891 amostras, 10 features)
  - [x] Water Potability (3,276 amostras, 9 features)
  - [x] Employee (4,653 amostras, 12 features)
  - [x] Weather (~10k amostras, 62 features)
- [x] **Baseline com SVM (kernel RBF):**
  - [x] Train/test split (80/20) para os 5 datasets
  - [x] Treinamento SVM básico
  - [x] Cálculo de métricas (Acurácia, Precisão, Recall, F1)
  - [x] Validação de que dados estão funcionando

### 🔜 Próximos Passos
- [ ] **Definir tratamentos** (ex: SVM linear, RBF, polynomial; ou diferentes algoritmos)
- [ ] **Implementar experimento RCBD** (aplicar cada tratamento em cada bloco)
- [ ] **Coletar resultados** (métricas de desempenho)
- [ ] **Análise estatística:**
  - [ ] ANOVA para RCBD
  - [ ] Verificar pressupostos (normalidade, homocedasticidade)
  - [ ] Testes post-hoc (se necessário)
- [ ] **Conclusões e recomendações**
- [ ] **Preparar apresentação final** (15 min)

---

## Prazos Importantes

- ✅ **18/11/2025:** Apresentação da proposta (10 min) - CONCLUÍDO
- 🔜 **09/12/2025:** Apresentação final (15 min) - PRÓXIMO

---

## Notas Técnicas Importantes

### Ambiente Python
- Python 3.13
- Ambiente virtual: `.venv/` e `venv/` (ambos presentes)
- Bibliotecas principais: pandas, numpy, sklearn, matplotlib, seaborn

### Git Status (25/11/2025)
- Branch: `main`
- Commits recentes:
  - `f9d9529` - notebook do trabalho final
  - `a19b03c` - Add .gitignore
- Arquivos deletados no staging: vários .zip de datasets (não mais necessários?)

### Considerações para Análise RCBD
1. **Modelo estatístico:**
   ```
   y_ij = μ + τ_i + β_j + ε_ij
   onde:
   - τ_i = efeito do tratamento i
   - β_j = efeito do bloco j (dataset)
   - ε_ij = erro aleatório
   ```

2. **Hipóteses a testar:**
   - H0: Não há diferença entre tratamentos
   - H1: Pelo menos um tratamento difere dos demais

3. **Validações necessárias:**
   - Normalidade dos resíduos (Shapiro-Wilk, Q-Q plot)
   - Homocedasticidade (Levene, Bartlett)
   - Independência das observações

---

## Ideias e Questões em Aberto

### Possíveis Tratamentos
- **Opção 1:** Diferentes kernels SVM (linear, RBF, polynomial, sigmoid)
- **Opção 2:** Diferentes algoritmos (SVM, Random Forest, KNN, Logistic Regression)
- **Opção 3:** SVM com diferentes valores de C ou gamma

### Métrica de Desempenho (Classificação Binária)
- **AUC-ROC** (recomendado - robusto a desbalanceamento, permite comparação justa)
- F1-Score (balança precisão e recall)
- Acurácia (simples, mas cuidado com classes desbalanceadas - especialmente Stroke)
- Precisão e Recall (úteis para análise complementar)

**Decisão**: Usar **AUC-ROC** como métrica principal pois:
- Não é afetada por desbalanceamento de classes
- Todos os datasets são binários
- Permite comparação justa entre datasets diferentes

### Estratégia de Validação
- K-fold cross-validation (k=5 ou k=10)
- Stratified para manter proporção de classes
- Média das k rodadas como resultado final

---

## Referências Úteis

### Literatura
- Montgomery, D.C. - Design and Analysis of Experiments
- Documentação sklearn: https://scikit-learn.org/

### Arquivos de Referência no Projeto
- `TrabalhoFinal (1).pdf` - instruções completas do professor
- `notebooks/data_import.ipynb` - código de importação e preparação

---

**Notas de Desenvolvimento:**
- Este arquivo será atualizado conforme o projeto avança
- Manter sempre sincronizado com decisões tomadas
- Documentar escolhas metodológicas e justificativas

---

## Histórico de Mudanças

### 25/11/2025 19:30 - Substituição: Stroke → Employee
**Motivação:** Dataset Stroke tinha forte desbalanceamento (~5% eventos positivos), resultando em métricas baseline ruins e dificultando análise.

**Ação:** Substituição completa do dataset Stroke por Employee Attrition.

**Dataset Employee:**
- 4,653 amostras × 9 colunas (8 features + target)
- Target: LeaveOrNot (0=Ficou, 1=Saiu)
- Classes balanceadas: ~34% saídas (vs 5% do Stroke)
- Sem valores nulos
- Features: Education, City, Gender, EverBenched (categóricas) + numéricas

**Modificações realizadas:**
- Seção 5 do notebook: novo pré-processamento Employee
- Seções 7 e 8: atualizações nas referências
- CLAUDE.md: todas as tabelas e descrições atualizadas

**Benefício:** Métricas baseline mais confiáveis e dataset com melhor qualidade para o experimento RCBD.

### 25/11/2025 19:15 - Baseline com SVM
**Adicionado:** Seção 8 no notebook `data_preprocessing.ipynb` com baseline SVM.

**Implementação:**
- Train/test split (80/20) estratificado para cada dataset
- Treinamento de SVM com kernel RBF (padrão)
- Cálculo de 4 métricas: Acurácia, Precisão, Recall, F1-Score
- Tabela resumo consolidada com resultados dos 5 datasets

**Resultado:**
- Validação de que todos os dados estão funcionando corretamente
- Métricas baseline disponíveis para comparação futura
- Identificação de desafios (Stroke muito desbalanceado - posteriormente substituído por Employee)

### 25/11/2025 19:00 - Pré-processamento Completo
**Criado:** Notebook `data_preprocessing.ipynb` com pipeline completo de pré-processamento.

**Implementações:**
- Tratamento de valores nulos (mediana para numérico, moda para categórico)
- Remoção de colunas não informativas (IDs, nomes, datas)
- One-hot encoding para features categóricas
- Normalização com StandardScaler (z-score)
- Amostragem estratificada do Weather dataset (145k → 10k)

**Resultado:** 5 pares (X, y) prontos para uso em modelos de ML, todos:
- Sem valores nulos
- Features numéricas e normalizadas
- Validados e documentados

### 25/11/2025 18:30 - Mudança de Datasets
**Motivação:** Os datasets originais (Iris, Wine, MNIST) incluíam problemas multiclasse, o que complicaria a análise por exigir métricas diferentes e interpretação mais complexa.

**Decisão:** Substituir TODOS os datasets por problemas de **classificação binária apenas**.

**Datasets Removidos:**
- Iris (3 classes)
- Wine (3 classes)
- MNIST Digits (10 classes)
- Heart Disease (mantido conceito mas substituído)

**Novos Datasets (Todos Binários):**
1. Breast Cancer (569 amostras) - dataset limpo
2. Titanic (891 amostras) - requer feature engineering
3. Water Potability (3,276 amostras) - valores nulos moderados
4. Stroke Prediction (5,110 amostras) - classes desbalanceadas
5. Australia Rain (145,460 amostras) - grande e com muitos nulos

**Benefícios:**
- Permite uso de métricas consistentes (AUC-ROC) em todos os blocos
- Simplifica interpretação dos resultados
- Mantém diversidade de complexidade e desafios
- Facilita análise estatística (ANOVA) com mesma variável resposta
