# CLAUDE.md - Contexto do Projeto

**Última Atualização:** 25/11/2025 20:45

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

## Otimização de Hiperparâmetros - SVM

### Seção 9 do Notebook

Implementação de 3 métodos de otimização de hiperparâmetros para comparação no experimento RCBD:

**Métodos Implementados:**
1. **GridSearch** (`grid_search_svm()`)
   - Busca exaustiva em grid definido
   - Grid quadrado: n_iter=16 → 4×4 = 16 combinações
   - Ranges: C=[0.01, 1000], gamma=[0.0001, 10] (escala log)

2. **RandomSearch** (`random_search_svm()`)
   - Amostragem aleatória no espaço de busca
   - n_iter combinações aleatórias
   - Distribuição log-uniforme para C e gamma

3. **BayesianOptimization** (`bayesian_search_svm()`)
   - Otimização bayesiana com scikit-optimize (skopt)
   - n_iter iterações usando Gaussian Process
   - Exploração inteligente do espaço de busca

**Configuração Comum:**
- Modelo: SVM com kernel='rbf'
- Hiperparâmetros otimizados: C e gamma
- Mesmo budget (n_iter) para comparação justa
- Sem cross-validation: treino direto em X_train, teste em X_test
- Métricas retornadas: acuracia, precisao, recall, f1_score, tempo
- Parâmetro verbose para silenciar prints em loops

**Assinatura das Funções:**
```python
def grid_search_svm(X_train, y_train, X_test, y_test, n_iter=16, verbose=True):
    # Retorna dict: metodo, best_params, acuracia, precisao, recall, f1_score, tempo

def random_search_svm(X_train, y_train, X_test, y_test, n_iter=16, verbose=True):
    # Retorna dict: metodo, best_params, acuracia, precisao, recall, f1_score, tempo

def bayesian_search_svm(X_train, y_train, X_test, y_test, n_iter=16, verbose=True):
    # Retorna dict: metodo, best_params, acuracia, precisao, recall, f1_score, tempo
```

---

## Experimento RCBD Completo

### Seção 10 do Notebook

Implementação da estrutura completa do experimento RCBD com loops aninhados.

**Configuração:**
- **Blocos:** 5 datasets (Breast Cancer, Titanic, Water Potability, Employee, Weather)
- **Repetições:** 7 seeds diferentes (1-7) para cada dataset
- **Tratamentos:** 3 métodos de otimização (GridSearch, RandomSearch, BayesianOptimization)
- **Total de experimentos:** 5 × 7 × 3 = **105 experimentos**

**Estrutura dos Loops:**
```python
for dataset in datasets (5):
    for seed in seeds (7):
        # 1. Train/test split ESTRATIFICADO (80/20) com random_state=seed
        # 2. Executar GridSearch → adicionar resultado (dataset, seed)
        # 3. Executar RandomSearch → adicionar resultado (dataset, seed)
        # 4. Executar BayesianOptimization → adicionar resultado (dataset, seed)
```

**Características:**
- Train/test split **estratificado** (mantém proporção de classes)
- Seed diferente em cada repetição (variabilidade estatística)
- Verbose=False para outputs limpos
- Barras de progresso TQDM (dataset externo, seeds interno)
- Resultados consolidados em lista de dicts

**Consolidação de Resultados:**
- DataFrame pandas com 105 linhas (35 por método)
- Colunas: dataset, seed, metodo, acuracia, precisao, recall, f1_score, tempo, best_params
- Estatísticas descritivas por método (média, std)
- 2 arquivos CSV salvos em `results/`:
  - `experimento_rcbd_resultados.csv` (com best_params como dict)
  - `experimento_rcbd_resultados_expandido.csv` (C e gamma em colunas separadas)

**Saídas Geradas:**
- DataFrame consolidado: `df_resultados`
- Arquivo CSV: `results/experimento_rcbd_resultados.csv`
- Arquivo CSV expandido: `results/experimento_rcbd_resultados_expandido.csv`
- Estatísticas resumidas por método impressas no notebook

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
- [x] **Definir tratamentos:** 3 métodos de otimização de hiperparâmetros para SVM
- [x] **Implementação das funções de otimização:**
  - [x] GridSearch com grid 4×4
  - [x] RandomSearch com 16 iterações
  - [x] BayesianOptimization com 16 iterações
  - [x] Todas com mesma interface e budget
- [x] **Implementar experimento RCBD completo:**
  - [x] Estrutura de loops aninhados (datasets × seeds × métodos)
  - [x] 5 datasets × 7 seeds × 3 métodos = 105 experimentos
  - [x] Train/test split estratificado com seeds diferentes
  - [x] Barras de progresso TQDM
- [x] **Coletar e consolidar resultados:**
  - [x] DataFrame com 105 linhas
  - [x] Estatísticas descritivas por método
  - [x] Salvar em CSV (2 versões)

### ✅ Concluído (Continuação)
- [x] **Criar notebook de análise de resultados** (`analise_resultados.ipynb`)
  - [x] Carregamento do CSV com 105 experimentos
  - [x] Scatter Plot: Tempo (log) vs Acurácia
  - [x] BoxPlot: Distribuição de Acurácia por Método
  - [x] BoxPlot: Tempo Computacional (escala log)
  - [x] BoxPlot: Tempo Computacional (escala linear)
  - [x] Resumo executivo com rankings
- [x] **Criar script de análise estatística em R** (`src/analise_estatistica_acuracia.R`)
  - [x] Carregamento e preparação (blocos = dataset_seed)
  - [x] Modelo RCBD: aov(acuracia ~ metodo + bloco)
  - [x] QQ-Plot dos resíduos (PNG)
  - [x] Teste Shapiro-Wilk (normalidade)
  - [x] Teste Fligner-Killeen (homocedasticidade)
  - [x] Lógica condicional: ANOVA (normal) vs Friedman (não-normal)
  - [x] Testes post-hoc: Tukey (normal) vs Wilcoxon+Bonferroni (não-normal)
  - [x] Relatório em TXT com todas as análises
- [x] **Criar script de análise estatística para TEMPO** (`src/analise_estatistica_tempo.R`)
  - [x] Mesma estrutura que acurácia, variável = tempo
  - [x] Modelo RCBD: aov(tempo ~ metodo + bloco)
  - [x] QQ-Plot, testes de premissas, lógica condicional
  - [x] Relatório em TXT com análise completa

### 🔄 Em Execução
- [ ] **Executar análise estatística:**
  - [ ] Rodar script R para validar pressupostos
  - [ ] Interpretar resultados dos testes

### 🔜 Próximos Passos
- [ ] **Visualizações adicionais:**
  - [ ] Gráficos de interação (método × dataset)
  - [ ] Análise dos hiperparâmetros escolhidos (C e gamma)
- [ ] **Conclusões e recomendações:**
  - [ ] Qual método teve melhor desempenho?
  - [ ] Diferenças foram significativas?
  - [ ] Trade-off entre desempenho e tempo
- [ ] **Preparar apresentação final** (15 min, 09/12/2025)

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

### 06/12/2025 - Scripts R de Análise Estatística para Acurácia e Tempo + Notebook de Análise
**Criado:** Dois scripts R complementares para análise estatística RCBD.

**Scripts Criados:**
1. `src/analise_estatistica_acuracia.R` - Análise da variável acurácia
2. `src/analise_estatistica_tempo.R` - Análise da variável tempo

**Implementação dos Scripts R:**
Ambos compartilham mesma estrutura:
- Carregamento e preparação (blocos = dataset_seed)
- Modelo RCBD: `aov(variavel ~ metodo + bloco)`
- QQ-Plot dos resíduos (PNG)
- Testes de premissas: Shapiro-Wilk e Fligner-Killeen
- Lógica condicional baseada em normalidade dos resíduos
- **Se Normal**: ANOVA paramétrica + Tukey HSD (se p < 0.05)
- **Se Não-Normal**: Friedman + Wilcoxon pareado com Bonferroni (se p < 0.05)
- Estatísticas descritivas por método
- Saídas: PNG (gráfico) + TXT (relatório completo)

**Diferenciais por Variável:**
- **Acurácia**: Métrica de desempenho dos classificadores
- **Tempo**: Custo computacional, distribuição típica assimétrica

**Características Gerais:**
- Scripts autocontidos, prontos para executar
- Tratam corretamente blocos como combinações (dataset_seed)
- Relatórios detalhados em arquivo TXT
- Decisão automática entre testes paramétricos e não-paramétricos

### 06/12/2025 - Notebook de Análise de Resultados Criado
**Criado:** Notebook `notebooks/analise_resultados.ipynb` para visualização dos 105 experimentos RCBD.

**Implementação:**
- 7 células bem definidas (imports, exploração, 4 gráficos, resumo)
- Scatter Plot: Tempo (log) vs Acurácia (diferenciado por método)
- BoxPlot: Acurácia por Método
- BoxPlot: Tempo Computacional (escala log)
- BoxPlot: Tempo Computacional (escala linear)
- Resumo executivo com rankings
- Código conciso, cada gráfico em célula separada
- Sem salvamento de imagens (apenas plt.show())

**Resultado:**
- Notebook pronto para exploração iterativa
- Visualizações profissionais para apresentação
- Análise rápida do trade-off tempo vs performance

### 25/11/2025 20:45 - Experimento RCBD Completo Implementado
**Implementado:** Seções 9 e 10 no notebook `data_preprocessing.ipynb`.

**Seção 9 - Otimização de Hiperparâmetros:**
- 3 funções implementadas: `grid_search_svm()`, `random_search_svm()`, `bayesian_search_svm()`
- Mesma interface: recebem X_train, y_train, X_test, y_test, n_iter, verbose
- Mesma saída: dict com metodo, best_params, acuracia, precisao, recall, f1_score, tempo
- GridSearch: grid 4×4 (16 combinações)
- RandomSearch: 16 amostragens aleatórias
- BayesianOptimization: 16 iterações com Gaussian Process
- Parâmetro verbose para silenciar prints durante loops

**Seção 10 - Experimento RCBD Completo:**
- Estrutura de loops aninhados: 5 datasets × 7 seeds × 3 métodos = 105 experimentos
- Dicionário de datasets organizando X e y
- Seeds de 1 a 7 para repetições
- Train/test split ESTRATIFICADO (80/20) mantendo proporção de classes
- Barras de progresso TQDM (dataset externo, seeds interno)
- Consolidação em DataFrame pandas
- Salvamento em 2 arquivos CSV:
  - `results/experimento_rcbd_resultados.csv`
  - `results/experimento_rcbd_resultados_expandido.csv` (C e gamma separados)
- Estatísticas descritivas por método impressas

**Resultado:**
- Código pronto para executar o experimento RCBD completo
- Estrutura permite fácil análise posterior (ANOVA, visualizações)
- Dados serão salvos automaticamente em CSV para análise estatística

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

---

## Análise de Resultados (analise_resultados.ipynb)

### Notebook: `notebooks/analise_resultados.ipynb`

Notebook para visualizar e explorar os resultados do experimento RCBD com 105 experimentos.

**Estrutura:**

1. **Célula 1:** Imports e carregamento
   - pandas, numpy, matplotlib, seaborn
   - Carrega `../results/experimento_rcbd_resultados.csv`

2. **Célula 2:** Exploração rápida
   - Shape, métodos, datasets
   - Estatísticas descritivas por método (média e std)

3. **Célula 3:** Scatter Plot
   - X: Tempo (escala logarítmica)
   - Y: Acurácia
   - Cores: Diferenciadas por método
   - Título: "Trade-off: Tempo vs Acurácia"

4. **Célula 4:** BoxPlot - Acurácia
   - Distribuição de Acurácia para cada método
   - Visualiza mediana, quartis e outliers

5. **Célula 5:** BoxPlot - Tempo (escala log)
   - Distribuição de Tempo Computacional
   - Eixo Y em escala logarítmica
   - Importante para visualizar diferenças grandes entre métodos

6. **Célula 6:** BoxPlot - Tempo (escala linear)
   - Mesma distribuição de tempo
   - Sem escala logarítmica para comparação

7. **Célula 7:** Resumo executivo
   - Melhor acurácia geral
   - Método mais rápido
   - Ranking por acurácia média

**Dados Analisados:**
- 105 experimentos (5 datasets × 7 seeds × 3 métodos)
- Métricas: acurácia, precisão, recall, f1_score, tempo
- Métodos: GridSearch, RandomSearch, BayesianOptimization
- Datasets: Breast Cancer, Titanic, Water Potability, Employee, Weather

**Características:**
- Código conciso (sem verbosidade desnecessária)
- Cada gráfico em célula separada
- Sem salvamento de imagens (apenas plt.show())
- Paleta visual: seaborn whitegrid + Set2
- Pronto para exploração iterativa

---

## Análise Estatística em R (analise_estatistica_acuracia.R)

### Script: `src/analise_estatistica_acuracia.R`

Script R autocontido que realiza análise estatística completa da acurácia em delineamento RCBD (Randomized Complete Block Design).

**Estrutura do Script:**

1. **Setup e Carregamento**
   - Carrega `results/experimento_rcbd_resultados.csv`
   - Cria coluna `bloco` = paste(dataset, seed, sep="_")
   - Converte `metodo` e `bloco` para factor

2. **Modelo RCBD**
   - Ajusta: `aov(acuracia ~ metodo + bloco, data=df)`
   - Modelo controla variabilidade entre blocos

3. **Gráfico QQ-Plot**
   - Salva em: `results/qqplot_acuracia.png`
   - Visualiza normalidade dos resíduos

4. **Testes de Premissas**
   - **Shapiro-Wilk**: Testa normalidade dos resíduos
   - **Fligner-Killeen**: Testa homocedasticidade entre métodos

5. **Lógica Condicional (if/else)**
   - **Se Normal (p > 0.05):**
     - Executa ANOVA paramétrica: `summary(modelo)`
     - Se metodo significativo (p < 0.05): Tukey HSD post-hoc
   - **Se Não-Normal (p ≤ 0.05):**
     - Executa Friedman test: `friedman.test(acuracia ~ metodo | bloco)`
     - Se significativo: Wilcoxon pareado com correção Bonferroni

6. **Estatísticas Descritivas**
   - Resumo por método: média, mediana, sd, min, max

7. **Saídas:**
   - **Console**: Mensagens de progresso
   - **Arquivo PNG**: `results/qqplot_acuracia.png` (QQ-Plot)
   - **Arquivo TXT**: `results/relatorio_estatistico_acuracia.txt` (Relatório completo)

**Definição de Bloco:**
- Cada combinação de (dataset, seed) é um bloco único
- Exemplo: "Breast Cancer_1", "Titanic_2", etc.
- Total: 5 datasets × 7 seeds = 35 blocos

**Delineamento:**
- Blocos: 35 (5 datasets × 7 seeds)
- Tratamentos: 3 (GridSearch, RandomSearch, BayesianOptimization)
- Observações: 105 (35 × 3)

**Como Executar:**
```r
source("src/analise_estatistica_acuracia.R")
```

Ou no terminal:
```bash
Rscript src/analise_estatistica_acuracia.R
```

---

## Análise Estatística em R - Tempo (analise_estatistica_tempo.R)

### Script: `src/analise_estatistica_tempo.R`

Script R autocontido que realiza análise estatística completa da variável **tempo computacional** em delineamento RCBD.

**Estrutura do Script:**

Idêntica ao script de acurácia, substituindo `acuracia` por `tempo`:

1. **Setup e Carregamento**
   - Carrega `results/experimento_rcbd_resultados.csv`
   - Cria coluna `bloco` = paste(dataset, seed, sep="_")
   - Converte `metodo` e `bloco` para factor

2. **Modelo RCBD**
   - Ajusta: `aov(tempo ~ metodo + bloco, data=df)`
   - Variável resposta: **tempo** em segundos

3. **Gráfico QQ-Plot**
   - Salva em: `results/qqplot_tempo.png`
   - Visualiza normalidade dos resíduos

4. **Testes de Premissas**
   - **Shapiro-Wilk**: Normalidade dos resíduos
   - **Fligner-Killeen**: Homocedasticidade entre métodos

5. **Lógica Condicional (if/else)**
   - **Se Normal (p > 0.05):**
     - Executa ANOVA paramétrica: `summary(modelo)`
     - Se metodo significativo (p < 0.05): Tukey HSD post-hoc
   - **Se Não-Normal (p ≤ 0.05):**
     - Executa Friedman test: `friedman.test(tempo ~ metodo | bloco)`
     - Se significativo: Wilcoxon pareado com correção Bonferroni

6. **Estatísticas Descritivas**
   - Resumo por método: média, mediana, sd, min, max (em segundos)

7. **Saídas:**
   - **Console**: Mensagens de progresso
   - **Arquivo PNG**: `results/qqplot_tempo.png` (QQ-Plot)
   - **Arquivo TXT**: `results/relatorio_estatistico_tempo.txt` (Relatório)

**Nota sobre Tempo:**
- Variável típica com distribuição assimétrica positiva
- Script detecta automaticamente via Shapiro-Wilk e aplica teste apropriado
- Importante para avaliar custo computacional de cada método

**Como Executar:**
```r
source("src/analise_estatistica_tempo.R")
```

Ou no terminal:
```bash
Rscript src/analise_estatistica_tempo.R
```

