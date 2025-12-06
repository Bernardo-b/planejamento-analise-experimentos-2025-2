# Análise Estatística do Tempo Computacional - Experimento RCBD
# Comparação de 3 métodos de otimização de hiperparâmetros para SVM
# Data: 06/12/2025

rm(list = ls())

# ============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================================

df <- read.csv("results/experimento_rcbd_resultados.csv")

# Criar coluna 'bloco' como concatenação de dataset + seed
df$bloco <- paste(df$dataset, df$seed, sep = "_")

# Converter para fatores
df$metodo <- as.factor(df$metodo)
df$bloco <- as.factor(df$bloco)

cat("Dados carregados:\n")
cat("Dimensões:", nrow(df), "x", ncol(df), "\n")
cat("Métodos:", levels(df$metodo), "\n")
cat("Número de blocos:", nlevels(df$bloco), "\n")
cat("Blocos únicos:", length(unique(df$bloco)), "\n\n")

# ============================================================================
# 2. AJUSTE DO MODELO RCBD
# ============================================================================

cat("Ajustando modelo RCBD: tempo ~ metodo + bloco\n\n")
modelo <- aov(tempo ~ metodo + bloco, data = df)

# ============================================================================
# 3. GRÁFICO QQ-PLOT
# ============================================================================

cat("Gerando gráfico QQ-Plot dos resíduos...\n")
png("results/qqplot_tempo.png", width = 800, height = 600)
plot(modelo, 2)  # Plot 2 do modelo aov é o QQ-Plot
dev.off()
cat("Arquivo salvo: results/qqplot_tempo.png\n\n")

# ============================================================================
# 4. TESTES DE PREMISSAS E ANÁLISE ESTATÍSTICA
# ============================================================================

sink("results/relatorio_estatistico_tempo.txt")

cat("================================================================================\n")
cat("RELATÓRIO DE ANÁLISE ESTATÍSTICA - TEMPO COMPUTACIONAL (RCBD)\n")
cat("================================================================================\n\n")

cat("Dados Gerais:\n")
cat("- Número de observações:", nrow(df), "\n")
cat("- Métodos:", paste(levels(df$metodo), collapse = ", "), "\n")
cat("- Blocos (dataset_seed):", nlevels(df$bloco), "\n")
cat("- Experimentos: 5 datasets × 7 seeds × 3 métodos = 105\n\n")

# Teste de Normalidade (Shapiro-Wilk)
cat("--------------------------------------------------------------------------------\n")
cat("TESTE DE NORMALIDADE DOS RESÍDUOS (Shapiro-Wilk)\n")
cat("--------------------------------------------------------------------------------\n")
residuos <- residuals(modelo)
teste_shapiro <- shapiro.test(residuos)
print(teste_shapiro)
cat("\n")

# Teste de Homocedasticidade (Fligner-Killeen)
cat("--------------------------------------------------------------------------------\n")
cat("TESTE DE HOMOCEDASTICIDADE (Fligner-Killeen)\n")
cat("--------------------------------------------------------------------------------\n")
teste_fligner <- fligner.test(tempo ~ metodo, data = df)
print(teste_fligner)
cat("\n")

# ============================================================================
# LÓGICA CONDICIONAL: NORMAL vs NÃO-NORMAL
# ============================================================================

cat("--------------------------------------------------------------------------------\n")
cat("DECISÃO ESTATÍSTICA\n")
cat("--------------------------------------------------------------------------------\n")

if (teste_shapiro$p.value > 0.05) {

  # ========== CASO 1: DADOS NORMAIS ==========
  cat("✓ Premissa de normalidade SATISFEITA (p =", round(teste_shapiro$p.value, 4), ")\n")
  cat("Procedimento: ANOVA Paramétrica (Teste F)\n\n")

  cat("--------------------------------------------------------------------------------\n")
  cat("ANÁLISE DE VARIÂNCIA (ANOVA) - MODELO RCBD\n")
  cat("--------------------------------------------------------------------------------\n")
  print(summary(modelo))
  cat("\n")

  # Extrair p-valor do efeito 'metodo'
  summary_modelo <- summary(modelo)
  p_metodo <- summary_modelo[[1]]["metodo", "Pr(>F)"]

  if (p_metodo < 0.05) {
    cat("✓ Diferenças significativas entre métodos detectadas (p =", round(p_metodo, 4), ")\n")
    cat("Executando teste POST-HOC: Tukey HSD\n\n")

    cat("--------------------------------------------------------------------------------\n")
    cat("TESTE POST-HOC: TUKEY HSD (Comparações Pairwise)\n")
    cat("--------------------------------------------------------------------------------\n")
    print(TukeyHSD(modelo, "metodo"))
    cat("\n")
  } else {
    cat("✗ Nenhuma diferença significativa entre métodos (p =", round(p_metodo, 4), ")\n\n")
  }

} else {

  # ========== CASO 2: DADOS NÃO-NORMAIS ==========
  cat("✗ Premissa de normalidade VIOLADA (p =", round(teste_shapiro$p.value, 4), ")\n")
  cat("Aviso: Os dados não seguem distribuição normal.\n")
  cat("Procedimento: ANOVA Não-Paramétrica (Teste de Friedman)\n\n")

  cat("--------------------------------------------------------------------------------\n")
  cat("TESTE DE FRIEDMAN (ANOVA Não-Paramétrica com Blocos)\n")
  cat("--------------------------------------------------------------------------------\n")
  teste_friedman <- friedman.test(tempo ~ metodo | bloco, data = df)
  print(teste_friedman)
  cat("\n")

  if (teste_friedman$p.value < 0.05) {
    cat("✓ Diferenças significativas entre métodos detectadas (p =", round(teste_friedman$p.value, 4), ")\n")
    cat("Executando teste POST-HOC: Wilcoxon Pareado com Bonferroni\n\n")

    cat("--------------------------------------------------------------------------------\n")
    cat("TESTE POST-HOC: WILCOXON PAREADO (com correção Bonferroni)\n")
    cat("--------------------------------------------------------------------------------\n")
    teste_wilcox <- pairwise.wilcox.test(df$tempo, df$metodo,
                                         p.adjust.method = "bonferroni",
                                         paired = TRUE)
    print(teste_wilcox)
    cat("\n")
  } else {
    cat("✗ Nenhuma diferença significativa entre métodos (p =", round(teste_friedman$p.value, 4), ")\n\n")
  }
}

# ============================================================================
# ESTATÍSTICAS DESCRITIVAS ENRIQUECIDAS
# ============================================================================

cat("--------------------------------------------------------------------------------\n")
cat("ESTATÍSTICAS DESCRITIVAS ENRIQUECIDAS POR MÉTODO\n")
cat("--------------------------------------------------------------------------------\n\n")

# Criar dataframe com agregações
metodos <- levels(df$metodo)
resumo <- data.frame(
  Método = metodos,
  N = sapply(metodos, function(m) sum(df$metodo == m)),
  Média = sapply(metodos, function(m) mean(df$tempo[df$metodo == m])),
  Mediana = sapply(metodos, function(m) median(df$tempo[df$metodo == m])),
  SD = sapply(metodos, function(m) sd(df$tempo[df$metodo == m])),
  Q1 = sapply(metodos, function(m) quantile(df$tempo[df$metodo == m], 0.25)),
  Q3 = sapply(metodos, function(m) quantile(df$tempo[df$metodo == m], 0.75)),
  Min = sapply(metodos, function(m) min(df$tempo[df$metodo == m])),
  Max = sapply(metodos, function(m) max(df$tempo[df$metodo == m]))
)

# Calcular IQR
resumo$IQR <- resumo$Q3 - resumo$Q1

# Baseline: GridSearch
mediana_grid <- resumo$Mediana[resumo$Método == "GridSearch"]

# Calcular diferença absoluta e melhoria relativa em relação ao GridSearch
resumo$Diff_Absoluta <- resumo$Mediana - mediana_grid
resumo$Melhoria_Relativa_Pct <- ((resumo$Mediana - mediana_grid) / mediana_grid) * 100

# Tabela resumida para apresentação
tabela_apresentacao <- data.frame(
  Método = resumo$Método,
  Mediana_s = round(resumo$Mediana, 4),
  IQR_s = round(resumo$IQR, 4),
  Diff_Absoluta_s = round(resumo$Diff_Absoluta, 4),
  Melhoria_Pct = round(resumo$Melhoria_Relativa_Pct, 2)
)

cat("TABELA RESUMIDA - Comparação com Baseline (GridSearch):\n\n")
print(tabela_apresentacao, row.names = FALSE)

cat("\n\nTABELA COMPLETA - Todas as Estatísticas (em segundos):\n\n")
tabela_completa <- data.frame(
  Método = resumo$Método,
  N = resumo$N,
  Média = round(resumo$Média, 4),
  Mediana = round(resumo$Mediana, 4),
  SD = round(resumo$SD, 4),
  Q1 = round(resumo$Q1, 4),
  Q3 = round(resumo$Q3, 4),
  IQR = round(resumo$IQR, 4),
  Min = round(resumo$Min, 4),
  Max = round(resumo$Max, 4),
  Diff_Absoluta = round(resumo$Diff_Absoluta, 4),
  Melhoria_Pct = round(resumo$Melhoria_Relativa_Pct, 2)
)
print(tabela_completa, row.names = FALSE)

cat("\n\nINTERPRETAÇÃO DAS COLUNAS:\n")
cat("- Mediana: Valor central (robusto a outliers)\n")
cat("- IQR: Intervalo Interquartil (mede consistência - menor é melhor para tempo)\n")
cat("- Diff_Absoluta: Diferença da mediana em relação ao GridSearch (em segundos)\n")
cat("- Melhoria_Pct: Percentual de melhoria/piora em relação ao GridSearch\n")
cat("  Negativo = mais rápido que GridSearch (melhor!) | Positivo = mais lento\n")

cat("\n")
cat("================================================================================\n")
cat("FIM DO RELATÓRIO\n")
cat("================================================================================\n")

sink()

# ============================================================================
# 5. MENSAGEM DE CONCLUSÃO NO CONSOLE
# ============================================================================

cat("\n")
cat("✓ Análise completa realizada com sucesso!\n")
cat("✓ Gráficos salvos em: results/qqplot_tempo.png\n")
cat("✓ Relatório salvo em: results/relatorio_estatistico_tempo.txt\n")
cat("\n")
