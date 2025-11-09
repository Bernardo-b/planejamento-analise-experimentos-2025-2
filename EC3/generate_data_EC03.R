#!/usr/bin/env Rscript
# ============================================================================
# Script de Geração de Dados - EC03 (Otimizado com Paralelização)
# Equipe F - Estudo de Caso 03: Comparação de Configurações DE
# Formato: Long (uma linha por experimento)
# ============================================================================

# ==============================================================================
# 1. SETUP INICIAL E PACOTES
# ==============================================================================
cat(">>> INICIANDO SETUP <<<\n")

# Configurar CRAN
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Pacotes necessários
pkgs <- c("ExpDE", "smoof", "parallel")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs)) {
  cat("Instalando pacotes:", paste(new_pkgs, collapse = ", "), "\n")
  install.packages(new_pkgs, quiet = TRUE)
}

invisible(lapply(pkgs, library, character.only = TRUE))

# Seed para reprodutibilidade
set.seed(42)

cat("Pacotes carregados com sucesso!\n\n")

# ==============================================================================
# 2. DEFINIÇÃO DE PARÂMETROS
# ==============================================================================
cat(">>> PLANEJAMENTO EXPERIMENTAL <<<\n")

# Parâmetros estatísticos
ALPHA <- 0.05         # Significância (5%)
POWER <- 0.80         # Poder (80%)
EFFECT_SIZE <- 0.5    # d de Cohen

# Parâmetros de simulação
N_REPETICOES <- 30    # Repetições por dimensão/configuração

# Cálculo do número de blocos (dimensões) necessários
pwr_res <- pwr::pwr.t.test(
  d = EFFECT_SIZE,
  sig.level = ALPHA,
  power = POWER,
  type = "paired",
  alternative = "two.sided"
)
N_BLOCOS <- ceiling(pwr_res$n)

# Sorteio das dimensões
dims_teste <- sort(sample(2:150, N_BLOCOS, replace = FALSE))

cat("Alpha (significância):", ALPHA, "\n")
cat("Power (potência):", POWER, "\n")
cat("Effect size (d de Cohen):", EFFECT_SIZE, "\n")
cat("Número de blocos (dimensões):", N_BLOCOS, "\n")
cat("Repetições por configuração:", N_REPETICOES, "\n")
cat("Dimensões sorteadas:", paste(dims_teste, collapse = ", "), "\n")
cat("Total de experimentos:", N_BLOCOS * 2 * N_REPETICOES, "\n\n")

# Parâmetros do Algoritmo (Equipe F)
recpars1 <- list(name = "recombination_blxAlphaBeta", alpha = 0, beta = 0)
mutpars1 <- list(name = "mutation_rand", f = 4)

recpars2 <- list(name = "recombination_exp", cr = 0.6)
mutpars2 <- list(name = "mutation_best", f = 2)

selpars <- list(name = "selection_standard")

# ==============================================================================
# 3. CONFIGURAÇÃO DE PARALELIZAÇÃO
# ==============================================================================
cat(">>> CONFIGURANDO PARALELIZAÇÃO <<<\n")

n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1

cat("Cores disponíveis:", parallel::detectCores(), "\n")
cat("Cores para usar:", n_cores, "\n\n")

# ==============================================================================
# 4. FUNÇÃO WRAPPER PARA EXECUÇÃO DE UM EXPERIMENTO
# ==============================================================================

run_experiment <- function(dim, config, run) {
  # Criar função Rosenbrock
  smoof_fn <- smoof::makeRosenbrockFunction(dimensions = dim)

  fn_wrap <- function(X) {
    if (!is.matrix(X)) X <- matrix(X, nrow = 1)
    apply(X, 1, smoof_fn)
  }

  # Parâmetros dependentes da dimensão
  probpars <- list(
    name = "fn",
    xmin = rep(-5, dim),
    xmax = rep(10, dim)
  )

  stopcrit <- list(
    list(name = "stop_maxeval",
         maxevals = 5000 * dim,
         maxiter = 100 * dim)
  )

  pop_sz <- 5 * dim

  # Selecionar configuração
  if (config == "cfg1") {
    mutpars <- mutpars1
    recpars <- recpars1
  } else {
    mutpars <- mutpars2
    recpars <- recpars2
  }

  # Executar ExpDE (silenciosamente)
  invisible(capture.output({
    out <- ExpDE::ExpDE(
      mutpars = mutpars,
      recpars = recpars,
      popsize = pop_sz,
      selpars = selpars,
      stopcrit = stopcrit,
      probpars = probpars,
      showpars = list(show.iters = "none")
    )
  }))

  # Retornar resultado
  return(data.frame(
    dim = dim,
    config = config,
    run = run,
    Fbest = out$Fbest,
    stringsAsFactors = FALSE
  ))
}

# ==============================================================================
# 5. LOOP DE EXECUÇÃO (COM PROGRESSO)
# ==============================================================================
cat(">>> INICIANDO EXECUÇÃO <<<\n\n")

# Criar cluster para paralelização
cl <- parallel::makeCluster(n_cores, type = "FORK")
parallel::clusterSetRNGStream(cl, iseed = 42)

# Exportar funções e variáveis para o cluster
parallel::clusterExport(
  cl,
  c("mutpars1", "mutpars2", "recpars1", "recpars2", "selpars"),
  envir = environment()
)

# Lista com todos os experimentos a rodar
experiments <- list()
idx <- 1

for (dim in dims_teste) {
  for (config in c("cfg1", "cfg2")) {
    for (run in 1:N_REPETICOES) {
      experiments[[idx]] <- list(dim = dim, config = config, run = run)
      idx <- idx + 1
    }
  }
}

cat("Total de experimentos para rodar:", length(experiments), "\n")
cat("Iniciando...")

# Executar em paralelo com feedback de progresso
start_time <- Sys.time()
n_exp <- length(experiments)

resultados <- parallel::parLapply(
  cl,
  experiments,
  function(exp) {
    result <- tryCatch(
      run_experiment(exp$dim, exp$config, exp$run),
      error = function(e) {
        cat("\nERRO em dim=", exp$dim, ", config=", exp$config, ", run=", exp$run, "\n")
        return(NULL)
      }
    )
    return(result)
  }
)

# Parar cluster
parallel::stopCluster(cl)

# Remover NULLs (erros)
resultados <- resultados[!sapply(resultados, is.null)]

elapsed_time <- Sys.time() - start_time

cat("\n\nTempo total de execução:",
    format(elapsed_time, digits = 2), "\n")
cat("Experimentos concluídos:", length(resultados), "/", n_exp, "\n\n")

# ==============================================================================
# 6. CONSOLIDAR RESULTADOS
# ==============================================================================
cat(">>> CONSOLIDANDO DADOS <<<\n")

dados_final <- do.call(rbind, resultados)
rownames(dados_final) <- NULL

cat("Dimensões do dataset:", nrow(dados_final), "x", ncol(dados_final), "\n")
cat("Colunas:", paste(names(dados_final), collapse = ", "), "\n\n")

# Validações
cat("Validações:\n")
cat("- Dimensões únicas:", length(unique(dados_final$dim)), "\n")
cat("- Configurações:", paste(unique(dados_final$config), collapse = ", "), "\n")
cat("- Runs por config/dim:",
    table(dados_final$config)[1], "\n")
cat("- Fbest min:", min(dados_final$Fbest), "\n")
cat("- Fbest max:", max(dados_final$Fbest), "\n\n")

# ==============================================================================
# 7. SALVAR DADOS
# ==============================================================================
cat(">>> SALVANDO DADOS <<<\n")

output_file <- "data.csv"
write.csv(dados_final, output_file, row.names = FALSE)

cat("Arquivo salvo:", output_file, "\n")
cat("Linhas:", nrow(dados_final), "\n")
cat("Colunas:", ncol(dados_final), "\n\n")

cat(">>> SUCESSO! Dados gerados e salvos. <<<\n")
