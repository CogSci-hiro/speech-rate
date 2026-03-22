#!/usr/bin/env Rscript

# ================================================================
# Speech Rate Convergence – Bayesian Model Tests
# ================================================================
#install.packages(c("brms", "optparse"), repos = "http://cran.us.r-project.org")
library(brms)
library(dplyr)
library(readr)
library(optparse)
library(lme4)

# ------------------------------------------------
# CLI
# ------------------------------------------------

option_list <- list(
  make_option("--events"),
  make_option("--phase"),
  make_option("--out_dir"),
  make_option("--preprocess")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Load preprocessing helpers
source(opt$preprocess)

opt <- parse_args(OptionParser(option_list = option_list))

events_path <- opt$events
phase_path  <- opt$phase
out_dir     <- opt$out_dir

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------
# Load data
# ------------------------------------------------

events_df <- read_csv(events_path)
phase_df  <- read_csv(phase_path)

df <- merge_datasets(events_df, phase_df)
df <- clean_data(df)

# ------------------------------------------------
# Containers
# ------------------------------------------------

model_rows <- list()
coef_rows <- list()

# ------------------------------------------------
# Helper
# ------------------------------------------------

extract_results <- function(test_id, model_null, model_alt, df_model) {

  bic_null <- BIC(model_null)
  bic_alt  <- BIC(model_alt)

  bf10 <- exp((bic_null - bic_alt) / 2)

  comp <- data.frame(
    test_id = test_id,
    formula_null = paste(deparse(formula(model_null)), collapse = ""),
    formula_alt  = paste(deparse(formula(model_alt)), collapse = ""),
    BF10 = bf10,
    n = nrow(df_model),
    subjects = length(unique(df_model$subject_id))
  )

  summ <- summary(model_alt)

  coefs <- as.data.frame(summ$coefficients)

  coef_df <- data.frame(
    test_id = test_id,
    term = rownames(coefs),
    beta = coefs$Estimate,
    se = coefs$`Std. Error`,
    t = coefs$`t value`,
    p = NA
  )

  return(list(comp = comp, coef = coef_df))
}

# ================================================================
# Test 1: Convergence vs Phase Continuation
# ================================================================

df3 <- prepare_cos_test(df)

m0 <- lmer(
  abs_diff ~ other_rate +
    latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df3,
  REML = FALSE
)

m1 <- lmer(
  abs_diff ~ cos_residual * other_rate +
    latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df3,
  REML = FALSE
)

res <- extract_results("phase_continuation", m0, m1, df3)

model_rows[[1]] <- res$comp
coef_rows[[1]]  <- res$coef

# ================================================================
# Test 2: Convergence vs Angular Velocity
# ================================================================

df4 <- prepare_omega_test(df)

m0 <- lmer(
  abs_diff ~ other_rate + latency + duration_s + run + time_within_run +
    (1|subject_id),
  data = df4,
  REML = FALSE
)

m1 <- lmer(
  abs_diff ~ omega * other_rate + latency + duration_s + run +
    time_within_run + (1|subject_id),
  data = df4,
  REML = FALSE
)

res <- extract_results("omega_convergence", m0, m1, df4)

model_rows[[2]] <- res$comp
coef_rows[[2]]  <- res$coef

# ================================================================
# Test 3: Convergence vs Phase Window
# ================================================================

df5 <- prepare_phase_window_test(df)

m0 <- lmer(
  self_rate ~ other_rate + phase_window +
    abs_latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df5,
  REML = FALSE
)

m1 <- lmer(
  self_rate ~ other_rate * phase_window +
    abs_latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df5,
  REML = FALSE
)

res <- extract_results("phase_window_convergence", m0, m1, df5)

model_rows[[3]] <- res$comp
coef_rows[[3]]  <- res$coef

# ================================================================
# Test 6: Convergence vs Continuous Latency
# ================================================================

df6 <- prepare_latency_continuous_test(df)

m0 <- lmer(
  self_rate ~ other_rate +
    abs_latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df6,
  REML = FALSE
)

m1 <- lmer(
  self_rate ~ other_rate * abs_latency +
    duration_s + run + time_within_run +
    (1 | subject_id),
  data = df6,
  REML = FALSE
)

res <- extract_results("latency_convergence", m0, m1, df6)

model_rows[[6]] <- res$comp
coef_rows[[6]]  <- res$coef

# ================================================================
# Test 7: Self Rate ~ Theta CF
# ================================================================

df7 <- prepare_self_rate_theta_cf_test(df)

m0 <- lmer(
  self_rate ~ latency + duration_s + run + time_within_run + (1 | subject_id),
  data = df7,
  REML = FALSE
)

m1 <- lmer(
  self_rate ~ theta_cf + latency + duration_s + run + time_within_run +
    (1 | subject_id),
  data = df7,
  REML = FALSE
)

res <- extract_results("self_rate_theta_cf", m0, m1, df7)

model_rows[[7]] <- res$comp
coef_rows[[7]]  <- res$coef

# ------------------------------------------------
# Export
# ------------------------------------------------

model_df <- bind_rows(model_rows)
coef_df  <- bind_rows(coef_rows)

write_csv(
  model_df,
  file.path(out_dir, "model_comparisons.csv")
)

write_csv(
  coef_df,
  file.path(out_dir, "coefficients.csv")
)
