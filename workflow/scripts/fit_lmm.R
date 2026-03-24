#!/usr/bin/env Rscript

# ================================================================
#  FOOOF + Behavioural LMM Analysis
#  Snakemake-compatible
# ================================================================

library(lme4)
library(lmerTest)

# ================================================================
# Argument parsing
# ================================================================

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  value <- args[which(args == flag) + 1]
  if (length(value) == 0) stop(paste("Missing argument:", flag))
  value
}

has_arg <- function(flag) {
  any(args == flag)
}

trial_data_path <- if (has_arg("--trial-data")) get_arg("--trial-data") else NA_character_
fooof_path  <- if (has_arg("--fooof")) get_arg("--fooof") else NA_character_
events_path <- if (has_arg("--events")) get_arg("--events") else NA_character_
out_dir     <- get_arg("--out")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ================================================================
# Utility functions
# ================================================================

fit_lrt <- function(formula_null, formula_target, data) {
  vars <- all.vars(formula_target)
  data_sub <- data[complete.cases(data[, vars]), ]

  m_null   <- lmer(formula_null,   data = data_sub, REML = FALSE)
  m_target <- lmer(formula_target, data = data_sub, REML = FALSE)

  lrt <- anova(m_null, m_target)

  list(
    m_null = m_null,
    m_target = m_target,
    lrt = lrt,
    data = data_sub
  )
}

extract_summary_row <- function(test_name, fit_obj, coef_name = NULL) {

  chisq <- fit_obj$lrt$Chisq[2]
  df    <- fit_obj$lrt$Df[2]
  p     <- fit_obj$lrt$`Pr(>Chisq)`[2]

  beta <- se <- tval <- NA

  if (!is.null(coef_name)) {
    summ <- summary(fit_obj$m_target)$coefficients
    if (coef_name %in% rownames(summ)) {
      beta <- summ[coef_name, "Estimate"]
      se   <- summ[coef_name, "Std. Error"]
      tval <- summ[coef_name, "t value"]
    }
  }

  data.frame(
    test = test_name,
    beta = beta,
    se = se,
    t = tval,
    chisq = chisq,
    df = df,
    p = p
  )
}

# ================================================================
# Load data
# ================================================================

if (!is.na(trial_data_path) && nzchar(trial_data_path)) {
  df_long <- read.csv(trial_data_path, stringsAsFactors = FALSE)
} else {
  if (is.na(fooof_path) || is.na(events_path)) {
    stop("Provide either --trial-data or both --fooof and --events.")
  }

  fooof  <- read.csv(fooof_path,  stringsAsFactors = FALSE)
  events <- read.csv(events_path, stringsAsFactors = FALSE)

  df_long <- merge(
    fooof,
    events,
    by = c("dyad_id", "subject_id", "run", "event_id")
  )
}

df_long$subject_id <- as.factor(df_long$subject_id)
df_long$run <- as.numeric(df_long$run)
df_long$time_within_run <- as.numeric(df_long$time_within_run)
df_long$duration_s <- as.numeric(df_long$duration_s)

# ================================================================
# A) Behavioural Test
# ================================================================

summary_rows <- list()
model_objects <- list()

required_behavior_cols <- c("latency", "other_rate", "run", "time_within_run", "duration_s", "subject_id")
if (!all(required_behavior_cols %in% colnames(df_long))) {
  missing_cols <- required_behavior_cols[!(required_behavior_cols %in% colnames(df_long))]
  stop(paste("trial data missing required behavioral columns:", paste(missing_cols, collapse = ", ")))
}

fit_latency <- fit_lrt(
  latency ~ run + time_within_run + duration_s + (1 | subject_id),
  latency ~ other_rate + run + time_within_run + duration_s + (1 | subject_id),
  df_long
)

summary_rows[[length(summary_rows)+1]] <-
  extract_summary_row(
    "latency_other_rate",
    fit_latency,
    "other_rate"
  )
model_objects$latency <- fit_latency

# ================================================================
# Prepare wide neural dataframe
# ================================================================

if (all(c("roi", "offset", "exponent", "theta_power", "alpha_power", "beta_power", "self_rate") %in% colnames(df_long))) {
  neural_vars <- c(
    "offset", "exponent",
    "theta_power", "alpha_power", "beta_power"
  )

  meta_vars <- c(
    "subject_id", "run", "event_id",
    "self_rate", "other_rate",
    "time_within_run", "duration_s"
  )

  df_neural <- df_long[, c("subject_id","run","event_id","roi", neural_vars)]
  df_meta   <- unique(df_long[, meta_vars])

  df_wide_neural <- reshape(
    df_neural,
    idvar = c("subject_id","run","event_id"),
    timevar = "roi",
    direction = "wide"
  )

  colnames(df_wide_neural) <- gsub("\\.", "_", colnames(df_wide_neural))

  df_wide <- merge(df_wide_neural, df_meta,
                   by = c("subject_id","run","event_id"))

  # replace missing power with 0
  power_cols <- grep("power_", colnames(df_wide), value = TRUE)
  for (v in power_cols) df_wide[[v]][is.na(df_wide[[v]])] <- 0

  # standardize predictors
  scale_vars <- grep("offset_|exponent_|power_", colnames(df_wide), value = TRUE)
  for (v in scale_vars)
    df_wide[[v]] <- as.numeric(scale(df_wide[[v]]))

  for (v in scale_vars)
    df_wide[[paste0(v, "_sq")]] <- df_wide[[v]]^2

  df_wide$subject_id <- as.factor(df_wide$subject_id)

  controls <- c("run","time_within_run","duration_s")

  # ================================================================
  # B) Block Tests
  # ================================================================

  N_A <- c("offset_ANTERIOR","offset_POSTERIOR",
           "exponent_ANTERIOR","exponent_POSTERIOR")

  N_P <- c("theta_power_ANTERIOR","theta_power_POSTERIOR",
           "alpha_power_ANTERIOR","alpha_power_POSTERIOR",
           "beta_power_ANTERIOR","beta_power_POSTERIOR")

  fit_block_A <- fit_lrt(
    as.formula(paste("self_rate ~", paste(controls, collapse="+"),
                     "+ (1|subject_id)")),
    as.formula(paste("self_rate ~",
                     paste(c(N_A, controls), collapse="+"),
                     "+ (1|subject_id)")),
    df_wide
  )

  summary_rows[[length(summary_rows)+1]] <-
    extract_summary_row("block_aperiodic", fit_block_A)
  model_objects$block_A <- fit_block_A

  fit_block_P <- fit_lrt(
    as.formula(paste("self_rate ~", paste(controls, collapse="+"),
                     "+ (1|subject_id)")),
    as.formula(paste("self_rate ~",
                     paste(c(N_P, controls), collapse="+"),
                     "+ (1|subject_id)")),
    df_wide
  )

  summary_rows[[length(summary_rows)+1]] <-
    extract_summary_row("block_periodic", fit_block_P)
  model_objects$block_P <- fit_block_P

  # ================================================================
  # C) Quadratic Test (Aperiodic only shown)
  # ================================================================

  N_A_sq <- paste0(N_A, "_sq")

  fit_quad_A <- fit_lrt(
    as.formula(paste("self_rate ~",
                     paste(c(N_A, controls), collapse="+"),
                     "+ (1|subject_id)")),
    as.formula(paste("self_rate ~",
                     paste(c(N_A, N_A_sq, controls), collapse="+"),
                     "+ (1|subject_id)")),
    df_wide
  )

  summary_rows[[length(summary_rows)+1]] <-
    extract_summary_row("quad_aperiodic", fit_quad_A)
  model_objects$quad_A <- fit_quad_A
}

# ================================================================
# Save results
# ================================================================

summary_df <- do.call(rbind, summary_rows)

write.csv(
  summary_df,
  file = file.path(out_dir, "summary.csv"),
  row.names = FALSE
)

saveRDS(
  model_objects,
  file = file.path(out_dir, "full_models.rds")
)

cat("Analysis complete.\n")
