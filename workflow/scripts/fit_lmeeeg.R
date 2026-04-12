suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(lme4)
  library(lmerTest)
})

# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_CONTROL_CANDIDATES <- c("latency", "duration_s", "time_within_run", "run")
DEFAULT_BIG_FIVE_CANDIDATES <- c(
  "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
  "big5_openness", "big5_conscientiousness", "big5_extraversion", "big5_agreeableness", "big5_neuroticism",
  "bfi_openness", "bfi_conscientiousness", "bfi_extraversion", "bfi_agreeableness", "bfi_neuroticism"
)

# ==============================================================================
# Argument parsing
# ==============================================================================

parse_cli_args <- function() {
  raw_args <- commandArgs(trailingOnly = TRUE)

  if (length(raw_args) %% 2 != 0) {
    stop("Arguments must be supplied as --key value pairs.", call. = FALSE)
  }

  parsed_args <- list()
  if (length(raw_args) == 0) {
    return(parsed_args)
  }

  key_indices <- seq(1, length(raw_args), by = 2)
  for (index in key_indices) {
    key <- raw_args[[index]]
    value <- raw_args[[index + 1]]
    key <- sub("^--", "", key)
    parsed_args[[key]] <- value
  }

  parsed_args
}

get_arg_value <- function(args, name, default = NULL, required = FALSE) {
  if (!is.null(args[[name]])) {
    return(args[[name]])
  }
  if (required) {
    stop(sprintf("Missing required argument --%s", name), call. = FALSE)
  }
  default
}

parse_optional_csv_list <- function(x) {
  if (is.null(x) || is.na(x) || x == "") {
    return(character(0))
  }
  str_split(x, pattern = ",", simplify = FALSE)[[1]] %>%
    str_trim() %>%
    discard(~ .x == "")
}

# ==============================================================================
# General helpers
# ==============================================================================

safe_as_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

ensure_columns_exist <- function(data, required_columns) {
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      sprintf("Missing required columns: %s", paste(missing_columns, collapse = ", ")),
      call. = FALSE
    )
  }
}

compact_formula_string <- function(formula_object) {
  paste(deparse(formula_object), collapse = " ")
}

coalesce_existing_column <- function(data, candidates, default = NULL) {
  matches <- candidates[candidates %in% names(data)]
  if (length(matches) > 0L) {
    return(matches[[1]])
  }
  default
}

filter_complete_model_rows <- function(data, model_variables) {
  model_variables <- unique(model_variables)
  data %>% filter(if_all(all_of(model_variables), ~ !is.na(.x)))
}

# ==============================================================================
# Preprocessing integration
# ==============================================================================

load_preprocessed_data <- function(input_path, preprocess_script_path) {
  source(preprocess_script_path)

  raw_events <- load_events_data(input_path)
  cleaned_events <- preprocess_events_data(raw_events)
  partner_events <- make_partner_events(cleaned_events)

  list(
    raw_events = raw_events,
    cleaned_events = cleaned_events,
    partner_events = partner_events
  )
}

# ==============================================================================
# Variable selection
# ==============================================================================

select_personality_columns <- function(data, requested_personality_columns = character(0)) {
  if (length(requested_personality_columns) > 0L) {
    found_columns <- requested_personality_columns[requested_personality_columns %in% names(data)]
    missing_columns <- setdiff(requested_personality_columns, found_columns)
    if (length(missing_columns) > 0L) {
      warning(sprintf(
        "Requested personality columns not found and will be skipped: %s",
        paste(missing_columns, collapse = ", ")
      ))
    }
    return(found_columns)
  }

  unique(DEFAULT_BIG_FIVE_CANDIDATES[DEFAULT_BIG_FIVE_CANDIDATES %in% names(data)])
}

resolve_dyad_diff_column <- function(data, requested_dyad_diff_col = NULL) {
  if (!is.null(requested_dyad_diff_col) && requested_dyad_diff_col %in% names(data)) {
    return(requested_dyad_diff_col)
  }

  candidate_columns <- c("dyad_diff", "dyad_difference", "partner_diff", "mean_rate_diff")
  coalesce_existing_column(data, candidate_columns, default = NULL)
}

get_controls_for_model <- function(outcome_name, predictor_name, available_columns) {
  DEFAULT_CONTROL_CANDIDATES[
    DEFAULT_CONTROL_CANDIDATES %in% available_columns &
      !(DEFAULT_CONTROL_CANDIDATES %in% c(outcome_name, predictor_name))
  ]
}

# ==============================================================================
# Formula builders
# ==============================================================================

make_lmm_formula <- function(outcome_name, predictor_name, controls, random_effect_terms) {
  fixed_terms <- c(predictor_name, controls)
  rhs_terms <- c(fixed_terms, random_effect_terms)
  stats::as.formula(sprintf("%s ~ %s", outcome_name, paste(rhs_terms, collapse = " + ")))
}

make_null_formula <- function(outcome_name, controls, random_effect_terms) {
  rhs_terms <- c(controls, random_effect_terms)
  stats::as.formula(sprintf("%s ~ %s", outcome_name, paste(rhs_terms, collapse = " + ")))
}

# ==============================================================================
# Model fitting and extraction
# ==============================================================================

fit_nested_lmm <- function(data, outcome_name, predictor_name, controls, random_effect_terms) {
  full_formula <- make_lmm_formula(
    outcome_name = outcome_name,
    predictor_name = predictor_name,
    controls = controls,
    random_effect_terms = random_effect_terms
  )
  null_formula <- make_null_formula(
    outcome_name = outcome_name,
    controls = controls,
    random_effect_terms = random_effect_terms
  )

  model_variables <- all.vars(full_formula)
  analysis_data <- filter_complete_model_rows(data, model_variables)

  full_model <- lmerTest::lmer(
    formula = full_formula,
    data = analysis_data,
    REML = FALSE
  )
  null_model <- lmerTest::lmer(
    formula = null_formula,
    data = analysis_data,
    REML = FALSE
  )

  list(
    analysis_data = analysis_data,
    full_model = full_model,
    null_model = null_model,
    full_formula = full_formula,
    null_formula = null_formula
  )
}

extract_fixed_effect_row <- function(full_model, predictor_name) {
  model_summary <- summary(full_model)
  coefficient_table <- as.data.frame(model_summary$coefficients)
  coefficient_table$term <- rownames(coefficient_table)

  if (!(predictor_name %in% coefficient_table$term)) {
    return(tibble(
      term = predictor_name,
      beta = NA_real_,
      std_error = NA_real_,
      df = NA_real_,
      t_value = NA_real_,
      p_value = NA_real_
    ))
  }

  predictor_row <- coefficient_table %>% filter(term == predictor_name)

  p_column_candidates <- intersect(c("Pr(>|t|)", "Pr(>|z|)"), names(predictor_row))
  p_column_name <- if (length(p_column_candidates) > 0L) p_column_candidates[[1]] else NULL

  tibble(
    term = predictor_name,
    beta = predictor_row$Estimate[[1]],
    std_error = predictor_row$`Std. Error`[[1]],
    df = if ("df" %in% names(predictor_row)) predictor_row$df[[1]] else NA_real_,
    t_value = if ("t value" %in% names(predictor_row)) predictor_row$`t value`[[1]] else NA_real_,
    p_value = if (!is.null(p_column_name)) predictor_row[[p_column_name]][[1]] else NA_real_
  )
}

extract_lrt_row <- function(null_model, full_model) {
  lrt_table <- as.data.frame(anova(null_model, full_model))
  comparison_row <- lrt_table[nrow(lrt_table), , drop = FALSE]

  tibble(
    lrt_chisq = if ("Chisq" %in% names(comparison_row)) comparison_row$Chisq[[1]] else NA_real_,
    lrt_df = if ("Chi Df" %in% names(comparison_row)) comparison_row$`Chi Df`[[1]] else NA_real_,
    lrt_p_value = if ("Pr(>Chisq)" %in% names(comparison_row)) comparison_row$`Pr(>Chisq)`[[1]] else NA_real_
  )
}

summarise_lmm_result <- function(model_id, model_family, outcome_name, predictor_name, controls, random_effect_terms, fitted_models, note = NA_character_) {
  fixed_effect_row <- extract_fixed_effect_row(
    full_model = fitted_models$full_model,
    predictor_name = predictor_name
  )
  lrt_row <- extract_lrt_row(
    null_model = fitted_models$null_model,
    full_model = fitted_models$full_model
  )

  tibble(
    analysis_id = model_id,
    analysis_family = model_family,
    test_type = "nested_lmm_lrt",
    outcome = outcome_name,
    predictor = predictor_name,
    term = fixed_effect_row$term,
    beta = fixed_effect_row$beta,
    std_error = fixed_effect_row$std_error,
    df = fixed_effect_row$df,
    t_value = fixed_effect_row$t_value,
    p_value = fixed_effect_row$p_value,
    lrt_chisq = lrt_row$lrt_chisq,
    lrt_df = lrt_row$lrt_df,
    lrt_p_value = lrt_row$lrt_p_value,
    aic_null = AIC(fitted_models$null_model),
    aic_full = AIC(fitted_models$full_model),
    bic_null = BIC(fitted_models$null_model),
    bic_full = BIC(fitted_models$full_model),
    loglik_null = as.numeric(logLik(fitted_models$null_model)),
    loglik_full = as.numeric(logLik(fitted_models$full_model)),
    n_obs = nrow(fitted_models$analysis_data),
    n_subjects = dplyr::n_distinct(fitted_models$analysis_data$subject_id),
    n_dyads = dplyr::n_distinct(fitted_models$analysis_data$dyad_id),
    controls = paste(controls, collapse = "; "),
    random_effects = paste(random_effect_terms, collapse = " + "),
    null_formula = compact_formula_string(fitted_models$null_formula),
    full_formula = compact_formula_string(fitted_models$full_formula),
    note = note
  )
}

run_single_lmm_analysis <- function(data, model_id, model_family, outcome_name, predictor_name, random_effect_terms, note = NA_character_) {
  available_columns <- names(data)
  controls <- get_controls_for_model(
    outcome_name = outcome_name,
    predictor_name = predictor_name,
    available_columns = available_columns
  )

  tryCatch({
    fitted_models <- fit_nested_lmm(
      data = data,
      outcome_name = outcome_name,
      predictor_name = predictor_name,
      controls = controls,
      random_effect_terms = random_effect_terms
    )

    summarise_lmm_result(
      model_id = model_id,
      model_family = model_family,
      outcome_name = outcome_name,
      predictor_name = predictor_name,
      controls = controls,
      random_effect_terms = random_effect_terms,
      fitted_models = fitted_models,
      note = note
    )
  }, error = function(error_condition) {
    tibble(
      analysis_id = model_id,
      analysis_family = model_family,
      test_type = "nested_lmm_lrt",
      outcome = outcome_name,
      predictor = predictor_name,
      term = predictor_name,
      beta = NA_real_,
      std_error = NA_real_,
      df = NA_real_,
      t_value = NA_real_,
      p_value = NA_real_,
      lrt_chisq = NA_real_,
      lrt_df = NA_real_,
      lrt_p_value = NA_real_,
      aic_null = NA_real_,
      aic_full = NA_real_,
      bic_null = NA_real_,
      bic_full = NA_real_,
      loglik_null = NA_real_,
      loglik_full = NA_real_,
      n_obs = NA_real_,
      n_subjects = NA_real_,
      n_dyads = NA_real_,
      controls = paste(controls, collapse = "; "),
      random_effects = paste(random_effect_terms, collapse = " + "),
      null_formula = compact_formula_string(make_null_formula(outcome_name, controls, random_effect_terms)),
      full_formula = compact_formula_string(make_lmm_formula(outcome_name, predictor_name, controls, random_effect_terms)),
      note = sprintf("%s | ERROR: %s", note, error_condition$message)
    )
  })
}

run_core_analyses <- function(partner_events) {
  results <- list()

  results[[length(results) + 1L]] <- run_single_lmm_analysis(
    data = partner_events,
    model_id = "self_rate_by_other_rate",
    model_family = "core",
    outcome_name = "self_rate",
    predictor_name = "other_rate",
    random_effect_terms = c("(1 + other_rate | subject_id)", "(1 | dyad_id)"),
    note = "Local speech-rate adaptation"
  )

  results[[length(results) + 1L]] <- run_single_lmm_analysis(
    data = partner_events,
    model_id = "latency_by_other_rate",
    model_family = "core",
    outcome_name = "latency",
    predictor_name = "other_rate",
    random_effect_terms = c("(1 + other_rate | subject_id)", "(1 | dyad_id)"),
    note = "Turn-transition timing"
  )

  bind_rows(results)
}

run_personality_analyses <- function(partner_events, personality_columns) {
  if (length(personality_columns) == 0L) {
    return(tibble())
  }

  results <- map_dfr(personality_columns, function(personality_column) {
    bind_rows(
      run_single_lmm_analysis(
        data = partner_events,
        model_id = sprintf("self_rate_by_%s", personality_column),
        model_family = "personality_self_rate",
        outcome_name = "self_rate",
        predictor_name = personality_column,
        random_effect_terms = c("(1 | subject_id)", "(1 | dyad_id)"),
        note = "Big Five main effect on self-rate"
      )
      
    )
  })

  self_rate_mask <- results$analysis_family == "personality_self_rate"
  results$fdr_p_value <- NA_real_
  results$fdr_p_value[self_rate_mask] <- p.adjust(results$lrt_p_value[self_rate_mask], method = "fdr")
  results
}

# ==============================================================================
# Main
# ==============================================================================

main <- function() {
  args <- parse_cli_args()

  input_path <- get_arg_value(args, "input", required = TRUE)
  preprocess_script_path <- get_arg_value(args, "preprocess_script", required = TRUE)
  output_path <- get_arg_value(args, "output", required = TRUE)
  personality_cols <- parse_optional_csv_list(get_arg_value(args, "personality_cols", default = NULL))

  loaded_data <- load_preprocessed_data(
    input_path = input_path,
    preprocess_script_path = preprocess_script_path
  )

  partner_events <- loaded_data$partner_events
  ensure_columns_exist(partner_events, c("subject_id", "dyad_id", "self_rate", "other_rate", "latency", "run", "time_within_run"))
  personality_columns <- select_personality_columns(partner_events, requested_personality_columns = personality_cols)

  core_results <- run_core_analyses(partner_events = partner_events)

  personality_results <- run_personality_analyses(
    partner_events = partner_events,
    personality_columns = personality_columns
  )

  combined_results <- bind_rows(
    core_results,
    personality_results
  ) %>%
    mutate(
      fdr_p_value = if ("fdr_p_value" %in% names(.)) fdr_p_value else NA_real_
    ) %>%
    mutate(
      predictor_category = case_when(
        analysis_family == "null_test" ~ "null",
        analysis_family == "core" ~ "core",
        analysis_family == "dyad_difference" ~ "dyad_diff",
        str_detect(analysis_family, "^personality") ~ "personality",
        TRUE ~ "other"
      )
    ) %>%
    select(
      analysis_id,
      analysis_family,
      predictor_category,
      test_type,
      outcome,
      predictor,
      term,
      beta,
      std_error,
      df,
      t_value,
      p_value,
      lrt_chisq,
      lrt_df,
      lrt_p_value,
      fdr_p_value,
      aic_null,
      aic_full,
      bic_null,
      bic_full,
      loglik_null,
      loglik_full,
      n_obs,
      n_subjects,
      n_dyads,
      note,
      controls,
      random_effects,
      null_formula,
      full_formula
    )

  readr::write_csv(combined_results, output_path)
}

main()
