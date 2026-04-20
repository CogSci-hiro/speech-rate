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

write_text_file <- function(text, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(text, con = output_path, useBytes = TRUE)
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

extract_model_convergence_diagnostics <- function(model) {
  optinfo <- model@optinfo
  conv_info <- optinfo$conv
  lme4_messages <- conv_info$lme4$messages
  optimizer_info <- conv_info$opt
  optimizer_messages <- if (is.list(optimizer_info)) optimizer_info$message else NULL

  derivs <- optinfo$derivs
  gradient_values <- derivs$gradient
  if (is.null(gradient_values)) {
    gradient_values <- derivs$grad
  }
  max_abs_gradient <- if (is.null(gradient_values)) {
    NA_real_
  } else {
    max(abs(as.numeric(gradient_values)), na.rm = TRUE)
  }
  if (!is.finite(max_abs_gradient)) {
    max_abs_gradient <- NA_real_
  }

  conv_messages <- c(lme4_messages, optimizer_messages)
  conv_messages <- conv_messages[!is.na(conv_messages) & nzchar(conv_messages)]

  optimizer_code_value <- if (is.list(optimizer_info)) {
    suppressWarnings(as.integer(optimizer_info$code))
  } else {
    suppressWarnings(as.integer(optimizer_info))
  }
  if (length(optimizer_code_value) == 0L || is.na(optimizer_code_value[[1]])) {
    optimizer_code_value <- NA_integer_
  } else {
    optimizer_code_value <- optimizer_code_value[[1]]
  }

  list(
    singular_fit = isTRUE(lme4::isSingular(model, tol = 1e-4)),
    convergence_warning = length(conv_messages) > 0L,
    convergence_ok = length(conv_messages) == 0L,
    convergence_message = if (length(conv_messages) > 0L) paste(conv_messages, collapse = " | ") else NA_character_,
    optimizer_code = optimizer_code_value,
    optimizer_message = if (length(optimizer_messages) > 0L) paste(optimizer_messages, collapse = " | ") else NA_character_,
    max_abs_gradient = max_abs_gradient
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
  full_convergence <- extract_model_convergence_diagnostics(fitted_models$full_model)
  null_convergence <- extract_model_convergence_diagnostics(fitted_models$null_model)

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
    full_singular_fit = full_convergence$singular_fit,
    full_convergence_ok = full_convergence$convergence_ok,
    full_convergence_warning = full_convergence$convergence_warning,
    full_convergence_message = full_convergence$convergence_message,
    full_optimizer_code = full_convergence$optimizer_code,
    full_optimizer_message = full_convergence$optimizer_message,
    full_max_abs_gradient = full_convergence$max_abs_gradient,
    null_singular_fit = null_convergence$singular_fit,
    null_convergence_ok = null_convergence$convergence_ok,
    null_convergence_warning = null_convergence$convergence_warning,
    null_convergence_message = null_convergence$convergence_message,
    null_optimizer_code = null_convergence$optimizer_code,
    null_optimizer_message = null_convergence$optimizer_message,
    null_max_abs_gradient = null_convergence$max_abs_gradient,
    controls = paste(controls, collapse = "; "),
    random_effects = paste(random_effect_terms, collapse = " + "),
    null_formula = compact_formula_string(fitted_models$null_formula),
    full_formula = compact_formula_string(fitted_models$full_formula),
    note = note
  )
}

format_lmm_summary_text <- function(model_id, model_family, outcome_name, predictor_name, controls, random_effect_terms, fitted_models, note = NA_character_) {
  full_convergence <- extract_model_convergence_diagnostics(fitted_models$full_model)
  null_convergence <- extract_model_convergence_diagnostics(fitted_models$null_model)

  paste(
    c(
      sprintf("analysis_id: %s", model_id),
      sprintf("analysis_family: %s", model_family),
      sprintf("outcome: %s", outcome_name),
      sprintf("predictor: %s", predictor_name),
      sprintf("controls: %s", paste(controls, collapse = "; ")),
      sprintf("random_effects: %s", paste(random_effect_terms, collapse = " + ")),
      sprintf("note: %s", note),
      sprintf("n_obs: %d", nrow(fitted_models$analysis_data)),
      sprintf("n_subjects: %d", dplyr::n_distinct(fitted_models$analysis_data$subject_id)),
      sprintf("n_dyads: %d", dplyr::n_distinct(fitted_models$analysis_data$dyad_id)),
      "",
      "full_model_diagnostics:",
      sprintf("  singular_fit: %s", full_convergence$singular_fit),
      sprintf("  convergence_ok: %s", full_convergence$convergence_ok),
      sprintf("  convergence_warning: %s", full_convergence$convergence_warning),
      sprintf("  convergence_message: %s", full_convergence$convergence_message),
      sprintf("  optimizer_code: %s", full_convergence$optimizer_code),
      sprintf("  optimizer_message: %s", full_convergence$optimizer_message),
      sprintf("  max_abs_gradient: %s", format(full_convergence$max_abs_gradient, digits = 8, scientific = FALSE)),
      "",
      "null_model_diagnostics:",
      sprintf("  singular_fit: %s", null_convergence$singular_fit),
      sprintf("  convergence_ok: %s", null_convergence$convergence_ok),
      sprintf("  convergence_warning: %s", null_convergence$convergence_warning),
      sprintf("  convergence_message: %s", null_convergence$convergence_message),
      sprintf("  optimizer_code: %s", null_convergence$optimizer_code),
      sprintf("  optimizer_message: %s", null_convergence$optimizer_message),
      sprintf("  max_abs_gradient: %s", format(null_convergence$max_abs_gradient, digits = 8, scientific = FALSE)),
      "",
      "full_model_summary:",
      capture.output(summary(fitted_models$full_model)),
      "",
      "null_model_summary:",
      capture.output(summary(fitted_models$null_model))
    ),
    collapse = "\n"
  )
}

build_error_result_row <- function(model_id, model_family, outcome_name, predictor_name, controls, random_effect_terms, note) {
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
    full_singular_fit = NA,
    full_convergence_ok = NA,
    full_convergence_warning = NA,
    full_convergence_message = NA_character_,
    full_optimizer_code = NA_integer_,
    full_optimizer_message = NA_character_,
    full_max_abs_gradient = NA_real_,
    null_singular_fit = NA,
    null_convergence_ok = NA,
    null_convergence_warning = NA,
    null_convergence_message = NA_character_,
    null_optimizer_code = NA_integer_,
    null_optimizer_message = NA_character_,
    null_max_abs_gradient = NA_real_,
    controls = paste(controls, collapse = "; "),
    random_effects = paste(random_effect_terms, collapse = " + "),
    null_formula = compact_formula_string(make_null_formula(outcome_name, controls, random_effect_terms)),
    full_formula = compact_formula_string(make_lmm_formula(outcome_name, predictor_name, controls, random_effect_terms)),
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

    list(
      result = summarise_lmm_result(
        model_id = model_id,
        model_family = model_family,
        outcome_name = outcome_name,
        predictor_name = predictor_name,
        controls = controls,
        random_effect_terms = random_effect_terms,
        fitted_models = fitted_models,
        note = note
      ),
      summary_text = format_lmm_summary_text(
        model_id = model_id,
        model_family = model_family,
        outcome_name = outcome_name,
        predictor_name = predictor_name,
        controls = controls,
        random_effect_terms = random_effect_terms,
        fitted_models = fitted_models,
        note = note
      )
    )
  }, error = function(error_condition) {
    error_note <- sprintf("%s | ERROR: %s", note, error_condition$message)
    list(
      result = build_error_result_row(
        model_id = model_id,
        model_family = model_family,
        outcome_name = outcome_name,
        predictor_name = predictor_name,
        controls = controls,
        random_effect_terms = random_effect_terms,
        note = error_note
      ),
      summary_text = paste(
        c(
          sprintf("analysis_id: %s", model_id),
          sprintf("analysis_family: %s", model_family),
          sprintf("outcome: %s", outcome_name),
          sprintf("predictor: %s", predictor_name),
          sprintf("controls: %s", paste(controls, collapse = "; ")),
          sprintf("random_effects: %s", paste(random_effect_terms, collapse = " + ")),
          sprintf("note: %s", error_note)
        ),
        collapse = "\n"
      )
    )
  })
}

run_core_analyses <- function(partner_events) {
  results <- list()
  summary_texts <- character()

  results[[length(results) + 1L]] <- run_single_lmm_analysis(
    data = partner_events,
    model_id = "self_rate_by_other_rate",
    model_family = "core",
    outcome_name = "self_rate",
    predictor_name = "other_rate",
    random_effect_terms = c("(1 | subject_id)", "(1 | dyad_id)"),
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

  summary_texts <- purrr::map_chr(results, "summary_text")

  list(
    results = bind_rows(purrr::map(results, "result")),
    summary_text = paste(summary_texts, collapse = "\n\n========================================\n\n")
  )
}

# ==============================================================================
# Main
# ==============================================================================

main <- function() {
  args <- parse_cli_args()

  input_path <- get_arg_value(args, "input", required = TRUE)
  preprocess_script_path <- get_arg_value(args, "preprocess_script", required = TRUE)
  output_path <- get_arg_value(args, "output", required = TRUE)
  model_summary_txt_path <- get_arg_value(args, "model_summary_txt", required = TRUE)

  loaded_data <- load_preprocessed_data(
    input_path = input_path,
    preprocess_script_path = preprocess_script_path
  )

  partner_events <- loaded_data$partner_events
  ensure_columns_exist(partner_events, c("subject_id", "dyad_id", "self_rate", "other_rate", "latency", "run", "time_within_run"))

  core_analysis <- run_core_analyses(partner_events = partner_events)

  combined_results <- core_analysis$results %>%
    mutate(
      fdr_p_value = if ("fdr_p_value" %in% names(.)) fdr_p_value else NA_real_
    ) %>%
    mutate(
      predictor_category = case_when(
        analysis_family == "null_test" ~ "null",
        analysis_family == "core" ~ "core",
        analysis_family == "dyad_difference" ~ "dyad_diff",
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
      full_singular_fit,
      full_convergence_ok,
      full_convergence_warning,
      full_convergence_message,
      full_optimizer_code,
      full_optimizer_message,
      full_max_abs_gradient,
      null_singular_fit,
      null_convergence_ok,
      null_convergence_warning,
      null_convergence_message,
      null_optimizer_code,
      null_optimizer_message,
      null_max_abs_gradient,
      note,
      controls,
      random_effects,
      null_formula,
      full_formula
    )

  readr::write_csv(combined_results, output_path)
  write_text_file(core_analysis$summary_text, model_summary_txt_path)
}

main()
