suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(jsonlite)
  library(lme4)
  library(lmerTest)
})

# ==============================================================================
# Validation helpers
# ==============================================================================

stop_if_missing_columns <- function(data, required_columns) {
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      sprintf(
        "Missing required columns: %s",
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

safe_zscore <- function(x) {
  if (all(is.na(x))) {
    return(rep(NA_real_, length(x)))
  }

  standard_deviation <- stats::sd(x, na.rm = TRUE)
  if (is.na(standard_deviation) || standard_deviation == 0) {
    return(rep(0, length(x)))
  }

  (x - mean(x, na.rm = TRUE)) / standard_deviation
}

empirical_p_value <- function(observed_value, null_values, direction = c("two_sided", "less", "greater")) {
  direction <- match.arg(direction)
  null_values <- null_values[is.finite(null_values)]

  if (length(null_values) == 0L || !is.finite(observed_value)) {
    return(NA_real_)
  }

  if (direction == "less") {
    return((sum(null_values <= observed_value) + 1) / (length(null_values) + 1))
  }

  if (direction == "greater") {
    return((sum(null_values >= observed_value) + 1) / (length(null_values) + 1))
  }

  null_center <- mean(null_values)
  observed_distance <- abs(observed_value - null_center)
  null_distances <- abs(null_values - null_center)
  (sum(null_distances >= observed_distance) + 1) / (length(null_values) + 1)
}

# ==============================================================================
# Behavioural residual helpers
# ==============================================================================

BEHAVIOURAL_REQUIRED_COLUMNS <- c(
  "self_rate",
  "other_rate",
  "latency",
  "duration_s",
  "time_within_run",
  "run"
)
BEHAVIOURAL_SUBJECT_COLUMN_CANDIDATES <- c("subject_id", "subject")
BEHAVIOURAL_TRIAL_ID_COLUMN_CANDIDATES <- c("event_id", "trial_id")
BEHAVIOURAL_PREDICTION_MODES <- c("in_sample", "leave_one_subject_out")
BEHAVIOURAL_GATING_MODEL_LABEL <- "gating_residual_other_rate_subject_slope"
BEHAVIOURAL_GATING_MODEL_FORMULA <- "self_rate ~ other_rate + (1 + other_rate | %s)"

resolve_existing_column <- function(data, candidates, label) {
  matched_columns <- candidates[candidates %in% names(data)]
  if (length(matched_columns) == 0L) {
    stop(
      sprintf(
        "Missing required %s column. Checked: %s",
        label,
        paste(candidates, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  matched_columns[[1]]
}

get_behavioural_model_columns <- function(data) {
  subject_column <- resolve_existing_column(
    data = data,
    candidates = BEHAVIOURAL_SUBJECT_COLUMN_CANDIDATES,
    label = "subject"
  )
  trial_id_column <- BEHAVIOURAL_TRIAL_ID_COLUMN_CANDIDATES[
    BEHAVIOURAL_TRIAL_ID_COLUMN_CANDIDATES %in% names(data)
  ]

  list(
    subject_column = subject_column,
    trial_id_column = if (length(trial_id_column) > 0L) trial_id_column[[1]] else NULL
  )
}

assert_unique_key <- function(data, key_column, label) {
  if (anyDuplicated(data[[key_column]]) > 0L) {
    stop(
      sprintf("Expected `%s` to be unique in %s.", key_column, label),
      call. = FALSE
    )
  }
}

validate_behavioural_residual_inputs <- function(data) {
  stop_if_missing_columns(data, BEHAVIOURAL_REQUIRED_COLUMNS)
  get_behavioural_model_columns(data)
  invisible(TRUE)
}

make_behavioural_model_formula <- function(subject_column) {
  stats::as.formula(sprintf(BEHAVIOURAL_GATING_MODEL_FORMULA, subject_column))
}

fit_behavioural_residual_model <- function(data, subject_column = NULL) {
  validate_behavioural_residual_inputs(data)

  if (is.null(subject_column)) {
    subject_column <- get_behavioural_model_columns(data)$subject_column
  }

  gating_formula <- make_behavioural_model_formula(subject_column)
  gating_fit <- lmerTest::lmer(formula = gating_formula, data = data, REML = FALSE)

  list(
    model = gating_fit,
    formula = gating_formula,
    model_label = BEHAVIOURAL_GATING_MODEL_LABEL,
    singular_fit = isTRUE(lme4::isSingular(gating_fit, tol = 1e-4)),
    fallback_used = FALSE,
    fallback_reason = NA_character_
  )
}

extract_behavioural_fixed_effects <- function(model) {
  coefficient_table <- as.data.frame(summary(model)$coefficients)
  coefficient_table$term <- rownames(coefficient_table)
  statistic_column <- if ("t value" %in% names(coefficient_table)) "t value" else if ("z value" %in% names(coefficient_table)) "z value" else NULL
  p_value_column <- if ("Pr(>|t|)" %in% names(coefficient_table)) "Pr(>|t|)" else if ("Pr(>|z|)" %in% names(coefficient_table)) "Pr(>|z|)" else NULL

  tibble::as_tibble(coefficient_table) %>%
    transmute(
      term = .data$term,
      estimate = .data$Estimate,
      std_error = .data$`Std. Error`,
      statistic = if (!is.null(statistic_column)) .data[[statistic_column]] else NA_real_,
      p_value = if (!is.null(p_value_column)) .data[[p_value_column]] else NA_real_
    )
}

build_behavioural_model_summary <- function(model_fit, analysis_data, subject_column, prediction_mode) {
  fixed_effects <- extract_behavioural_fixed_effects(model_fit$model)

  fixed_effects %>%
    mutate(
      model_label = model_fit$model_label,
      formula = paste(deparse(model_fit$formula), collapse = " "),
      singular_fit = isTRUE(model_fit$singular_fit),
      fallback_used = isTRUE(model_fit$fallback_used),
      fallback_reason = model_fit$fallback_reason,
      prediction_mode = prediction_mode,
      n_observations = nrow(analysis_data),
      n_subjects = dplyr::n_distinct(analysis_data[[subject_column]])
    ) %>%
    relocate(
      .data$formula,
      .data$model_label,
      .data$prediction_mode,
      .data$singular_fit,
      .data$fallback_used,
      .data$fallback_reason,
      .data$n_observations,
      .data$n_subjects
    )
}

extract_behavioural_subject_slopes <- function(model_fit, analysis_data, subject_column) {
  random_effects <- lme4::ranef(model_fit$model, condVar = TRUE)
  subject_effects <- random_effects[[subject_column]]
  if (is.null(subject_effects) || !"other_rate" %in% colnames(subject_effects)) {
    stop(
      sprintf("Subject-level random slope `other_rate` not available for grouping column `%s`.", subject_column),
      call. = FALSE
    )
  }

  posterior_variance <- attr(subject_effects, "postVar")
  if (is.null(posterior_variance) || length(dim(posterior_variance)) != 3L) {
    stop("Unable to extract posterior variance for subject-level random effects.", call. = FALSE)
  }

  slope_index <- match("other_rate", colnames(subject_effects))
  if (is.na(slope_index)) {
    stop("Unable to locate `other_rate` slope index in subject random effects.", call. = FALSE)
  }

  fixed_effects <- lme4::fixef(model_fit$model)
  fixed_slope <- unname(fixed_effects[["other_rate"]])
  if (is.null(fixed_slope) || length(fixed_slope) == 0L || is.na(fixed_slope)) {
    stop("Unable to extract fixed-effect slope for `other_rate`.", call. = FALSE)
  }

  subject_ids <- rownames(subject_effects)
  random_slopes <- as.numeric(subject_effects[, "other_rate"])
  random_ses <- vapply(
    seq_along(subject_ids),
    function(index) {
      variance_value <- posterior_variance[slope_index, slope_index, index]
      sqrt(as.numeric(variance_value))
    },
    numeric(1)
  )

  dyad_lookup <- analysis_data %>%
    distinct(across(all_of(c(subject_column, "dyad_id")))) %>%
    mutate(across(all_of(subject_column), as.character))

  tibble::tibble(
    subject_id = as.character(subject_ids),
    slope = fixed_slope + random_slopes,
    se = random_ses,
    fixed_slope = fixed_slope,
    random_slope = random_slopes
  ) %>%
    left_join(
      dyad_lookup,
      by = setNames(subject_column, "subject_id")
    ) %>%
    relocate(.data$subject_id, .data$dyad_id, .data$slope, .data$se)
}

format_behavioural_model_summary_text <- function(model_fit, analysis_data, subject_column, prediction_mode) {
  fixed_effects <- extract_behavioural_fixed_effects(model_fit$model)
  fixed_effect_lines <- purrr::pmap_chr(
    fixed_effects,
    function(term, estimate, std_error, statistic, p_value) {
      sprintf(
        "%s: estimate=%.6f, SE=%.6f, statistic=%.6f, p=%.6f",
        term,
        estimate,
        std_error,
        statistic,
        p_value
      )
    }
  )

  paste(
    c(
      sprintf("formula: %s", paste(deparse(model_fit$formula), collapse = " ")),
      sprintf("model_label: %s", model_fit$model_label),
      sprintf("prediction_mode: %s", prediction_mode),
      sprintf("singular_fit: %s", isTRUE(model_fit$singular_fit)),
      sprintf("fallback_used: %s", isTRUE(model_fit$fallback_used)),
      sprintf("fallback_reason: %s", model_fit$fallback_reason),
      sprintf("n_observations: %d", nrow(analysis_data)),
      sprintf("n_subjects: %d", dplyr::n_distinct(analysis_data[[subject_column]])),
      "fixed_effects:",
      fixed_effect_lines
    ),
    collapse = "\n"
  )
}

compute_behavioural_residuals <- function(
  data,
  prediction_mode = c("in_sample", "leave_one_subject_out")
) {
  prediction_mode <- match.arg(prediction_mode, choices = BEHAVIOURAL_PREDICTION_MODES)
  validate_behavioural_residual_inputs(data)
  if (prediction_mode != "in_sample") {
    stop(
      "The gating residual analysis requires in-sample fitted values so predictions include subject-level random effects.",
      call. = FALSE
    )
  }

  resolved_columns <- get_behavioural_model_columns(data)
  subject_column <- resolved_columns$subject_column
  model_columns <- c(BEHAVIOURAL_REQUIRED_COLUMNS, subject_column)
  analysis_data <- data %>%
    mutate(.source_row_id = dplyr::row_number()) %>%
    mutate(
      across(all_of(BEHAVIOURAL_REQUIRED_COLUMNS), ~ suppressWarnings(as.numeric(.x))),
      across(all_of(subject_column), as.character)
    ) %>%
    filter(if_all(all_of(model_columns), ~ !is.na(.x))) %>%
    mutate(.row_id = .source_row_id)

  if (nrow(analysis_data) == 0L) {
    stop("No rows available to fit the behavioural residual model after filtering missing values.", call. = FALSE)
  }

  model_fit <- fit_behavioural_residual_model(
    data = analysis_data,
    subject_column = subject_column
  )

  analysis_data$predicted_self_rate <- as.numeric(stats::fitted(model_fit$model))

  assert_unique_key(analysis_data, ".row_id", "behavioural analysis data")

  analysis_data <- analysis_data %>%
    mutate(
      residual_model_label = model_fit$model_label,
      residual_model_formula = paste(deparse(model_fit$formula), collapse = " "),
      residual_signed = self_rate - predicted_self_rate,
      residual_abs = abs(residual_signed),
      residual_prediction_mode = prediction_mode
    )

  join_payload <- analysis_data %>%
    select(
      .row_id,
      predicted_self_rate,
      residual_model_label,
      residual_model_formula,
      residual_signed,
      residual_abs,
      residual_prediction_mode
    )
  assert_unique_key(join_payload, ".row_id", "behavioural residual join payload")

  augmented_data <- data %>%
    mutate(.source_row_id = dplyr::row_number()) %>%
    left_join(join_payload, by = c(".source_row_id" = ".row_id"))

  if (nrow(augmented_data) != nrow(data)) {
    stop("Behavioural residual merge changed the number of rows.", call. = FALSE)
  }

  augmented_data <- augmented_data %>%
    select(-.source_row_id)

  list(
    augmented_data = augmented_data,
    analysis_data = analysis_data %>% select(-.row_id, -.source_row_id),
    model_fit = model_fit,
    subject_slopes = extract_behavioural_subject_slopes(
      model_fit = model_fit,
      analysis_data = analysis_data,
      subject_column = subject_column
    ),
    summary_table = build_behavioural_model_summary(
      model_fit = model_fit,
      analysis_data = analysis_data,
      subject_column = subject_column,
      prediction_mode = prediction_mode
    ),
    summary_text = format_behavioural_model_summary_text(
      model_fit = model_fit,
      analysis_data = analysis_data,
      subject_column = subject_column,
      prediction_mode = prediction_mode
    ),
    subject_column = subject_column,
    trial_id_column = resolved_columns$trial_id_column
  )
}

# ==============================================================================
# Data loading and preprocessing
# ==============================================================================

load_events_data <- function(input_path) {
  readr::read_csv(input_path, show_col_types = FALSE)
}

rename_legacy_event_columns <- function(events_data) {
  events_data
}

get_required_event_columns <- function() {
  c(
    "dyad_id",
    "subject_id",
    "run",
    "event_id",
    "self_rate",
    "other_rate",
    "latency",
    "time_within_run"
  )
}

preprocess_events_data <- function(raw_events) {
  raw_events <- rename_legacy_event_columns(raw_events)
  stop_if_missing_columns(raw_events, get_required_event_columns())

  raw_events %>%
    mutate(
      dyad_id = as.character(dyad_id),
      subject_id = as.character(subject_id),
      event_id = as.character(event_id),
      run = as.numeric(run),
      self_rate = as.numeric(self_rate),
      other_rate = as.numeric(other_rate),
      latency = as.numeric(latency),
      time_within_run = as.numeric(time_within_run)
    ) %>%
    arrange(dyad_id, run, t_onset, event_id) %>%
    group_by(subject_id, run) %>%
    mutate(
      self_rate_lag1 = dplyr::lag(self_rate, order_by = t_onset),
      row_in_subject_run = dplyr::row_number()
    ) %>%
    ungroup() %>%
    group_by(subject_id) %>%
    mutate(
      subject_mean_rate = mean(self_rate, na.rm = TRUE),
      self_rate_baseline_median = if (all(is.na(self_rate))) NA_real_ else stats::median(self_rate, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    group_by(dyad_id) %>%
    mutate(
      other_rate_baseline_median = purrr::map_dbl(subject_id, function(current_subject_id) {
        partner_baseline_values <- unique(self_rate_baseline_median[subject_id != current_subject_id])
        partner_baseline_values <- partner_baseline_values[!is.na(partner_baseline_values)]

        if (length(partner_baseline_values) == 1L) {
          return(partner_baseline_values[[1]])
        }

        NA_real_
      })
    ) %>%
    ungroup() %>%
    group_by(dyad_id) %>%
    mutate(
      n_subjects_in_dyad = dplyr::n_distinct(subject_id)
    ) %>%
    ungroup() %>%
    mutate(
      median_diff_log = dplyr::if_else(
        !is.na(self_rate_baseline_median) &
          !is.na(other_rate_baseline_median) &
          self_rate_baseline_median > 0 &
          other_rate_baseline_median > 0,
        abs(log(self_rate_baseline_median / other_rate_baseline_median)),
        NA_real_
      ),
      valid_dyad_for_partner_analysis = n_subjects_in_dyad >= 2,
      has_partner_observation = !is.na(other_rate),
      has_latency = !is.na(latency),
      run_z = safe_zscore(run),
      time_within_run_z = safe_zscore(time_within_run),
      other_rate_z = safe_zscore(other_rate),
      median_diff_log_z = safe_zscore(median_diff_log),
      self_rate_z = safe_zscore(self_rate),
      latency_z = safe_zscore(latency),
      self_rate_lag1_z = safe_zscore(self_rate_lag1)
    )
}

make_partner_events <- function(cleaned_events) {
  cleaned_events %>%
    filter(valid_dyad_for_partner_analysis, has_partner_observation)
}

# ==============================================================================
# Null-generation helpers
# ==============================================================================

summarise_null_results <- function(partner_events) {
  tibble(
    dyad_id = character(),
    run = numeric(),
    metric = character(),
    observed_value = numeric(),
    null_mean = numeric(),
    null_sd = numeric(),
    empirical_p = numeric()
  )
}

# ==============================================================================
# Output helpers
# ==============================================================================

build_preprocess_summary <- function(raw_events, cleaned_events, partner_events, input_path = NA_character_, cleaned_path = NA_character_, partner_path = NA_character_, null_path = NA_character_, n_null_iterations = NA_integer_, seed = NA_integer_) {
  list(
    input_path = input_path,
    cleaned_path = cleaned_path,
    partner_path = partner_path,
    null_path = null_path,
    n_rows_raw = nrow(raw_events),
    n_rows_cleaned = nrow(cleaned_events),
    n_rows_partner = nrow(partner_events),
    n_dyads_total = dplyr::n_distinct(cleaned_events$dyad_id),
    n_dyads_partner_valid = dplyr::n_distinct(partner_events$dyad_id),
    n_subjects_total = dplyr::n_distinct(cleaned_events$subject_id),
    n_null_iterations = n_null_iterations,
    seed = seed,
    columns_cleaned = names(cleaned_events),
    notes = c(
      "partner_events includes only rows with non-missing other_rate from dyads with at least two subjects",
      "null_summary is empty because similarity-based null analyses were removed"
    )
  )
}

write_preprocess_outputs <- function(cleaned_events, partner_events, null_output = NULL, cleaned_path = NULL, partner_path = NULL, null_path = NULL, summary_payload = NULL, summary_path = NULL) {
  if (!is.null(cleaned_path)) {
    dir.create(dirname(cleaned_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(cleaned_events, cleaned_path)
  }

  if (!is.null(partner_path)) {
    dir.create(dirname(partner_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(partner_events, partner_path)
  }

  if (!is.null(null_output) && !is.null(null_path)) {
    dir.create(dirname(null_path), recursive = TRUE, showWarnings = FALSE)
    readr::write_csv(null_output, null_path)
  }

  if (!is.null(summary_payload) && !is.null(summary_path)) {
    dir.create(dirname(summary_path), recursive = TRUE, showWarnings = FALSE)
    jsonlite::write_json(summary_payload, summary_path, auto_unbox = TRUE, pretty = TRUE)
  }
}

# ==============================================================================
# CLI
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

main <- function() {
  args <- parse_cli_args()

  input_path <- get_arg_value(args, "input", required = TRUE)
  cleaned_path <- get_arg_value(args, "cleaned_output", required = TRUE)
  partner_path <- get_arg_value(args, "partner_output", required = TRUE)
  null_path <- get_arg_value(args, "null_output", required = TRUE)
  summary_path <- get_arg_value(args, "summary_output", required = TRUE)
  n_null <- as.integer(get_arg_value(args, "n_null", default = "1000"))
  seed <- as.integer(get_arg_value(args, "seed", default = "123"))

  raw_events <- load_events_data(input_path)
  cleaned_events <- preprocess_events_data(raw_events)
  partner_events <- make_partner_events(cleaned_events)
  null_summary <- summarise_null_results(partner_events)

  summary_payload <- build_preprocess_summary(
    raw_events = raw_events,
    cleaned_events = cleaned_events,
    partner_events = partner_events,
    input_path = input_path,
    cleaned_path = cleaned_path,
    partner_path = partner_path,
    null_path = null_path,
    n_null_iterations = n_null,
    seed = seed
  )

  write_preprocess_outputs(
    cleaned_events = cleaned_events,
    partner_events = partner_events,
    null_output = null_summary,
    cleaned_path = cleaned_path,
    partner_path = partner_path,
    null_path = null_path,
    summary_payload = summary_payload,
    summary_path = summary_path
  )
}

if (sys.nframe() == 0) {
  main()
}
