suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(jsonlite)
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
# Data loading and preprocessing
# ==============================================================================

load_events_data <- function(input_path) {
  readr::read_csv(input_path, show_col_types = FALSE)
}

rename_alignment_to_proximity <- function(events_data) {
  if ("alignment" %in% names(events_data) && !"proximity" %in% names(events_data)) {
    events_data <- events_data %>% dplyr::rename(proximity = alignment)
  }
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
    "proximity",
    "convergence",
    "time_within_run"
  )
}

preprocess_events_data <- function(raw_events) {
  raw_events <- rename_alignment_to_proximity(raw_events)
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
      proximity = as.numeric(proximity),
      convergence = as.numeric(convergence),
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
      subject_mean_rate = mean(self_rate, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    group_by(dyad_id) %>%
    mutate(
      n_subjects_in_dyad = dplyr::n_distinct(subject_id)
    ) %>%
    ungroup() %>%
    mutate(
      valid_dyad_for_partner_analysis = n_subjects_in_dyad >= 2,
      has_partner_observation = !is.na(other_rate),
      has_latency = !is.na(latency),
      has_proximity = !is.na(proximity),
      has_convergence = !is.na(convergence),
      run_z = safe_zscore(run),
      time_within_run_z = safe_zscore(time_within_run),
      other_rate_z = safe_zscore(other_rate),
      self_rate_z = safe_zscore(self_rate),
      latency_z = safe_zscore(latency),
      proximity_z = safe_zscore(proximity),
      convergence_z = safe_zscore(convergence),
      self_rate_lag1_z = safe_zscore(self_rate_lag1)
    )
}

make_partner_events <- function(cleaned_events) {
  partner_events <- cleaned_events %>%
    filter(valid_dyad_for_partner_analysis, has_partner_observation) %>%
    group_by(dyad_id, subject_id, run) %>%
    mutate(
      partner_mean_rate = mean(other_rate, na.rm = TRUE)
    ) %>%
    ungroup()

  partner_events %>%
    mutate(
      baseline_proximity = -abs(subject_mean_rate - partner_mean_rate),
      convergence_from_means = proximity - baseline_proximity
    )
}

# ==============================================================================
# Null-generation helpers
# ==============================================================================

make_circular_shift_null <- function(group_data, n_null, metric_names) {
  number_of_rows <- nrow(group_data)

  if (number_of_rows < 2L || all(is.na(group_data$other_rate))) {
    return(tibble())
  }

  available_shifts <- seq_len(number_of_rows - 1L)
  sampled_shifts <- sample(available_shifts, size = n_null, replace = TRUE)

  purrr::map_dfr(seq_along(sampled_shifts), function(iteration_index) {
    shift_value <- sampled_shifts[[iteration_index]]
    surrogate_data <- group_data
    surrogate_data$other_rate_null <- c(
      tail(group_data$other_rate, shift_value),
      head(group_data$other_rate, number_of_rows - shift_value)
    )

    if ("proximity" %in% metric_names) {
      surrogate_data$proximity_null <- -abs(surrogate_data$self_rate - surrogate_data$other_rate_null)
    }

    if (
      "convergence" %in% metric_names &&
      "subject_mean_rate" %in% names(group_data) &&
      "partner_mean_rate" %in% names(group_data)
    ) {
      surrogate_data$convergence_null <-
        -abs(surrogate_data$self_rate - surrogate_data$other_rate_null) -
        (-abs(surrogate_data$subject_mean_rate - surrogate_data$partner_mean_rate))
    }

    metric_rows <- list()

    if ("proximity" %in% metric_names) {
      metric_rows[[length(metric_rows) + 1L]] <- tibble(
        metric = "proximity",
        null_value = mean(surrogate_data$proximity_null, na.rm = TRUE)
      )
    }

    if ("convergence" %in% metric_names && "convergence_null" %in% names(surrogate_data)) {
      metric_rows[[length(metric_rows) + 1L]] <- tibble(
        metric = "convergence",
        null_value = mean(surrogate_data$convergence_null, na.rm = TRUE)
      )
    }

    bind_rows(metric_rows) %>%
      mutate(
        null_iteration = iteration_index,
        shift = shift_value
      )
  })
}

compute_observed_metrics <- function(partner_events) {
  partner_events %>%
    group_by(dyad_id, run) %>%
    summarise(
      observed_proximity = mean(proximity, na.rm = TRUE),
      observed_convergence = mean(convergence_from_means, na.rm = TRUE),
      n_events = dplyr::n(),
      .groups = "drop"
    )
}

compute_null_distribution <- function(partner_events, n_null, seed = NULL, metric_names = c("proximity", "convergence")) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  partner_events %>%
    group_by(dyad_id, run) %>%
    group_modify(~ make_circular_shift_null(
      group_data = .x,
      n_null = n_null,
      metric_names = metric_names
    )) %>%
    ungroup()
}

summarise_null_results <- function(observed_metrics, null_distribution) {
  null_summary <- null_distribution %>%
    group_by(dyad_id, run, metric) %>%
    summarise(
      null_mean = mean(null_value, na.rm = TRUE),
      null_sd = stats::sd(null_value, na.rm = TRUE),
      .groups = "drop"
    )

  observed_long <- observed_metrics %>%
    pivot_longer(
      cols = c(observed_proximity, observed_convergence),
      names_to = "metric",
      values_to = "observed_value"
    ) %>%
    mutate(
      metric = dplyr::recode(
        metric,
        observed_proximity = "proximity",
        observed_convergence = "convergence"
      )
    )

  empirical_tests <- observed_long %>%
    group_by(dyad_id, run, metric) %>%
    group_modify(~ {
      current_null <- null_distribution %>%
        filter(
          dyad_id == .y$dyad_id,
          run == .y$run,
          metric == .y$metric
        )

      current_observed <- .x$observed_value[[1]]

      tibble(
        empirical_p = empirical_p_value(
          observed_value = current_observed,
          null_values = current_null$null_value,
          direction = "greater"
        )
      )
    }) %>%
    ungroup()

  observed_long %>%
    left_join(null_summary, by = c("dyad_id", "run", "metric")) %>%
    left_join(empirical_tests, by = c("dyad_id", "run", "metric")) %>%
    arrange(metric, dyad_id, run)
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
      "alignment was renamed to proximity when needed",
      "partner_events includes only rows with non-missing other_rate from dyads with at least two subjects",
      "convergence_from_means is recomputed as proximity - baseline_proximity for transparency in the null analysis",
      "null analysis uses within-dyad, within-run circular shifts of other_rate"
    )
  )
}

write_preprocess_outputs <- function(cleaned_events, partner_events, null_output = NULL, cleaned_path = NULL, partner_path = NULL, null_path = NULL, summary_payload = NULL, summary_path = NULL) {
  if (!is.null(cleaned_path)) {
    readr::write_csv(cleaned_events, cleaned_path)
  }

  if (!is.null(partner_path)) {
    readr::write_csv(partner_events, partner_path)
  }

  if (!is.null(null_output) && !is.null(null_path)) {
    readr::write_csv(null_output, null_path)
  }

  if (!is.null(summary_payload) && !is.null(summary_path)) {
    jsonlite::write_json(summary_payload, summary_path, auto_unbox = TRUE, pretty = TRUE)
  }
}
