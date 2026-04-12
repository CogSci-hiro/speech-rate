suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# ==============================================================================
# CLI helpers
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

# ==============================================================================
# Output helpers
# ==============================================================================

make_residual_output_table <- function(augmented_data, trial_id_column = NULL) {
  key_columns <- c("dyad_id", "subject_id", "run", "task", "event_id")
  present_key_columns <- key_columns[key_columns %in% names(augmented_data)]

  selected_columns <- c(
    present_key_columns,
    "self_rate",
    "other_rate",
    "openness",
    "extraversion",
    "predicted_self_rate",
    "residual_model_label",
    "residual_model_formula",
    "residual_signed",
    "residual_abs",
    "latency",
    "duration_s",
    "time_within_run",
    "run"
  )
  selected_columns <- unique(selected_columns[selected_columns %in% names(augmented_data)])

  residual_table <- augmented_data %>%
    select(all_of(selected_columns))

  if (!is.null(trial_id_column) && trial_id_column %in% names(augmented_data) && !"trial_id" %in% names(residual_table)) {
    residual_table <- residual_table %>%
      mutate(trial_id = augmented_data[[trial_id_column]]) %>%
      relocate(.data$trial_id, .after = dplyr::last_col())
  }

  residual_table
}

write_csv_output <- function(data, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(data, output_path)
}

write_text_file <- function(text, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(text, con = output_path, useBytes = TRUE)
}

# ==============================================================================
# Main
# ==============================================================================

main <- function() {
  args <- parse_cli_args()

  input_path <- get_arg_value(args, "input", required = TRUE)
  preprocess_script_path <- get_arg_value(args, "preprocess_script", required = TRUE)
  output_path <- get_arg_value(args, "output", required = TRUE)
  subject_slopes_output_path <- get_arg_value(args, "subject_slopes_output", required = TRUE)
  model_summary_csv_path <- get_arg_value(args, "model_summary_csv", required = TRUE)
  model_summary_txt_path <- get_arg_value(args, "model_summary_txt", required = TRUE)
  prediction_mode <- get_arg_value(args, "prediction_mode", default = "in_sample")

  source(preprocess_script_path)

  raw_events <- load_events_data(input_path)
  cleaned_events <- preprocess_events_data(raw_events)
  residual_results <- compute_behavioural_residuals(
    data = cleaned_events,
    prediction_mode = prediction_mode
  )

  residual_table <- make_residual_output_table(
    augmented_data = residual_results$augmented_data,
    trial_id_column = residual_results$trial_id_column
  )

  write_csv_output(residual_table, output_path)
  write_csv_output(residual_results$subject_slopes, subject_slopes_output_path)
  write_csv_output(residual_results$summary_table, model_summary_csv_path)
  write_text_file(residual_results$summary_text, model_summary_txt_path)
}

if (sys.nframe() == 0) {
  main()
}
