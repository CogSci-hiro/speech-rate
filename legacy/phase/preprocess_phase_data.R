# ================================================================
# Speech Rate Convergence – Data Preparation
# ================================================================

library(dplyr)

# ----------------------------------------------------------------
# Merge datasets
# ----------------------------------------------------------------

merge_datasets <- function(events_df, phase_df) {

  # rename phase variables to avoid collision
  phase_df <- phase_df %>%
    rename(
      phase_abs_diff = abs_diff,
      omega = omega_trial
    ) %>%
    # Keep only phase-specific columns so event-level keys like run/subject_id
    # from events_df are not suffixed away during join.
    select(
      any_of(c("event_id", "delta_phi", "delta_t", "residual", "cos_residual", "omega", "phase_abs_diff"))
    ) %>%
    distinct(event_id, .keep_all = TRUE)

  # merge
  df <- events_df %>%
    left_join(phase_df, by = "event_id")

  stopifnot(length(unique(df$event_id)) == nrow(df))

  return(df)
}

# ----------------------------------------------------------------
# Basic cleaning
# ----------------------------------------------------------------

clean_data <- function(df) {

  df <- df %>%
    filter(
      !is.na(self_rate),
      !is.na(other_rate)
    )

  df$run <- factor(df$run)
  df$subject_id <- factor(df$subject_id)

  return(df)
}

# ----------------------------------------------------------------
# Test-specific datasets
# ----------------------------------------------------------------

prepare_cos_test <- function(df) {

  df %>%
    filter(!is.na(cos_residual)) %>%
    select(
      subject_id,
      run,
      abs_diff,
      self_rate,
      other_rate,
      latency,
      duration_s,
      time_within_run,
      cos_residual
    )
}


prepare_omega_test <- function(df) {

  df %>%
    filter(!is.na(omega)) %>%
    select(
      subject_id,
      abs_diff,
      other_rate,
      omega,
      latency,
      duration_s,
      run,
      time_within_run
    )
}


prepare_phase_window_test <- function(df) {

  df <- df %>%
    filter(
      !is.na(self_rate),
      !is.na(other_rate),
      !is.na(latency)
    )

  df$phase_window <- abs(df$latency) <= 0.25
  df$abs_latency  <- abs(df$latency)

  return(df)
}


prepare_latency_continuous_test <- function(df) {

  df <- df %>%
    filter(
      !is.na(self_rate),
      !is.na(other_rate),
      !is.na(latency)
    )

  df$abs_latency <- abs(df$latency)

  return(df)
}

