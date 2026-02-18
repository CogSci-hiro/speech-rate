"""QC summaries for canonical event tables."""

import pandas as pd

QC_COLUMNS = [
    "dyad_id",
    "run",
    "n_events",
    "duration_min",
    "duration_mean",
    "duration_max",
    "syllables_min",
    "syllables_mean",
    "syllables_max",
]


def build_ipu_qc_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Build a dyad/run-level QC summary from canonical events."""
    if events_df.empty:
        return pd.DataFrame(columns=QC_COLUMNS)

    summary_df = (
        events_df.groupby(["dyad_id", "run"], dropna=False)
        .agg(
            n_events=("event_id", "size"),
            duration_min=("duration_s", "min"),
            duration_mean=("duration_s", "mean"),
            duration_max=("duration_s", "max"),
            syllables_min=("syllables", "min"),
            syllables_mean=("syllables", "mean"),
            syllables_max=("syllables", "max"),
        )
        .reset_index()
    )
    return summary_df[QC_COLUMNS]
