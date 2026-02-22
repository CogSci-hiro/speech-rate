from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed QC thresholds are explicit constants (not outcome-dependent).
EXPONENT_VALID_RANGE = (0.0, 5.0)
OFFSET_ABS_MAX = 30.0
R_SQUARED_MIN = 0.80
ERROR_MAX = 0.30


def apply_fit_qc(row: pd.Series) -> pd.Series:
    """Annotate a channel-level FOOOF row with deterministic QC flags."""
    out = row.copy()

    fit_success = bool(out.get("fit_success", False))
    offset = float(out.get("offset", np.nan))
    exponent = float(out.get("exponent", np.nan))
    r_squared = float(out.get("r_squared", np.nan))
    error = float(out.get("error", np.nan))

    out["fit_success"] = fit_success
    out["nonfinite_aperiodic"] = bool((not np.isfinite(offset)) or (not np.isfinite(exponent)))
    out["nonfinite_diagnostics"] = bool((not np.isfinite(r_squared)) or (not np.isfinite(error)))
    out["extreme_exponent"] = bool(
        np.isfinite(exponent) and (exponent < EXPONENT_VALID_RANGE[0] or exponent > EXPONENT_VALID_RANGE[1])
    )
    out["extreme_offset"] = bool(np.isfinite(offset) and (abs(offset) > OFFSET_ABS_MAX))
    out["low_r_squared"] = bool(np.isfinite(r_squared) and (r_squared < R_SQUARED_MIN))
    out["high_error"] = bool(np.isfinite(error) and (error > ERROR_MAX))
    out["qc_any_flag"] = bool(
        (not fit_success)
        or out["nonfinite_aperiodic"]
        or out["nonfinite_diagnostics"]
        or out["extreme_exponent"]
        or out["extreme_offset"]
        or out["low_r_squared"]
        or out["high_error"]
    )
    return out
