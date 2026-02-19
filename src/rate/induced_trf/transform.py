import numpy as np


def apply_log_transform(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values

    eps = 1e-12
    safe = values + eps

    if method == "log10":
        return np.log10(safe)
    if method == "ln":
        return np.log(safe)

    raise ValueError(f"Unknown log transform: {method}")
