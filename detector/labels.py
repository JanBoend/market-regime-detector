"""Utility for mapping HMM states to human-readable regime labels."""

VALID_LABELS = {"bull", "bear", "sideways"}


def validate_label(label: str) -> str:
    if label not in VALID_LABELS:
        raise ValueError(f"Unknown regime label: {label!r}. Expected one of {VALID_LABELS}")
    return label
