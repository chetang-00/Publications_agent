"""Text normalisation shared by SQL filters and vector payloads."""


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace and drop trailing dots: "Ranganath  R." -> "ranganath r"."""
    return " ".join(text.lower().split()).rstrip(".").strip()
