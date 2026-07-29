import re

_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    normalized = value.strip().casefold()
    return _SPACE_RE.sub(" ", normalized)
