"""Filename utilities for safe filesystem operations."""
from __future__ import annotations

import re


def url_to_safe_filename(url: str) -> str:
    """
    Convert an URL into a safe, cross-platform single filesystem filename.

    Parameters
    ----------
    url : str
        The input URL string.

    Returns
    -------
    str
        A sanitized, valid single-file filename.

    Raises
    ------
    ValueError
        If *url* is empty or whitespace-only.
    """
    if not url or not url.strip():
        raise ValueError("'url' cannot be empty")

    # Replace all Windows/POSIX invalid characters and control chars with underscore
    safe = re.sub(r'[<>:"/\\|?*]', '_', url)
    safe = re.sub(r'[\x00-\x1f]', '_', safe)
    # Handle Windows reserved names (CON, PRN, etc.)
    safe = re.sub(
        r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-3])$', r'_\1', safe, flags=re.IGNORECASE
    )
    return safe
