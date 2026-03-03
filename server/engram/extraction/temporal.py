"""Temporal hint resolution — converts relative time phrases to datetimes."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# Month name → number
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def resolve_temporal_hint(
    hint: str,
    reference_date: datetime | None = None,
) -> datetime | None:
    """Resolve a temporal hint string to a datetime.

    Handles:
    - ISO 8601 dates (passthrough)
    - "last month", "last week", "last year"
    - "X days/weeks/months ago"
    - "since January", "since March 2024"
    - "recently" → 7 days ago

    Returns None for unparseable hints.
    """
    if not hint or not hint.strip():
        return None

    hint = hint.strip().lower()
    ref = reference_date or datetime.utcnow()

    # ISO 8601 passthrough
    try:
        return datetime.fromisoformat(hint)
    except (ValueError, TypeError):
        pass

    # "last month"
    if hint == "last month":
        year = ref.year
        month = ref.month - 1
        if month < 1:
            month = 12
            year -= 1
        return datetime(year, month, 1)

    # "last week"
    if hint == "last week":
        return ref - timedelta(weeks=1)

    # "last year"
    if hint == "last year":
        return datetime(ref.year - 1, 1, 1)

    # "X days/weeks/months/years ago"
    ago_match = re.match(r"(\d+)\s+(day|week|month|year)s?\s+ago", hint)
    if ago_match:
        n = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == "day":
            return ref - timedelta(days=n)
        elif unit == "week":
            return ref - timedelta(weeks=n)
        elif unit == "month":
            year = ref.year
            month = ref.month - n
            while month < 1:
                month += 12
                year -= 1
            return datetime(year, month, 1)
        elif unit == "year":
            return datetime(ref.year - n, 1, 1)

    # "since <month> <year>" or "since <month>"
    since_match = re.match(r"since\s+(\w+)(?:\s+(\d{4}))?", hint)
    if since_match:
        month_name = since_match.group(1).lower()
        year_str = since_match.group(2)
        month_num = _MONTHS.get(month_name)
        if month_num:
            year = int(year_str) if year_str else ref.year
            # If the month is in the future of the current year, use previous year
            if not year_str and (year == ref.year and month_num > ref.month):
                year -= 1
            return datetime(year, month_num, 1)

    # "recently"
    if hint == "recently":
        return ref - timedelta(days=7)

    return None
