"""Shared constants and helpers for the intelligence layer (pure pandas, no Streamlit)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MEGA_MW = 500.0  # orders at/above this size are "mega" orders
GIGA_MW = 1000.0  # orders at/above this size are "giga" orders
QUARTER_END_WINDOW_DAYS = 14  # window used for the quarter-end clustering metric

# Labels that are not a single country (kept visible, but excluded from country-level analysis).
MULTI_COUNTRY_LABELS = {"EU", "Eastern Europe", "North America", "United States and Canada", "Unknown"}
# Variants that belong to a single country.
COUNTRY_FIXES = {"United Kingdom - Offshore": "United Kingdom", "Wales": "United Kingdom"}
# Installed-base workbook names -> names used in the order data.
INSTALLED_BASE_ALIASES = {
    "USA": "United States",
    "UK": "United Kingdom",
    "Türkiye": "Turkey",
    "T�rkiye": "Turkey",
    "Luxemburg": "Luxembourg",
    "UAE": "United Arab Emirates",
    "Cura�ao": "Curaçao",
}

OFFSHORE_FAMILIES = {"V164", "V174", "V236"}  # offshore-class platforms (proxy; orders carry no onshore/offshore flag)

RATING_BUCKET_ORDER = ["<2.5 MW", "2.5-3.9 MW", "4.0-5.9 MW", "6.0-9.9 MW", "10+ MW"]
UNKNOWN = "Unknown"


def installed_base_path() -> Path:
    return Path("data") / "vestas_installed_base.json"


def same_day(year: int, ref: pd.Timestamp) -> pd.Timestamp:
    """The ref month/day in another year (Feb-29 falls back to Feb-28)."""
    try:
        return pd.Timestamp(year=year, month=ref.month, day=ref.day)
    except ValueError:
        return pd.Timestamp(year=year, month=ref.month, day=28)


def rating_bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        return UNKNOWN
    v = float(value)
    if v < 2.5:
        return RATING_BUCKET_ORDER[0]
    if v < 4.0:
        return RATING_BUCKET_ORDER[1]
    if v < 6.0:
        return RATING_BUCKET_ORDER[2]
    if v < 10.0:
        return RATING_BUCKET_ORDER[3]
    return RATING_BUCKET_ORDER[4]


def platform_family(name: Any) -> str:
    """'V150-4.5MW' -> 'V150'; generic labels ('3MW', 'Unknown') -> 'Unknown'."""
    if name is None or pd.isna(name):
        return UNKNOWN
    m = re.match(r"^(V\d{2,3})", str(name))
    return m.group(1) if m else UNKNOWN


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the analytics need (does not modify the input)."""
    o = orders.copy()
    o["country_clean"] = o["country"].replace(COUNTRY_FIXES)
    o["multi_country"] = o["country_clean"].isin(MULTI_COUNTRY_LABELS)
    quarter_end = o["order_date"] + pd.offsets.QuarterEnd(0)
    o["days_to_quarter_end"] = (quarter_end - o["order_date"]).dt.days
    o["in_quarter_end_window"] = o["days_to_quarter_end"] < QUARTER_END_WINDOW_DAYS
    o["order_quarter_label"] = o["order_year"].astype(str) + "-Q" + o["order_quarter"].astype(str)
    o["is_mega"] = o["size_mw"] >= MEGA_MW
    o["is_giga"] = o["size_mw"] >= GIGA_MW
    return o


def prepare_platforms(platforms: pd.DataFrame) -> pd.DataFrame:
    p = platforms.copy()
    p["family"] = p["platform"].map(platform_family)
    p["offshore_class"] = p["family"].isin(OFFSHORE_FAMILIES)
    p["rating_bucket"] = p["mw_rating"].map(rating_bucket)
    return p


def as_of_date(orders: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(orders["order_date"].max()).normalize()


def _norm_metric(name: Any) -> str:
    text = str(name).replace("–", "-").replace("—", "-").replace("�", "-")
    return re.sub(r"\s+", " ", text.strip().lower())


def eco_wide(economy: pd.DataFrame) -> pd.DataFrame:
    """Economy as year x metric with normalised (lower-case, ASCII dash) metric names."""
    if economy is None or economy.empty:
        return pd.DataFrame()
    e = economy.copy()
    e["key"] = e["metric"].map(_norm_metric)
    wide = e.pivot_table(index="year", columns="key", values="value", aggfunc="first").sort_index()
    return wide


def eco_series(wide: pd.DataFrame, *candidates: str) -> pd.Series:
    """First matching metric as a Series indexed by year (empty Series when none match)."""
    for cand in candidates:
        key = _norm_metric(cand)
        if key in wide.columns:
            return wide[key].dropna()
    return pd.Series(dtype=float)


def load_installed_base(path: Path | None = None) -> dict[str, Any] | None:
    path = path or installed_base_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def safe_div(a: float, b: float) -> float:
    if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def pct_change(new: float, old: float) -> float:
    """(new/old - 1) as a fraction; NaN when the base is missing or zero."""
    if old is None or pd.isna(old) or old == 0 or new is None or pd.isna(new):
        return float("nan")
    return float(new) / float(old) - 1.0


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))
