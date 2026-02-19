from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Vestas Sales Intelligence Dashboard"

DEFAULT_ECON_METRICS = [
    "revenue (mEUR)",
    "EBIT",
    "Order intake (bnEUR)",
    "Order inkake (MW)",
    "Order backlog – wind turbines (bnEUR)",
    "Order backlog – service (bnEUR)",
    "deliveries [MW]",
]

PLATFORM_SLOT_SPECS = [
    {
        "slot": 1,
        "qty": ["#", "No_WTG_1"],
        "platform": ["platform"],
        "rotor": ["Rotor", "Rotor_1"],
        "mw": ["MW", "MW_1"],
    },
    {
        "slot": 2,
        "qty": ["#.1", "No_WTG_2"],
        "platform": ["Multi platform"],
        "rotor": ["Rotor.1", "Rotor_2"],
        "mw": ["MW.1", "MW_2"],
    },
    {
        "slot": 3,
        "qty": ["#.2", "No_WTG_3"],
        "platform": ["Multi platform.1"],
        "rotor": ["Rotor.2", "Rotor"],
        "mw": ["MW.2", "MW"],
    },
    {
        "slot": 4,
        "qty": ["#.3", "No_WTG_4"],
        "platform": ["Multi platform.2"],
        "rotor": ["Rotor.3", "Rotor.1"],
        "mw": ["MW.3", "MW.1"],
    },
]

COUNTRY_MAP = {
    "usa": "United States",
    "united states": "United States",
    "united states and canada": "United States and Canada",
    "uk": "United Kingdom",
    "uk - offshore": "United Kingdom - Offshore",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "irland": "Ireland",
    "spanien": "Spain",
    "mongoliet": "Mongolia",
    "den dominikanske republik": "Dominican Republic",
    "dominican republic": "Dominican Republic",
    "?": "Unknown",
    "no info": "Unknown",
    "notmentioned": "Unknown",
    "un disclosed": "Undisclosed",
}

GEO_COUNTRY_OVERRIDES = {
    "United States and Canada": "United States",
    "United Kingdom - Offshore": "United Kingdom",
    "Wales": "United Kingdom",
    "Unknown": None,
    "EU": None,
    "NotMentioned": None,
    "Undisclosed": None,
}

COUNTRY_ISO3_MAP = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Denmark": "DNK",
    "Dominican Republic": "DOM",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Mexico": "MEX",
    "Mongolia": "MNG",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Romania": "ROU",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sweden": "SWE",
    "Taiwan": "TWN",
    "Thailand": "THA",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Vietnam": "VNM",
}

SUMMARY_LABELS = {
    "# mw",
    "# of sites",
    "# aom5000 energy based yield",
    "# aom4000 time based yield",
    "% aom5000 of confirmed",
    "% of confirmed energy yield sc",
    "services counters",
    "market share",
    "country",
    "type of contract",
    "potential turbines",
    "total # blades",
    "total # turbines incl undisclosed",
    "w. average rotor dia / #",
    "w. average rotor dia / mw",
    "w. average service contract",
    "weighted average",
    "counters",
    "turbines",
    "sales",
    "1 qtr",
    "2 qtr",
    "3 qtr",
    "4 qtr",
    "0-5",
    "5-10",
    "10-15",
    "15-20",
    "20+",
}

SVC_UNKNOWN_VALUES = {"?", "nan", "none", "no info", "not mentioned", "unknown", "0"}
CUSTOMER_UNKNOWN_VALUES = {"?", "nan", "none", "no info", "not mentioned", "unknown"}


def apply_page_style(dark_mode: bool) -> None:
    if dark_mode:
        text_color = "#e9eef5"
        bg_main = (
            "radial-gradient(1100px 580px at 100% -10%, rgba(34, 88, 156, 0.35) 0%, rgba(34, 88, 156, 0) 55%),"
            "radial-gradient(900px 500px at -10% 15%, rgba(14, 128, 110, 0.28) 0%, rgba(14, 128, 110, 0) 48%),"
            "linear-gradient(180deg, #0d1118 0%, #0a1016 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, #111926 0%, #0d1520 100%)"
        header_bg = "rgba(10, 16, 24, 0.96)"
        header_border = "rgba(92, 140, 190, 0.25)"
        card_bg = "rgba(19, 29, 42, 0.82)"
        card_border = "rgba(92, 140, 190, 0.25)"
        frame_border = "rgba(92, 140, 190, 0.2)"
    else:
        text_color = "#102132"
        bg_main = (
            "radial-gradient(1200px 600px at 95% -5%, #dceef7 0%, rgba(220,238,247,0) 45%),"
            "radial-gradient(1000px 500px at -10% 20%, #f8e9d8 0%, rgba(248,233,216,0) 40%),"
            "linear-gradient(180deg, #f7fafc 0%, #ecf2f7 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, #f3f7fb 0%, #e9f0f7 100%)"
        header_bg = "rgba(247, 250, 252, 0.96)"
        header_border = "rgba(33, 63, 92, 0.12)"
        card_bg = "rgba(255,255,255,0.85)"
        card_border = "rgba(33, 63, 92, 0.12)"
        frame_border = "rgba(33, 63, 92, 0.12)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

html, body, .stApp {{
  font-family: 'Manrope', sans-serif;
  color: {text_color};
}}

.stApp {{
  background: {bg_main};
}}

section[data-testid="stSidebar"] {{
  background: {bg_sidebar};
  border-right: 1px solid {frame_border};
}}

section[data-testid="stSidebar"] * {{
  color: {text_color};
}}

section[data-testid="stSidebar"] > div:first-child {{
  height: 100vh;
}}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 100%;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
  background: {header_bg} !important;
  border-bottom: 1px solid {header_border} !important;
}}

[data-testid="stDecoration"] {{
  background: transparent !important;
}}

/* Restore Streamlit/Material icon rendering (prevents ligature text such as keyboard_double_arrow_right) */
[class*="material-icons"],
[class*="material-symbols"],
[data-testid="stHeader"] button span,
[data-testid="stToolbar"] button span,
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarNav"] button span,
button[aria-label*="sidebar"] span,
button[aria-label*="Sidebar"] span,
button[title*="sidebar"] span,
button[title*="Sidebar"] span {{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-feature-settings: "liga" 1, "calt" 1;
  -webkit-font-feature-settings: "liga" 1, "calt" 1;
  font-weight: normal;
  font-style: normal;
  letter-spacing: normal;
  text-transform: none;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
  background: {card_bg};
  border: 1px solid {card_border};
  color: {text_color};
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

div[data-testid="stMetric"] {{
  background: {card_bg};
  border: 1px solid {card_border};
  border-radius: 12px;
  padding: 0.4rem 0.8rem;
}}

div[data-testid="stMetricDelta"] {{
  font-size: 0.72rem !important;
  line-height: 1.05 !important;
}}

div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] {{
  border: 1px solid {frame_border};
  border-radius: 10px;
}}

.sidebar-footer {{
  padding: 0.35rem 0.45rem 0.05rem 0.45rem;
  border-top: 1px solid {frame_border};
  font-size: 0.62rem;
  line-height: 1.2;
  opacity: 0.8;
}}

.sidebar-footer-wrap {{
  margin-top: auto;
  margin-left: 0.15rem;
  margin-right: 0.15rem;
  margin-bottom: 0.2rem;
  background: {card_bg};
  border: 1px solid {card_border};
  border-radius: 8px;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template() -> str:
    return "plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white"


def normalize_col_name(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_metric_name(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    # Normalize different dash encodings to plain ASCII dash
    text = text.replace("–", "-").replace("—", "-").replace("â€“", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).replace("\xa0", " ").replace("\u202f", " ").strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def to_float(value: Any) -> float:
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if not math.isnan(float(value)) else np.nan

    text = clean_text(value)
    if text is None:
        return np.nan

    compact = text.replace(" ", "").replace("%", "").replace(",", ".")
    if compact in {"?", "-", "--"}:
        return np.nan
    try:
        return float(compact)
    except ValueError:
        return np.nan


def parse_mixed_date(value: Any) -> pd.Timestamp | pd.NaT:
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    text = clean_text(value)
    if text is None:
        return pd.NaT

    compact = text.replace(" ", "")

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", compact):
        num = float(compact)
        if 20_000 <= num <= 70_000:
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(num, unit="D")
            if 1990 <= ts.year <= 2035:
                return ts.normalize()
        return pd.NaT

    fix_year = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{5})", compact)
    if fix_year:
        day, month, year = fix_year.groups()
        compact = f"{day}.{month}.{year[-4:]}"

    ts = pd.to_datetime(compact, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return pd.NaT
    if ts.year < 1990 or ts.year > 2035:
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def normalize_country(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if norm in COUNTRY_MAP:
        return COUNTRY_MAP[norm]
    if norm in SUMMARY_LABELS:
        return "Unknown"
    if norm.startswith("#"):
        return "Unknown"
    return text


def map_country_for_geo(country: Any) -> str | None:
    text = clean_text(country)
    if text is None:
        return None
    text = GEO_COUNTRY_OVERRIDES.get(text, text)
    if text is None:
        return None
    return COUNTRY_ISO3_MAP.get(text)


def render_sidebar_footer() -> None:
    st.sidebar.markdown(
        """
<div class="sidebar-footer-wrap">
  <div class="sidebar-footer">
  Unofficial dashboard - not affiliated with or endorsed by Vestas. Data compiled manually from public Vestas reports/announcements; verify against original sources. Provided "as is" without warranty.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def normalize_service_scheme(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = re.sub(r"\s+", " ", text).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", norm)

    if norm in SVC_UNKNOWN_VALUES:
        return "Unknown"
    if "5000" in compact:
        return "AOM5000"
    if "4000" in compact:
        return "AOM4000"
    return "Unknown"


def normalize_customer(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = text.strip().lower()
    if norm in CUSTOMER_UNKNOWN_VALUES:
        return "Unknown"
    return text


def normalize_platform(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    text = text.upper().replace(",", ".")
    text = re.sub(r"\s+", "", text)
    if text in {"?", "NAN", "NONE"}:
        return "Unknown"
    return text


def parse_platform_specs(platform: str) -> tuple[float, float]:
    if platform == "Unknown":
        return np.nan, np.nan

    p = platform.upper().replace(" ", "")
    full = re.search(r"V(\d{2,3})-([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if full:
        rotor = float(full.group(1))
        power = float(full.group(2))
        if full.group(3) == "KW":
            power = power / 1000.0
        return rotor, power

    mw_only = re.search(r"([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if mw_only:
        power = float(mw_only.group(1))
        if mw_only.group(2) == "KW":
            power = power / 1000.0
        return np.nan, power

    return np.nan, np.nan


def first_text(row: pd.Series, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in row.index:
            val = clean_text(row[col])
            if val is not None:
                return val
    return None


def first_float(row: pd.Series, candidates: list[str]) -> float:
    for col in candidates:
        if col in row.index:
            val = to_float(row[col])
            if not math.isnan(val):
                return val
    return np.nan


def quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)


def parse_quarter_label(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    token = re.sub(r"[^a-z0-9]", "", text.lower())
    match = re.fullmatch(r"q([1-4])", token)
    if match:
        return int(match.group(1))
    return None


def extract_unannounced_quarters(raw: pd.DataFrame, sheet_name: str, sheet_year: int) -> list[dict[str, Any]]:
    cols_norm = {normalize_col_name(c): c for c in raw.columns}
    size_col = cols_norm.get("size [mw]")
    if size_col is None and len(raw.columns) > 3:
        size_col = raw.columns[3]

    label_cols: list[str] = []
    for candidate in ["date", "country"]:
        col = cols_norm.get(candidate)
        if col is not None:
            label_cols.append(col)
    for idx in [0, 1, 2]:
        if idx < len(raw.columns):
            col = raw.columns[idx]
            if col not in label_cols:
                label_cols.append(col)

    rows: list[dict[str, Any]] = []
    for i in range(len(raw)):
        row_values = raw.iloc[i].tolist()
        texts = [clean_text(v) for v in row_values]
        normalized = [re.sub(r"\s+", " ", t).strip().lower() for t in texts if t is not None]
        if not any(t.startswith("unannounced") or t.startswith("unannounce") for t in normalized):
            continue

        for j in range(i + 1, min(i + 10, len(raw))):
            row_j = raw.iloc[j]
            quarter = None
            for col in label_cols:
                quarter = parse_quarter_label(row_j.get(col))
                if quarter is not None:
                    break
            if quarter is None:
                for val in row_j.tolist():
                    quarter = parse_quarter_label(val)
                    if quarter is not None:
                        break
            if quarter is None:
                continue

            mw = to_float(row_j.get(size_col)) if size_col is not None else np.nan
            if math.isnan(mw):
                for val in row_j.tolist():
                    num = to_float(val)
                    if not math.isnan(num):
                        mw = num
                        break
            if math.isnan(mw):
                continue

            rows.append(
                {
                    "sheet_name": sheet_name,
                    "sheet_year": sheet_year,
                    "quarter": int(quarter),
                    "unannounced_mw": float(mw),
                    "source_row": int(j),
                }
            )

        break

    return rows


def extract_order_block(df: pd.DataFrame) -> pd.DataFrame:
    cols_norm = {normalize_col_name(c): c for c in df.columns}
    country_check_col = cols_norm.get("country check") or cols_norm.get("check country")
    platform_check_col = cols_norm.get("platform check") or cols_norm.get("check platform")

    core_candidates = [
        "date",
        "country",
        "size [mw]",
        "service scheme",
        "service time [years]",
        "service time  [years]",
        "customer",
        "#",
        "no_wtg_1",
        "platform",
        "multi platform",
        "multi platform.1",
        "multi platform.2",
    ]
    core_cols = [cols_norm[c] for c in core_candidates if c in cols_norm]
    if not core_cols:
        return df.iloc[0:0].copy()

    if country_check_col or platform_check_col:
        keep = pd.Series(False, index=df.index)
        if country_check_col:
            keep |= df[country_check_col].notna()
        if platform_check_col:
            keep |= df[platform_check_col].notna()
        block = df.loc[keep].copy()
    else:
        nonempty = df[core_cols].notna().any(axis=1).to_numpy()
        if not nonempty.any():
            return df.iloc[0:0].copy()
        start = int(np.argmax(nonempty))
        run = 0
        cutoff = len(df)
        for idx in range(start, len(df)):
            if nonempty[idx]:
                run = 0
            else:
                run += 1
                if run >= 5:
                    cutoff = idx - run + 1
                    break
        block = df.iloc[start:cutoff].copy()

    block = block[block[core_cols].notna().any(axis=1)]
    block.reset_index(drop=True, inplace=True)
    return block


def parse_economy_sheet(workbook: Path) -> pd.DataFrame:
    economy = pd.read_excel(workbook, sheet_name="Vestas Economy")
    if economy.empty:
        return pd.DataFrame(columns=["metric", "year", "value"])
    # Keep the primary economics block requested by user (rows 0..40)
    economy = economy.iloc[:41].copy()

    metric_col = economy.columns[0]
    year_map: dict[Any, int] = {}
    for col in economy.columns[1:]:
        if isinstance(col, (int, np.integer)):
            year = int(col)
        elif isinstance(col, (float, np.floating)) and float(col).is_integer():
            year = int(col)
        else:
            continue
        if 1900 <= year <= 2100:
            year_map[col] = year

    rows: list[dict[str, Any]] = []
    for _, row in economy.iterrows():
        metric = normalize_metric_name(row.get(metric_col))
        if metric is None:
            continue

        metric_norm = metric.strip()
        if re.fullmatch(r"\d{2}q[1-4]", metric_norm, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{4}", metric_norm):
            continue
        if metric_norm.lower() in {"in %", "q1", "q2", "q3", "q4"}:
            continue

        values = {}
        for source_col, year in year_map.items():
            value = to_float(row.get(source_col))
            if not math.isnan(value):
                values[year] = value

        if len(values) < 1:
            continue

        rec = {"metric": metric_norm}
        rec.update(values)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["metric", "year", "value"])

    econ = pd.DataFrame(rows)
    value_cols = [c for c in econ.columns if isinstance(c, int)]
    long = econ.melt(id_vars="metric", value_vars=value_cols, var_name="year", value_name="value")
    long = long.dropna(subset=["value"]).copy()
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.groupby(["metric", "year"], as_index=False)["value"].mean()
    return long


def parse_oi_sheets(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(workbook)
    oi_sheets = sorted(
        [s for s in xl.sheet_names if re.fullmatch(r"OI\s\d{4}", s)],
        key=lambda s: int(s.split()[1]),
    )

    order_rows: list[dict[str, Any]] = []
    platform_rows: list[dict[str, Any]] = []
    unannounced_rows: list[dict[str, Any]] = []

    for sheet in oi_sheets:
        year = int(sheet.split()[1])
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
        raw.columns = [str(c).strip() for c in raw.columns]
        unannounced_rows.extend(extract_unannounced_quarters(raw, sheet, year))
        block = extract_order_block(raw)

        for row_idx, row in block.iterrows():
            order_id = f"{year}_{row_idx:04d}"
            order_date = parse_mixed_date(row.get("date"))
            size_mw = to_float(row.get("size [MW]"))
            service_time_years = to_float(row.get("Service time  [years]"))
            if math.isnan(service_time_years):
                service_time_years = to_float(row.get("Service time [years]"))

            delivery_q = to_float(row.get("delivery q"))
            delivery_year = to_float(row.get("delivery y"))
            delivery_days = to_float(row.get("Delivery days"))
            if math.isnan(delivery_q) or delivery_q not in {1.0, 2.0, 3.0, 4.0}:
                delivery_q = np.nan
            if math.isnan(delivery_year):
                delivery_year = np.nan
            elif delivery_year < 1990 or delivery_year > 2035:
                delivery_year = np.nan

            if math.isnan(delivery_days) and pd.notna(order_date) and not math.isnan(delivery_q) and not math.isnan(delivery_year):
                q_end = quarter_end_date(int(delivery_year), int(delivery_q))
                delivery_days = float((q_end - order_date).days)

            country = normalize_country(row.get("country"))
            service_scheme = normalize_service_scheme(row.get("Service scheme"))
            customer = normalize_customer(row.get("customer"))

            order_data = {
                "order_id": order_id,
                "sheet_name": sheet,
                "sheet_year": year,
                "order_date": order_date,
                "order_year": int(order_date.year) if pd.notna(order_date) else year,
                "order_quarter": int(order_date.quarter) if pd.notna(order_date) else np.nan,
                "country": country,
                "service_scheme": service_scheme,
                "service_time_years": service_time_years,
                "customer": customer,
                "size_mw": size_mw,
                "delivery_q": int(delivery_q) if not math.isnan(delivery_q) else np.nan,
                "delivery_year": int(delivery_year) if not math.isnan(delivery_year) else np.nan,
                "delivery_days": delivery_days,
            }

            row_platform_indices: list[int] = []

            for spec in PLATFORM_SLOT_SPECS:
                qty = first_float(row, spec["qty"])
                platform_raw = first_text(row, spec["platform"])
                rotor_raw = first_float(row, spec["rotor"])
                mw_raw = first_float(row, spec["mw"])

                if (
                    platform_raw is None
                    and math.isnan(qty)
                    and math.isnan(rotor_raw)
                    and math.isnan(mw_raw)
                ):
                    continue

                platform = normalize_platform(platform_raw)
                rotor_guess, mw_guess = parse_platform_specs(platform)

                rotor = rotor_raw if not math.isnan(rotor_raw) else rotor_guess
                mw_rating = mw_raw if not math.isnan(mw_raw) else mw_guess
                slot_mw = qty * mw_rating if (not math.isnan(qty) and not math.isnan(mw_rating)) else np.nan

                platform_rows.append(
                    {
                        "order_id": order_id,
                        "sheet_year": year,
                        "order_year": int(order_data["order_year"]),
                        "order_date": order_data["order_date"],
                        "country": country,
                        "service_scheme": service_scheme,
                        "service_time_years": service_time_years,
                        "customer": customer,
                        "delivery_q": order_data["delivery_q"],
                        "delivery_year": order_data["delivery_year"],
                        "delivery_days": order_data["delivery_days"],
                        "order_size_mw": size_mw,
                        "slot": spec["slot"],
                        "platform": platform,
                        "turbines_qty": qty,
                        "rotor_m": rotor,
                        "mw_rating": mw_rating,
                        "slot_mw": slot_mw,
                    }
                )
                row_platform_indices.append(len(platform_rows) - 1)

            if row_platform_indices:
                if len(row_platform_indices) == 1 and not math.isnan(size_mw):
                    idx = row_platform_indices[0]
                    if math.isnan(platform_rows[idx]["slot_mw"]):
                        platform_rows[idx]["slot_mw"] = size_mw
                        q = platform_rows[idx]["turbines_qty"]
                        p = platform_rows[idx]["mw_rating"]
                        if math.isnan(q) and not math.isnan(p) and p != 0:
                            platform_rows[idx]["turbines_qty"] = size_mw / p
                first_idx = row_platform_indices[0]
                order_data["primary_platform"] = platform_rows[first_idx]["platform"]
            else:
                order_data["primary_platform"] = "Unknown"
                if not math.isnan(size_mw):
                    platform_rows.append(
                        {
                            "order_id": order_id,
                            "sheet_year": year,
                            "order_year": int(order_data["order_year"]),
                            "order_date": order_data["order_date"],
                            "country": country,
                            "service_scheme": service_scheme,
                            "service_time_years": service_time_years,
                            "customer": customer,
                            "delivery_q": order_data["delivery_q"],
                            "delivery_year": order_data["delivery_year"],
                            "delivery_days": order_data["delivery_days"],
                            "order_size_mw": size_mw,
                            "slot": 0,
                            "platform": "Unknown",
                            "turbines_qty": np.nan,
                            "rotor_m": np.nan,
                            "mw_rating": np.nan,
                            "slot_mw": size_mw,
                        }
                    )

            order_rows.append(order_data)

    orders = pd.DataFrame(order_rows)
    platforms = pd.DataFrame(platform_rows)
    unannounced = pd.DataFrame(unannounced_rows)
    if orders.empty:
        if unannounced.empty:
            unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw", "source_row"])
        return orders, platforms, unannounced

    for col in ["size_mw", "service_time_years", "delivery_days"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    for col in ["turbines_qty", "rotor_m", "mw_rating", "slot_mw", "order_size_mw"]:
        if col in platforms.columns:
            platforms[col] = pd.to_numeric(platforms[col], errors="coerce")

    valid_ids = set(orders.loc[orders["size_mw"] > 0, "order_id"])
    valid_ids.update(platforms.loc[platforms["slot_mw"] > 0, "order_id"])
    orders = orders[orders["order_id"].isin(valid_ids)].copy()
    platforms = platforms[platforms["order_id"].isin(valid_ids)].copy()

    order_platform_mw = platforms.groupby("order_id")["slot_mw"].sum(min_count=1)
    order_turbines = platforms.groupby("order_id")["turbines_qty"].sum(min_count=1)
    orders["platform_mw_sum"] = orders["order_id"].map(order_platform_mw)
    orders["turbines_qty_sum"] = orders["order_id"].map(order_turbines)
    orders["mw_gap"] = orders["size_mw"] - orders["platform_mw_sum"]

    orders["delivery_time_years"] = orders["delivery_days"] / 365.25
    platforms["delivery_time_years"] = platforms["delivery_days"] / 365.25

    if unannounced.empty:
        unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw", "source_row"])
    else:
        unannounced["sheet_year"] = pd.to_numeric(unannounced["sheet_year"], errors="coerce").astype("Int64")
        unannounced["quarter"] = pd.to_numeric(unannounced["quarter"], errors="coerce").astype("Int64")
        unannounced["unannounced_mw"] = pd.to_numeric(unannounced["unannounced_mw"], errors="coerce")
        unannounced = unannounced.dropna(subset=["sheet_year", "quarter", "unannounced_mw"]).copy()
        unannounced["sheet_year"] = unannounced["sheet_year"].astype(int)
        unannounced["quarter"] = unannounced["quarter"].astype(int)
        unannounced = (
            unannounced.groupby(["sheet_name", "sheet_year", "quarter"], as_index=False)["unannounced_mw"].sum()
        )

    return orders, platforms, unannounced


def find_default_workbook() -> Path | None:
    patterns = ("*.xlsx", "*.xlsm", "*.xls")
    for pattern in patterns:
        files = sorted(Path(".").glob(pattern))
        files = [f for f in files if not f.name.startswith("~$")]
        if files:
            return files[0]
    return None


@st.cache_data(show_spinner=False)
def load_data(cache_key: str, workbook_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = cache_key
    workbook = Path(workbook_path)
    economy = parse_economy_sheet(workbook)
    orders, platforms, unannounced = parse_oi_sheets(workbook)
    return economy, orders, platforms, unannounced


def mw_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} MW"


def int_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f}"


def days_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} days"


def top_n_by_mw(df: pd.DataFrame, col: str, value_col: str, n: int) -> list[str]:
    if df.empty:
        return []
    grouped = df.groupby(col, as_index=False)[value_col].sum()
    grouped = grouped.sort_values(value_col, ascending=False).head(n)
    return grouped[col].tolist()


def render_header_metrics(orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    total_orders = orders["order_id"].nunique()
    announced_mw = orders["size_mw"].sum()
    unannounced_mw = 0.0
    if unannounced is not None and not unannounced.empty and "unannounced_mw" in unannounced.columns:
        unannounced_mw = float(pd.to_numeric(unannounced["unannounced_mw"], errors="coerce").fillna(0).sum())
    total_mw = announced_mw + unannounced_mw
    total_platform_mw = platforms["slot_mw"].sum()
    avg_order_mw = orders["size_mw"].mean()
    avg_delivery_days = orders["delivery_days"].mean()
    countries = orders["country"].nunique()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Orders", int_fmt(total_orders))
    m2.metric(
        "Ordered MW",
        mw_fmt(total_mw),
        delta=f"hereof unannounced {unannounced_mw:,.0f} MW",
        delta_color="off",
    )
    m3.metric("Platform-Mapped MW", mw_fmt(total_platform_mw))
    m4.metric("Avg Order Size", mw_fmt(avg_order_mw))
    m5.metric("Avg Delivery Time", days_fmt(avg_delivery_days))
    m6.metric("Countries", int_fmt(countries))


def render_information_page() -> None:
    st.subheader("Information")
    st.markdown("**Data source**")
    st.write("This dashboard is build on public available data.")

    st.markdown("**How to use and navigate**")
    st.write(
        "Use the sidebar filters to narrow years, countries, service schemes, platforms, and minimum order size."
    )
    st.write(
        "Navigate tabs from high-level trends (Overall/Yearly/Quarterly) to deep dives (Across Years, Platform, Country, Delivery, Correlations)."
    )

    st.markdown("**About me**")
    st.write(
        "Thomas Buhl is a wind-energy engineering leader with 20+ years across research and industry, including professor/department-head and director/VP roles."
    )
    st.write(
        "He combines deep technical expertise in wind turbine design and optimization with international people leadership and strategy execution."
    )

    st.markdown("**Disclaimer**")
    st.write(
        "This dashboard is provided for informational purposes only. Data is compiled from public sources and may contain gaps, delays, or inaccuracies."
    )
    st.write(
        "Always verify critical figures with official company reporting before using the information for financial, legal, strategic, or investment decisions."
    )
    st.markdown(
        """
## Sources and attribution
All figures are compiled manually from Vestas public investor communications (press releases/announcements) and annual reports.
Each datapoint should be verified against the original source documents.

## No affiliation / no endorsement
This is an unofficial, independent dashboard and is not affiliated with, endorsed by, or sponsored by Vestas.

## IP and rights
The dataset in this repository is my own compilation (selection/structure) based on publicly available sources.
Underlying source documents remain the property of their respective owners.

## Disclaimer
The data and dashboard are provided "as is" without warranties of any kind. I make no guarantees of accuracy, completeness, or fitness for a particular purpose. Use at your own risk. I am not responsible or liable for any losses or damages arising from use of the data or dashboard.
"""
    )


def render_overall_economics(economy: pd.DataFrame) -> None:
    st.subheader("Overall Economics")
    if economy.empty:
        st.info("No parsable data found in `Vestas Economy`.")
        return

    years = sorted(economy["year"].unique().tolist())
    y_min, y_max = int(min(years)), int(max(years))
    year_range = st.slider("Economy year range", y_min, y_max, (max(y_min, 2007), y_max))
    econ = economy[economy["year"].between(year_range[0], year_range[1])].copy()
    if econ.empty:
        st.warning("No economic values available for the selected range.")
        return

    metric_options = sorted(econ["metric"].unique().tolist())

    def metric_key(text: str) -> str:
        norm = normalize_metric_name(text)
        if norm is None:
            return ""
        return re.sub(r"[^a-z0-9]+", "", norm.lower())

    def find_metric(required_tokens: list[str], any_tokens: list[str] | None = None) -> str | None:
        for metric in metric_options:
            key = metric_key(metric)
            if not all(tok in key for tok in required_tokens):
                continue
            if any_tokens is not None and not any(tok in key for tok in any_tokens):
                continue
            return metric
        return None

    def format_metric_value(metric: str, value: float) -> str:
        key = metric_key(metric)
        if "meurmw" in key:
            return f"{value:,.3f}"
        if "bneur" in key:
            return f"{value:,.2f}"
        if "mw" in key and "meurmw" not in key:
            return f"{value:,.0f}"
        return f"{value:,.0f}"

    def plot_metric_group(metrics: list[str | None], title: str, y_title: str, height: int = 380) -> None:
        chosen = [m for m in metrics if m is not None]
        if not chosen:
            st.info(f"No metrics found for: {title}")
            return
        frame = econ[econ["metric"].isin(chosen)].copy()
        if frame.empty:
            st.info(f"No values found for: {title}")
            return
        fig = px.line(
            frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            title=title,
            height=height,
        )
        fig.update_layout(
            yaxis_title=y_title,
            legend_title_text="",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    revenue_metric = find_metric(["revenue"])
    gross_profit_metric = find_metric(["gross", "profit"])
    sales_projects_metric = find_metric(["sales", "projects"])
    sales_service_metric = find_metric(["sales", "service"])

    order_intake_beur_metric = find_metric(["order", "bneur"], any_tokens=["intake", "inkake"])
    order_backlog_wt_beur_metric = find_metric(["order", "backlog", "wind", "bneur"])
    order_backlog_service_beur_metric = find_metric(["order", "backlog", "service", "bneur"])

    order_intake_mw_metric = find_metric(["order", "mw"], any_tokens=["intake", "inkake"])
    order_backlog_mw_metric = find_metric(["order", "backlog", "wind", "mw"])
    deliveries_mw_metric = find_metric(["deliveries", "mw"])

    avg_mw_price_metric = find_metric(["averagemwprice"]) or find_metric(["average", "mw", "price"])

    latest_year = int(econ["year"].max())
    base_year = int(year_range[0])
    latest = econ[econ["year"] == latest_year].set_index("metric")["value"]
    baseline = econ[econ["year"] == base_year].set_index("metric")["value"]
    card_metrics = [
        revenue_metric,
        gross_profit_metric,
        sales_projects_metric,
        sales_service_metric,
        order_intake_beur_metric,
        order_backlog_service_beur_metric,
        order_backlog_wt_beur_metric,
        avg_mw_price_metric,
    ]
    card_metrics = [m for m in card_metrics if m is not None]

    if card_metrics:
        for idx in range(0, len(card_metrics), 4):
            row_metrics = card_metrics[idx : idx + 4]
            cols = st.columns(len(row_metrics))
            for col, metric_name in zip(cols, row_metrics, strict=False):
                val = latest.get(metric_name, np.nan)
                value_text = "-"
                if pd.notna(val):
                    value_text = f"{format_metric_value(metric_name, val)} ({latest_year})"
                base_val = baseline.get(metric_name, np.nan)
                delta_text = None
                if pd.notna(val) and pd.notna(base_val):
                    delta_value = val - base_val
                    delta_text = f"{delta_value:+,.2f} ({base_year})"
                col.metric(
                    metric_name,
                    value_text,
                    delta=delta_text,
                )

    g1, g2 = st.columns(2)
    with g1:
        plot_metric_group(
            [revenue_metric, gross_profit_metric, sales_projects_metric, sales_service_metric],
            "mEUR Metrics (Revenue, Gross Profit, Project Sales, Service Sales)",
            "mEUR",
        )
    with g2:
        plot_metric_group(
            [order_intake_beur_metric, order_backlog_wt_beur_metric, order_backlog_service_beur_metric],
            "bnEUR Metrics (Order Intake and Backlog)",
            "bnEUR",
        )

    g3, g4 = st.columns(2)
    with g3:
        plot_metric_group(
            [order_intake_mw_metric, order_backlog_mw_metric, deliveries_mw_metric],
            "MW Metrics (Order Intake, Backlog, Deliveries)",
            "MW",
        )
    with g4:
        plot_metric_group(
            [avg_mw_price_metric],
            "Average MW Price [mEUR/MW]",
            "mEUR/MW",
        )

    st.markdown("**Custom Metric Explorer**")
    default_custom = [
        m
        for m in [
            revenue_metric,
            gross_profit_metric,
            sales_projects_metric,
            sales_service_metric,
            order_intake_beur_metric,
            order_backlog_wt_beur_metric,
            order_backlog_service_beur_metric,
            order_intake_mw_metric,
            order_backlog_mw_metric,
            avg_mw_price_metric,
        ]
        if m is not None
    ]
    selected_metrics = st.multiselect(
        "Select custom metrics",
        options=metric_options,
        default=default_custom if default_custom else metric_options[:8],
    )
    custom_frame = econ[econ["metric"].isin(selected_metrics)].copy()
    if not custom_frame.empty:
        fig_custom = px.line(
            custom_frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            height=460,
            title="Custom Economics View",
        )
        fig_custom.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_custom, width="stretch")

    st.markdown("**All Economy Parameters (Rows 0-40)**")
    econ_table = (
        economy.pivot(index="metric", columns="year", values="value")
        .sort_index()
        .reset_index()
        .rename(columns={"metric": "Parameter"})
    )
    econ_table.columns = [str(c) for c in econ_table.columns]
    st.dataframe(econ_table, width="stretch", hide_index=True)


def render_yearly_overview(orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    st.subheader("Year-by-Year Order Intake Overview")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    yearly = (
        orders.groupby("order_year", as_index=False)
        .agg(
            ordered_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
            countries=("country", "nunique"),
            customers=("customer", "nunique"),
        )
        .sort_values("order_year")
    )
    platform_yearly = platforms.groupby("order_year", as_index=False).agg(
        platform_mw=("slot_mw", "sum"),
        platforms=("platform", "nunique"),
    )
    yearly = yearly.merge(platform_yearly, on="order_year", how="left")
    if unannounced is not None and not unannounced.empty:
        unannounced_yearly = (
            unannounced.groupby("sheet_year", as_index=False)["unannounced_mw"].sum().rename(columns={"sheet_year": "order_year"})
        )
        yearly = yearly.merge(unannounced_yearly, on="order_year", how="left")
    else:
        yearly["unannounced_mw"] = 0.0

    yearly["unannounced_mw"] = yearly["unannounced_mw"].fillna(0.0)
    yearly["total_with_unannounced_mw"] = yearly["ordered_mw"] + yearly["unannounced_mw"]

    c1, c2 = st.columns(2)
    with c1:
        mw_mix = yearly.melt(
            id_vars=["order_year"],
            value_vars=["ordered_mw", "unannounced_mw"],
            var_name="mw_type",
            value_name="mw",
        )
        mw_type_labels = {
            "ordered_mw": "Announced MW",
            "unannounced_mw": "Unannounced MW",
        }
        mw_mix["mw_type"] = mw_mix["mw_type"].map(mw_type_labels)
        fig1 = px.bar(
            mw_mix,
            x="order_year",
            y="mw",
            color="mw_type",
            barmode="stack",
            template=plotly_template(),
            labels={"order_year": "Year", "mw": "MW", "mw_type": ""},
            title="Announced and Unannounced MW by Year",
            height=410,
        )
        fig1.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["orders"],
                mode="lines+markers",
                name="Orders",
                line=dict(width=3),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["avg_order_mw"],
                mode="lines+markers",
                name="Avg order MW",
                yaxis="y2",
                line=dict(width=3),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["total_with_unannounced_mw"],
                mode="lines+markers",
                name="Total MW incl. unannounced",
                line=dict(width=3, dash="dot"),
            )
        )
        fig2.update_layout(
            template=plotly_template(),
            title="Order Count, Avg Order Size, and Total MW incl. Unannounced",
            height=410,
            yaxis=dict(title="Orders / MW"),
            yaxis2=dict(title="Avg MW", overlaying="y", side="right"),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig2, width="stretch")

    st.dataframe(
        yearly.rename(
            columns={
                "order_year": "Year",
                "ordered_mw": "Ordered MW",
                "unannounced_mw": "Unannounced MW",
                "total_with_unannounced_mw": "Total MW incl. Unannounced",
                "platform_mw": "Platform-Mapped MW",
                "avg_order_mw": "Avg Order MW",
                "orders": "Orders",
                "countries": "Countries",
                "customers": "Customers",
                "platforms": "Platforms",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chosen_year = st.selectbox("Year detail", options=sorted(yearly["order_year"].unique().tolist(), reverse=True))
    yr_orders = orders[orders["order_year"] == chosen_year]
    yr_platforms = platforms[platforms["order_year"] == chosen_year]
    if unannounced is not None and not unannounced.empty:
        yr_unann = unannounced[unannounced["sheet_year"] == chosen_year]
        if not yr_unann.empty:
            total_unann = yr_unann["unannounced_mw"].sum()
            q_breakdown = ", ".join([f"Q{int(r.quarter)}: {r.unannounced_mw:,.0f} MW" for r in yr_unann.itertuples()])
            st.caption(f"Unannounced in {chosen_year}: {total_unann:,.0f} MW ({q_breakdown})")

    c3, c4 = st.columns(2)
    with c3:
        top_country = (
            yr_orders.groupby("country", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
            .head(12)
        )
        fig3 = px.bar(
            top_country,
            x="size_mw",
            y="country",
            orientation="h",
            template=plotly_template(),
            title=f"Top Countries in {chosen_year}",
            labels={"size_mw": "MW", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        top_platform = (
            yr_platforms.groupby("platform", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(12)
        )
        fig4 = px.bar(
            top_platform,
            x="slot_mw",
            y="platform",
            orientation="h",
            template=plotly_template(),
            title=f"Top Platforms in {chosen_year}",
            labels={"slot_mw": "MW", "platform": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")


def render_quarterly_analytics(orders: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    st.subheader("Quarterly Analytics")
    if orders.empty and (unannounced is None or unannounced.empty):
        st.info("No quarterly data available.")
        return

    announced = orders.dropna(subset=["order_date"]).copy()
    if not announced.empty:
        announced["year"] = announced["order_date"].dt.year.astype(int)
        announced["quarter"] = announced["order_date"].dt.quarter.astype(int)
        announced_q = (
            announced.groupby(["year", "quarter"], as_index=False)
            .agg(
                announced_mw=("size_mw", "sum"),
                orders=("order_id", "nunique"),
                avg_order_mw=("size_mw", "mean"),
                avg_service_time=("service_time_years", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
        )
    else:
        announced_q = pd.DataFrame(columns=["year", "quarter", "announced_mw", "orders", "avg_order_mw", "avg_service_time", "avg_delivery_days"])

    if unannounced is not None and not unannounced.empty:
        unann_q = (
            unannounced.groupby(["sheet_year", "quarter"], as_index=False)["unannounced_mw"]
            .sum()
            .rename(columns={"sheet_year": "year"})
        )
    else:
        unann_q = pd.DataFrame(columns=["year", "quarter", "unannounced_mw"])

    quarter_df = announced_q.merge(unann_q, on=["year", "quarter"], how="outer")
    if quarter_df.empty:
        st.info("No quarterly data available.")
        return

    quarter_df["announced_mw"] = pd.to_numeric(quarter_df["announced_mw"], errors="coerce").fillna(0.0)
    quarter_df["unannounced_mw"] = pd.to_numeric(quarter_df["unannounced_mw"], errors="coerce").fillna(0.0)
    quarter_df["orders"] = pd.to_numeric(quarter_df["orders"], errors="coerce").fillna(0).astype(int)
    quarter_df["total_mw"] = quarter_df["announced_mw"] + quarter_df["unannounced_mw"]
    quarter_df["quarter"] = pd.to_numeric(quarter_df["quarter"], errors="coerce").astype("Int64")
    quarter_df = quarter_df.dropna(subset=["year", "quarter"]).copy()
    quarter_df["year"] = quarter_df["year"].astype(int)
    quarter_df["quarter"] = quarter_df["quarter"].astype(int)
    quarter_df = quarter_df.sort_values(["year", "quarter"])
    quarter_df["quarter_label"] = quarter_df["quarter"].apply(lambda q: f"Q{q}")
    quarter_df["year_quarter"] = quarter_df["year"].astype(str) + "-" + quarter_df["quarter_label"]

    c1, c2 = st.columns(2)
    with c1:
        mw_mix = quarter_df.melt(
            id_vars=["year_quarter", "year", "quarter"],
            value_vars=["announced_mw", "unannounced_mw"],
            var_name="mw_type",
            value_name="mw",
        )
        mw_mix["mw_type"] = mw_mix["mw_type"].map(
            {"announced_mw": "Announced MW", "unannounced_mw": "Unannounced MW"}
        )
        fig1 = px.bar(
            mw_mix,
            x="year_quarter",
            y="mw",
            color="mw_type",
            barmode="stack",
            template=plotly_template(),
            labels={"year_quarter": "Quarter", "mw": "MW", "mw_type": ""},
            title="Quarterly MW Mix (Announced vs Unannounced)",
            height=420,
        )
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        pivot = quarter_df.pivot(index="year", columns="quarter_label", values="total_mw").fillna(0)
        fig2 = px.imshow(
            pivot,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Total MW Heatmap (Year x Quarter)",
            height=420,
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        by_quarter = (
            quarter_df.groupby("quarter_label", as_index=False)
            .agg(
                avg_announced_mw=("announced_mw", "mean"),
                avg_unannounced_mw=("unannounced_mw", "mean"),
                avg_total_mw=("total_mw", "mean"),
            )
            .sort_values("quarter_label")
        )
        fig3 = px.bar(
            by_quarter,
            x="quarter_label",
            y=["avg_announced_mw", "avg_unannounced_mw", "avg_total_mw"],
            barmode="group",
            template=plotly_template(),
            title="Average Quarterly MW by Quarter Number",
            labels={"value": "Average MW", "quarter_label": "Quarter"},
            height=420,
        )
        fig3.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        corr_cols = ["announced_mw", "unannounced_mw", "total_mw", "orders", "avg_order_mw", "avg_service_time", "avg_delivery_days"]
        corr_src = quarter_df[corr_cols].dropna()
        if len(corr_src) >= 4:
            corr = corr_src.corr(numeric_only=True)
            fig4 = px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Quarterly Correlation Matrix",
                height=420,
            )
            fig4.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig4, width="stretch")
        else:
            st.info("Not enough complete quarterly rows for a correlation matrix.")

    st.dataframe(
        quarter_df.rename(
            columns={
                "year": "Year",
                "quarter_label": "Quarter",
                "announced_mw": "Announced MW",
                "unannounced_mw": "Unannounced MW",
                "total_mw": "Total MW",
                "orders": "Orders",
                "avg_order_mw": "Avg Order MW",
                "avg_service_time": "Avg Service Time",
                "avg_delivery_days": "Avg Delivery Days",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_across_years(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across-Years Analytics")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    tab_country, tab_platform, tab_service, tab_rotor, tab_customer = st.tabs(
        ["Country Stats", "Platform Stats", "Service & Time", "Rotor and MW Rating", "Customer Stats"]
    )

    with tab_country:
        top_n = st.slider("Top countries to display", 5, 20, 12, key="top_country_years")
        top_countries = top_n_by_mw(orders, "country", "size_mw", top_n)
        country_year = (
            orders[orders["country"].isin(top_countries)]
            .groupby(["order_year", "country"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.area(
            country_year,
            x="order_year",
            y="size_mw",
            color="country",
            template=plotly_template(),
            labels={"size_mw": "MW", "order_year": "Year"},
            title="Country MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            orders.groupby("country", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_platform:
        top_n = st.slider("Top platforms to display", 5, 20, 12, key="top_platform_years")
        p_base = platforms[platforms["platform"] != "Unknown"]
        top_platforms = top_n_by_mw(p_base, "platform", "slot_mw", top_n)
        platform_year = (
            p_base[p_base["platform"].isin(top_platforms)]
            .groupby(["order_year", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.area(
            platform_year,
            x="order_year",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            labels={"slot_mw": "MW", "order_year": "Year"},
            title="Platform MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            p_base.groupby("platform", as_index=False)
            .agg(
                total_mw=("slot_mw", "sum"),
                orders=("order_id", "nunique"),
                avg_delivery_days=("delivery_days", "mean"),
            )
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_service:
        c1, c2 = st.columns(2)

        with c1:
            service_year = (
                orders.groupby(["order_year", "service_scheme"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            fig = px.bar(
                service_year,
                x="order_year",
                y="size_mw",
                color="service_scheme",
                template=plotly_template(),
                labels={"size_mw": "MW", "order_year": "Year"},
                title="Service Scheme Mix (MW) Across Years",
                height=430,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, width="stretch")

        with c2:
            time_year = (
                orders.dropna(subset=["service_time_years"])
                .groupby("order_year", as_index=False)
                .agg(
                    avg_service_time=("service_time_years", "mean"),
                    min_service_time=("service_time_years", "min"),
                    max_service_time=("service_time_years", "max"),
                )
            )
            fig = px.line(
                time_year,
                x="order_year",
                y=["avg_service_time", "min_service_time", "max_service_time"],
                template=plotly_template(),
                markers=True,
                labels={"value": "Years", "order_year": "Year"},
                title="Service Contract Length Across Years (avg/min/max)",
                height=430,
            )
            fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, width="stretch")

    with tab_rotor:
        rot = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
        rotor_stats = (
            rot.dropna(subset=["rotor_m"])
            .groupby("order_year", as_index=False)
            .agg(rotor_avg=("rotor_m", "mean"), rotor_min=("rotor_m", "min"), rotor_max=("rotor_m", "max"))
        )
        mw_stats = (
            rot.dropna(subset=["mw_rating"])
            .groupby("order_year", as_index=False)
            .agg(mw_avg=("mw_rating", "mean"), mw_min=("mw_rating", "min"), mw_max=("mw_rating", "max"))
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=rotor_stats["order_year"], y=rotor_stats["rotor_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(43,119,173,0.22)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#1f5b86"),
                    name="Average",
                )
            )
            fig_r.update_layout(
                template=plotly_template(),
                title="Rotor Size Across Years (min/avg/max)",
                yaxis_title="Rotor diameter (m)",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_r, width="stretch")

        with c2:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=mw_stats["order_year"], y=mw_stats["mw_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(241,120,50,0.20)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#ca6318"),
                    name="Average",
                )
            )
            fig_m.update_layout(
                template=plotly_template(),
                title="Platform MW Rating Across Years (min/avg/max)",
                yaxis_title="MW rating",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_m, width="stretch")

    with tab_customer:
        cust_base = orders[orders["customer"] != "Unknown"].copy()
        top_n = st.slider("Top customers to display", 5, 25, 12, key="top_customers_years")
        top_customers = top_n_by_mw(cust_base, "customer", "size_mw", top_n)
        by_year = (
            cust_base[cust_base["customer"].isin(top_customers)]
            .groupby(["order_year", "customer"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.bar(
            by_year,
            x="order_year",
            y="size_mw",
            color="customer",
            template=plotly_template(),
            title="Customer MW Bought Across Years",
            labels={"size_mw": "MW", "order_year": "Year"},
            height=430,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            cust_base.groupby("customer", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(25)
        )
        st.dataframe(table, width="stretch", hide_index=True)


def render_platform_lens(platforms: pd.DataFrame) -> None:
    st.subheader("Across Platforms")
    if platforms.empty:
        st.info("No platform-level rows after filtering.")
        return

    base = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
    if base.empty:
        st.info("No platform rows with MW available.")
        return

    options = sorted(base["platform"].unique().tolist())
    default_sel = top_n_by_mw(base, "platform", "slot_mw", min(6, len(options)))
    selected = st.multiselect("Platforms to focus", options=options, default=default_sel)
    view = base[base["platform"].isin(selected)] if selected else base

    timeline = view.groupby(["order_year", "platform"], as_index=False)["slot_mw"].sum()
    fig = px.line(
        timeline,
        x="order_year",
        y="slot_mw",
        color="platform",
        template=plotly_template(),
        markers=True,
        title="Timeline: Sold MW per Platform per Year",
        labels={"slot_mw": "MW", "order_year": "Year"},
        height=430,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        svc_heat = view.groupby(["platform", "service_scheme"], as_index=False)["slot_mw"].sum()
        pivot = svc_heat.pivot(index="platform", columns="service_scheme", values="slot_mw").fillna(0)
        if not pivot.empty:
            fig_h = px.imshow(
                pivot,
                text_auto=".0f",
                color_continuous_scale="Blues",
                aspect="auto",
                title="Across Platforms: Service Scheme Mix (MW)",
                height=460,
            )
            fig_h.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_h, width="stretch")

    with c2:
        svc_time = view.dropna(subset=["service_time_years"])
        if not svc_time.empty:
            fig_b = px.box(
                svc_time,
                x="platform",
                y="service_time_years",
                template=plotly_template(),
                title="Across Platforms: Service Time Distribution",
                labels={"service_time_years": "Service time (years)", "platform": ""},
                height=460,
            )
            fig_b.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_b, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        platform_country = view.groupby(["platform", "country"], as_index=False)["slot_mw"].sum()
        top_pc = platform_country.sort_values("slot_mw", ascending=False).head(140)
        fig_t = px.treemap(
            top_pc,
            path=["platform", "country"],
            values="slot_mw",
            template=plotly_template(),
            title="Across Platforms: Country Mix",
            height=520,
        )
        fig_t.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_t, width="stretch")

    with c4:
        focus_platform = st.selectbox("Customer view by platform", options=sorted(view["platform"].unique().tolist()))
        customer_data = (
            view[(view["platform"] == focus_platform) & (view["customer"] != "Unknown")]
            .groupby("customer", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(15)
        )
        fig_c = px.bar(
            customer_data,
            x="slot_mw",
            y="customer",
            orientation="h",
            template=plotly_template(),
            title=f"Across Platforms: Top Customers for {focus_platform}",
            labels={"slot_mw": "MW", "customer": ""},
            height=520,
        )
        fig_c.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_c, width="stretch")

    platform_delivery = (
        view.dropna(subset=["delivery_days"])
        .groupby("platform", as_index=False)
        .agg(avg_days=("delivery_days", "mean"), median_days=("delivery_days", "median"), orders=("order_id", "nunique"))
        .sort_values("avg_days", ascending=False)
    )
    if not platform_delivery.empty:
        fig_d = px.bar(
            platform_delivery.head(20),
            x="avg_days",
            y="platform",
            orientation="h",
            template=plotly_template(),
            title="Across Platforms: Average Delivery Time",
            labels={"avg_days": "Days", "platform": ""},
            height=470,
        )
        fig_d.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_d, width="stretch")


def render_country_maps(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    selected_orders: pd.DataFrame,
    selected_platforms: pd.DataFrame,
) -> None:
    st.markdown("### Zoomable Country Maps")
    map_scope = st.radio(
        "Map scope",
        options=["All countries", "Selected countries"],
        index=0,
        horizontal=True,
        key="country_map_scope",
    )
    if map_scope == "Selected countries":
        map_orders = selected_orders.copy()
        map_platforms = selected_platforms.copy()
    else:
        map_orders = orders.copy()
        map_platforms = platforms.copy()

    map_orders["size_mw"] = pd.to_numeric(map_orders["size_mw"], errors="coerce")
    map_orders["service_time_years"] = pd.to_numeric(map_orders["service_time_years"], errors="coerce")
    map_orders["delivery_days"] = pd.to_numeric(map_orders["delivery_days"], errors="coerce")

    map_base = (
        map_orders.groupby("country", as_index=False)
        .agg(
            total_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
            avg_service_time=("service_time_years", "mean"),
            min_service_time=("service_time_years", "min"),
            max_service_time=("service_time_years", "max"),
            avg_delivery_days=("delivery_days", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
        .copy()
    )

    scheme_rank = (
        map_orders.groupby(["country", "service_scheme"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["country", "size_mw"], ascending=[True, False])
        .drop_duplicates("country")
        .rename(columns={"service_scheme": "dominant_service_scheme", "size_mw": "dominant_service_mw"})
    )
    map_base = map_base.merge(
        scheme_rank[["country", "dominant_service_scheme", "dominant_service_mw"]],
        on="country",
        how="left",
    )

    valid_map_platforms = map_platforms[(map_platforms["platform"] != "Unknown") & (map_platforms["slot_mw"] > 0)].copy()
    platform_rank = (
        valid_map_platforms.groupby(["country", "platform"], as_index=False)["slot_mw"]
        .sum()
        .sort_values(["country", "slot_mw"], ascending=[True, False])
        .drop_duplicates("country")
        .rename(columns={"platform": "dominant_platform", "slot_mw": "dominant_platform_mw"})
    )
    map_base = map_base.merge(
        platform_rank[["country", "dominant_platform", "dominant_platform_mw"]],
        on="country",
        how="left",
    )
    map_base["dominant_service_scheme"] = map_base["dominant_service_scheme"].fillna("Unknown")
    map_base["dominant_platform"] = map_base["dominant_platform"].fillna("Unknown")

    platform_options = ["All platforms"]
    if not valid_map_platforms.empty:
        platform_options.extend(sorted(valid_map_platforms["platform"].dropna().unique().tolist()))
    selected_platform = st.selectbox(
        "Platform overlay for maps",
        options=platform_options,
        index=0,
        key="country_map_platform_overlay",
    )

    if valid_map_platforms.empty:
        overlay = pd.DataFrame(columns=["country", "platform_overlay_mw"])
    elif selected_platform == "All platforms":
        overlay = (
            valid_map_platforms.groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )
    else:
        overlay = (
            valid_map_platforms[valid_map_platforms["platform"] == selected_platform]
            .groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )

    map_base = map_base.merge(overlay, on="country", how="left")
    map_base["platform_overlay_mw"] = pd.to_numeric(map_base["platform_overlay_mw"], errors="coerce").fillna(0.0)
    map_base["iso3"] = map_base["country"].map(map_country_for_geo)
    map_geo = map_base[map_base["iso3"].notna()].copy()

    if map_geo.empty:
        st.info("No mappable countries available for the current scope.")
        return

    map_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
        "avg_service_time": "Average service length (years)",
        "avg_delivery_days": "Average delivery time (days)",
    }
    size_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
    }
    color_metric_labels = {
        "avg_service_time": "Average service length (years)",
        "avg_delivery_days": "Average delivery time (days)",
        "orders": "Number of orders",
    }

    c5, c6, c7 = st.columns(3)
    with c5:
        choropleth_metric = st.selectbox(
            "Choropleth metric",
            options=list(map_metric_labels.keys()),
            format_func=lambda k: map_metric_labels[k],
            key="country_map_choropleth_metric",
        )
    with c6:
        bubble_size_metric = st.selectbox(
            "Bubble size metric",
            options=list(size_metric_labels.keys()),
            format_func=lambda k: size_metric_labels[k],
            key="country_map_bubble_size_metric",
        )
    with c7:
        bubble_color_metric = st.selectbox(
            "Bubble color metric",
            options=list(color_metric_labels.keys()),
            format_func=lambda k: color_metric_labels[k],
            key="country_map_bubble_color_metric",
        )

    map_geo[choropleth_metric] = pd.to_numeric(map_geo[choropleth_metric], errors="coerce")
    map_geo[bubble_size_metric] = pd.to_numeric(map_geo[bubble_size_metric], errors="coerce")
    map_geo[bubble_color_metric] = pd.to_numeric(map_geo[bubble_color_metric], errors="coerce")
    map_geo["orders"] = pd.to_numeric(map_geo["orders"], errors="coerce")

    choropleth_data = map_geo.dropna(subset=[choropleth_metric]).copy()
    if choropleth_metric in {"total_mw", "platform_overlay_mw", "orders"}:
        choropleth_data = choropleth_data[choropleth_data[choropleth_metric] > 0]

    bubble_data = map_geo.dropna(subset=[bubble_size_metric]).copy()
    bubble_data = bubble_data[bubble_data[bubble_size_metric] > 0]
    if bubble_data[bubble_color_metric].notna().any():
        bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()
    else:
        bubble_color_metric = "orders"
        bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()

    m1, m2 = st.columns(2)
    with m1:
        if choropleth_data.empty:
            st.info("No values available for the selected choropleth metric.")
        else:
            fig_map = px.choropleth(
                choropleth_data,
                locations="iso3",
                locationmode="ISO-3",
                color=choropleth_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_service_time": ":.2f",
                    "avg_delivery_days": ":,.0f",
                    "dominant_platform": True,
                    "dominant_service_scheme": True,
                    "iso3": False,
                },
                template=plotly_template(),
                color_continuous_scale="YlGnBu",
                title=f"Zoomable Choropleth: {map_metric_labels[choropleth_metric]}",
                height=560,
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_map.update_layout(
                coloraxis_colorbar_title=map_metric_labels[choropleth_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_map, width="stretch")

    with m2:
        if bubble_data.empty:
            st.info("No values available for the selected bubble map metrics.")
        else:
            bubble_data["_bubble_size"] = bubble_data[bubble_size_metric].clip(lower=0.1)
            fig_bubble = px.scatter_geo(
                bubble_data,
                locations="iso3",
                locationmode="ISO-3",
                size="_bubble_size",
                color=bubble_color_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_service_time": ":.2f",
                    "avg_delivery_days": ":,.0f",
                    "dominant_platform": True,
                    "dominant_service_scheme": True,
                    "_bubble_size": False,
                    "iso3": False,
                },
                size_max=45,
                projection="natural earth",
                template=plotly_template(),
                color_continuous_scale="Turbo",
                title=(
                    f"Zoomable Bubble Map: size={size_metric_labels[bubble_size_metric]}, "
                    f"color={color_metric_labels[bubble_color_metric]}"
                ),
                height=560,
            )
            fig_bubble.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_bubble.update_layout(
                coloraxis_colorbar_title=color_metric_labels[bubble_color_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_bubble, width="stretch")


def render_country_lens(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across Countries")
    if orders.empty:
        st.info("No country-level rows after filtering.")
        return

    country_options = sorted(orders["country"].unique().tolist())
    default = top_n_by_mw(orders, "country", "size_mw", min(6, len(country_options)))
    selected_countries = st.multiselect("Countries to focus", options=country_options, default=default)
    if selected_countries:
        o = orders[orders["country"].isin(selected_countries)].copy()
        p = platforms[platforms["country"].isin(selected_countries)].copy()
    else:
        o = orders.copy()
        p = platforms.copy()

    if o.empty:
        st.info("No rows available for selected countries.")
        return

    render_country_maps(orders, platforms, o, p)

    c1, c2 = st.columns(2)
    with c1:
        p_valid = p[(p["platform"] != "Unknown") & (p["slot_mw"] > 0)]
        top_platforms = top_n_by_mw(p_valid, "platform", "slot_mw", 10)
        mix = (
            p_valid[p_valid["platform"].isin(top_platforms)]
            .groupby(["country", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.bar(
            mix,
            x="country",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            title="Across Country: Platform Mix",
            labels={"slot_mw": "MW", "country": ""},
            height=430,
        )
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        scheme = o.groupby(["country", "service_scheme"], as_index=False)["size_mw"].sum()
        fig2 = px.bar(
            scheme,
            x="country",
            y="size_mw",
            color="service_scheme",
            template=plotly_template(),
            title="Across Country: Service Scheme Mix",
            labels={"size_mw": "MW", "country": ""},
            height=430,
        )
        fig2.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        service_time = (
            o.dropna(subset=["service_time_years"])
            .groupby("country", as_index=False)
            .agg(avg_service_time=("service_time_years", "mean"), median_service_time=("service_time_years", "median"))
            .sort_values("avg_service_time", ascending=False)
        )
        fig3 = px.bar(
            service_time,
            x="avg_service_time",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Avg Service Time",
            labels={"avg_service_time": "Years", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        delivery = (
            o.dropna(subset=["delivery_days"])
            .groupby("country", as_index=False)
            .agg(avg_delivery_days=("delivery_days", "mean"), median_delivery_days=("delivery_days", "median"), orders=("order_id", "nunique"))
            .sort_values("avg_delivery_days", ascending=False)
        )
        fig4 = px.bar(
            delivery,
            x="avg_delivery_days",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Delivery Time",
            labels={"avg_delivery_days": "Days", "country": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")


def render_delivery_capacity(orders: pd.DataFrame) -> None:
    st.subheader("Installed Capacity and Delivery")
    if orders.empty:
        st.info("No delivery rows after filtering.")
        return

    delivery = orders.dropna(subset=["delivery_year", "size_mw"]).copy()
    if delivery.empty:
        st.info("No delivery year information available in filtered data.")
        return

    delivery["delivery_year"] = delivery["delivery_year"].astype(int)
    if "delivery_q" in delivery.columns:
        delivery["delivery_q"] = pd.to_numeric(delivery["delivery_q"], errors="coerce")

    by_year = delivery.groupby("delivery_year", as_index=False)["size_mw"].sum()
    by_quarter = (
        delivery.dropna(subset=["delivery_q"])
        .groupby(["delivery_year", "delivery_q"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["delivery_year", "delivery_q"])
    )

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(
            by_year,
            x="delivery_year",
            y="size_mw",
            template=plotly_template(),
            title="Across Years: Installed Capacity (by delivery year)",
            labels={"size_mw": "MW", "delivery_year": "Delivery year"},
            height=430,
        )
        fig1.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        if not by_quarter.empty:
            by_quarter["delivery_q"] = by_quarter["delivery_q"].astype(int).astype(str).radd("Q")
            fig2 = px.bar(
                by_quarter,
                x="delivery_year",
                y="size_mw",
                color="delivery_q",
                template=plotly_template(),
                title="Across Years: Installed Capacity by Quarter",
                labels={"size_mw": "MW", "delivery_year": "Delivery year", "delivery_q": "Quarter"},
                height=430,
            )
            fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig2, width="stretch")

    if not by_quarter.empty:
        pivot = by_quarter.pivot(index="delivery_year", columns="delivery_q", values="size_mw").fillna(0)
        fig_h = px.imshow(
            pivot,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Installed Capacity Heatmap (Year x Quarter)",
            height=420,
        )
        fig_h.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_h, width="stretch")

    delivery_time = (
        orders.dropna(subset=["delivery_days"])
        .groupby("order_year", as_index=False)
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            median_delivery_days=("delivery_days", "median"),
            p90_delivery_days=("delivery_days", lambda s: s.quantile(0.9)),
            orders=("order_id", "nunique"),
        )
        .sort_values("order_year")
    )
    if not delivery_time.empty:
        fig_t = px.line(
            delivery_time,
            x="order_year",
            y=["avg_delivery_days", "median_delivery_days", "p90_delivery_days"],
            template=plotly_template(),
            markers=True,
            title="Across Years: Delivery Time Statistics",
            labels={"value": "Days", "order_year": "Order year"},
            height=430,
        )
        fig_t.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_t, width="stretch")


def render_correlations(orders: pd.DataFrame) -> None:
    st.subheader("Additional Correlations")
    if orders.empty:
        st.info("No rows for correlation analysis.")
        return

    numeric_cols = ["size_mw", "service_time_years", "delivery_days", "platform_mw_sum", "turbines_qty_sum"]
    corr_data = orders[numeric_cols].dropna()
    if len(corr_data) >= 12:
        corr = corr_data.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Numeric Correlation Matrix",
            height=420,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    scatter_data = orders.dropna(subset=["size_mw", "delivery_days"]).copy()
    if not scatter_data.empty:
        scatter_data["service_time_years"] = pd.to_numeric(scatter_data["service_time_years"], errors="coerce")
        scatter_data["service_scheme"] = scatter_data["service_scheme"].fillna("Unknown")

        scatter_args: dict[str, Any] = {
            "data_frame": scatter_data,
            "x": "size_mw",
            "y": "delivery_days",
            "color": "service_scheme",
            "hover_data": ["country", "customer", "primary_platform", "order_year"],
            "template": plotly_template(),
            "title": "Delivery Time vs Order Size",
            "labels": {"size_mw": "Order size (MW)", "delivery_days": "Delivery days"},
            "height": 500,
        }

        if scatter_data["service_time_years"].notna().any():
            positive_vals = scatter_data.loc[scatter_data["service_time_years"] > 0, "service_time_years"]
            fallback_size = float(positive_vals.median()) if not positive_vals.empty else 1.0
            scatter_data["_bubble_size"] = scatter_data["service_time_years"].fillna(fallback_size).clip(lower=0.1)
            scatter_args["size"] = "_bubble_size"
            scatter_args["size_max"] = 38

        try:
            fig2 = px.scatter(**scatter_args)
        except ValueError:
            scatter_args.pop("size", None)
            scatter_args.pop("size_max", None)
            fig2 = px.scatter(**scatter_args)
            st.caption("Bubble size disabled for this view due missing/invalid service-time values.")
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    if len(corr_data) >= 12:
        flat_corr = (
            corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "Metric A", "level_1": "Metric B", 0: "Correlation"})
        )
        flat_corr["AbsCorr"] = flat_corr["Correlation"].abs()
        flat_corr = flat_corr.sort_values("AbsCorr", ascending=False).drop(columns=["AbsCorr"]).head(10)
        st.dataframe(flat_corr, width="stretch", hide_index=True)

def apply_global_filters(
    orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if orders.empty:
        if unannounced is None:
            unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw"])
        return orders, platforms, unannounced

    st.sidebar.header("Global Filters")
    y_min, y_max = int(orders["order_year"].min()), int(orders["order_year"].max())
    year_range = st.sidebar.slider("Order year range", y_min, y_max, (y_min, y_max))

    countries = sorted(orders["country"].dropna().unique().tolist())
    services = sorted(orders["service_scheme"].dropna().unique().tolist())
    platform_options = sorted(platforms["platform"].dropna().unique().tolist()) if not platforms.empty else []

    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=[])
    selected_services = st.sidebar.multiselect("Service schemes", options=services, default=[])
    selected_platforms = st.sidebar.multiselect("Platforms", options=platform_options, default=[])

    min_order_mw = float(np.nanmin(orders["size_mw"])) if orders["size_mw"].notna().any() else 0.0
    max_order_mw = float(np.nanmax(orders["size_mw"])) if orders["size_mw"].notna().any() else 1.0
    mw_floor = st.sidebar.slider("Minimum order MW", min_value=float(min_order_mw), max_value=float(max_order_mw), value=float(min_order_mw))

    o = orders[
        orders["order_year"].between(year_range[0], year_range[1]) & (orders["size_mw"].fillna(0) >= mw_floor)
    ].copy()

    if selected_countries:
        o = o[o["country"].isin(selected_countries)]
    if selected_services:
        o = o[o["service_scheme"].isin(selected_services)]

    p = platforms[platforms["order_id"].isin(o["order_id"])].copy()

    if selected_platforms:
        p = p[p["platform"].isin(selected_platforms)]
        o = o[o["order_id"].isin(p["order_id"])]

    if unannounced is None or unannounced.empty:
        u = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw"])
    else:
        u = unannounced[unannounced["sheet_year"].between(year_range[0], year_range[1])].copy()

    return o, p, u


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="W", layout="wide")
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    st.sidebar.toggle("Dark mode", key="dark_mode")
    dark_mode = bool(st.session_state.get("dark_mode", False))
    apply_page_style(dark_mode)

    st.title(APP_TITLE)
    st.caption("Build on public available data.")

    workbook = find_default_workbook()
    if workbook is None:
        st.error("No Excel workbook found in the current directory.")
        return

    stat = workbook.stat()
    cache_key = f"{workbook.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    economy, orders, platforms, unannounced = load_data(cache_key, str(workbook))

    if orders.empty:
        st.error("No order intake records were parsed from the `OI YYYY` sheets.")
        return

    orders_f, platforms_f, unannounced_f = apply_global_filters(orders, platforms, unannounced)
    render_sidebar_footer()

    if orders_f.empty:
        st.warning("Filters returned no rows. Adjust the sidebar filters.")
        return

    render_header_metrics(orders_f, platforms_f, unannounced_f)
    tabs = st.tabs(
        [
            "Overall Economics",
            "Year-by-Year Overview",
            "Quarterly Analytics",
            "Across Years",
            "Platform Analytics",
            "Country Analytics",
            "Delivery and Capacity",
            "Correlations",
            "Information",
        ]
    )

    with tabs[0]:
        render_overall_economics(economy)
    with tabs[1]:
        render_yearly_overview(orders_f, platforms_f, unannounced_f)
    with tabs[2]:
        render_quarterly_analytics(orders_f, unannounced_f)
    with tabs[3]:
        render_across_years(orders_f, platforms_f)
    with tabs[4]:
        render_platform_lens(platforms_f)
    with tabs[5]:
        render_country_lens(orders_f, platforms_f)
    with tabs[6]:
        render_delivery_capacity(orders_f)
    with tabs[7]:
        render_correlations(orders_f)
    with tabs[8]:
        render_information_page()


if __name__ == "__main__":
    main()
