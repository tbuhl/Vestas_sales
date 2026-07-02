from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


APP_TITLE = "Vestas Sales Intelligence Dashboard"
DATA_DIR = Path("data")
DATA_CACHE_DIR = Path("data_cache")
PARSED_DATA_FILE = DATA_CACHE_DIR / "vestas_parsed_data.pkl"
STOCK_MONTHLY_FILE = DATA_DIR / "vestas_stock_monthly.json"
MARKET_MONTHLY_FILE = DATA_DIR / "market_prices_monthly.json"
PATENTS_FALLBACK_FILE = DATA_DIR / "vestas_patents_fallback.json"

NEWS_FEEDS = [
    {
        "source": "Windpower Monthly - Vestas",
        "url": "https://www.windpowermonthly.com/rss/vestas",
        "focus": "industry",
    },
    {
        "source": "OffshoreWIND.biz - Vestas",
        "url": "https://www.offshorewind.biz/?s=vestas&feed=rss2",
        "focus": "industry",
    },
    {
        "source": "Google News (Reuters + Vestas)",
        "url": "https://news.google.com/rss/search?q=site%3Areuters.com%20Vestas%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Bloomberg + Vestas)",
        "url": "https://news.google.com/rss/search?q=site%3Abloomberg.com%20Vestas%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Recharge + Vestas)",
        "url": "https://news.google.com/rss/search?q=site%3Arechargenews.com%20Vestas%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "industry",
    },
    {
        "source": "Google News (Power Technology + Vestas)",
        "url": "https://news.google.com/rss/search?q=site%3Apower-technology.com%20Vestas%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "technical",
    },
]

VESTAS_KEYWORDS = ("vestas", "vws.co", "vwdr")
VESTAS_NEWS_BLACKLIST = (
    "your career at vestas",
    "this is vestas",
    "onshore wind turbines - vestas",
    "offshore wind turbines - vestas",
    "enventus platform - vestas",
    "from 2005-2008 - vestas",
    "stock price & latest news",
    "/en/careers",
    "/career",
    "/jobs",
    "deals of the week",
    "wind power patents",
    "weekly roundup",
)

VESTAS_TITLE_RE = re.compile(r"\b(?:vestas|vws\.co|vwdr)\b", re.IGNORECASE)
PATENTSCOPE_SEARCH_URLS = [
    (
        "https://patentscope.wipo.int/search/en/result.jsf"
        "?query=PA%3A%22Vestas%20Wind%20Systems%22&sortOption=-DP"
    ),
    (
        "https://patentscope2.wipo.int/search/en/result.jsf"
        "?query=PA%3A%22Vestas%20Wind%20Systems%22&sortOption=-DP"
    ),
]
PATENT_COLUMNS = [
    "title",
    "link",
    "publication_number",
    "publication_date",
    "filing_date",
    "priority_date",
    "inventor",
    "assignee",
    "summary",
]

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

COUNTRY_REGION_CONTINENT = {
    "Argentina": ("South America", "Americas"),
    "Australia": ("Oceania", "Oceania"),
    "Austria": ("Western Europe", "Europe"),
    "Belgium": ("Western Europe", "Europe"),
    "Bolivia": ("South America", "Americas"),
    "Brazil": ("South America", "Americas"),
    "Bulgaria": ("Eastern Europe", "Europe"),
    "Canada": ("North America", "Americas"),
    "Cape Verde": ("West Africa", "Africa"),
    "Chile": ("South America", "Americas"),
    "China": ("East Asia", "Asia"),
    "Colombia": ("South America", "Americas"),
    "Costa Rica": ("Central America", "Americas"),
    "Croatia": ("Southern Europe", "Europe"),
    "Cyprus": ("Southern Europe", "Europe"),
    "Denmark": ("Northern Europe", "Europe"),
    "Dominican Republic": ("Caribbean", "Americas"),
    "Egypt": ("North Africa", "Africa"),
    "El Salvador": ("Central America", "Americas"),
    "Estonia": ("Northern Europe", "Europe"),
    "EU": ("Europe", "Europe"),
    "Finland": ("Northern Europe", "Europe"),
    "France": ("Western Europe", "Europe"),
    "Germany": ("Western Europe", "Europe"),
    "Greece": ("Southern Europe", "Europe"),
    "Guatemala": ("Central America", "Americas"),
    "Honduras": ("Central America", "Americas"),
    "Hungary": ("Eastern Europe", "Europe"),
    "India": ("South Asia", "Asia"),
    "Ireland": ("Northern Europe", "Europe"),
    "Italy": ("Southern Europe", "Europe"),
    "Jamaica": ("Caribbean", "Americas"),
    "Japan": ("East Asia", "Asia"),
    "Jordan": ("Middle East", "Asia"),
    "Kazakhstan": ("Central Asia", "Asia"),
    "Kenya": ("East Africa", "Africa"),
    "Latvia": ("Northern Europe", "Europe"),
    "Lithuania": ("Northern Europe", "Europe"),
    "Mexico": ("North America", "Americas"),
    "Mongolia": ("East Asia", "Asia"),
    "Morocco": ("North Africa", "Africa"),
    "Netherlands": ("Western Europe", "Europe"),
    "New Zealand": ("Oceania", "Oceania"),
    "Nicaragua": ("Central America", "Americas"),
    "Norway": ("Northern Europe", "Europe"),
    "Pakistan": ("South Asia", "Asia"),
    "Panama": ("Central America", "Americas"),
    "Peru": ("South America", "Americas"),
    "Philippines": ("Southeast Asia", "Asia"),
    "Poland": ("Eastern Europe", "Europe"),
    "Portugal": ("Southern Europe", "Europe"),
    "Puerto Rico": ("Caribbean", "Americas"),
    "Romania": ("Eastern Europe", "Europe"),
    "Russia": ("Eastern Europe", "Europe"),
    "Saudi Arabia": ("Middle East", "Asia"),
    "Senegal": ("West Africa", "Africa"),
    "Serbia": ("Southern Europe", "Europe"),
    "South Africa": ("Southern Africa", "Africa"),
    "South Korea": ("East Asia", "Asia"),
    "Spain": ("Southern Europe", "Europe"),
    "Sri Lanka": ("South Asia", "Asia"),
    "Sweden": ("Northern Europe", "Europe"),
    "Taiwan": ("East Asia", "Asia"),
    "Thailand": ("Southeast Asia", "Asia"),
    "Turkey": ("Middle East", "Asia"),
    "Ukraine": ("Eastern Europe", "Europe"),
    "United Kingdom": ("Northern Europe", "Europe"),
    "United Kingdom - Offshore": ("Northern Europe", "Europe"),
    "United States": ("North America", "Americas"),
    "United States and Canada": ("North America", "Americas"),
    "Unknown": ("Unknown", "Unknown"),
    "Undisclosed": ("Unknown", "Unknown"),
    "Uruguay": ("South America", "Americas"),
    "Vietnam": ("Southeast Asia", "Asia"),
    "Wales": ("Northern Europe", "Europe"),
}

COUNTRY_REGION_CONTINENT_LOWER = {k.lower(): v for k, v in COUNTRY_REGION_CONTINENT.items()}

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
        text_color = "#f5f7fb"
        muted_color = "#a6b3c3"
        bg_main = (
            "radial-gradient(1200px 620px at 96% -12%, rgba(250, 204, 21, 0.16) 0%, rgba(250, 204, 21, 0) 52%),"
            "radial-gradient(900px 520px at -4% 18%, rgba(16, 185, 129, 0.18) 0%, rgba(16, 185, 129, 0) 48%),"
            "linear-gradient(180deg, #090d16 0%, #0d1424 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(11, 17, 31, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%)"
        header_bg = "rgba(9, 14, 26, 0.82)"
        header_border = "rgba(250, 204, 21, 0.20)"
        card_bg = "rgba(17, 24, 39, 0.62)"
        card_border = "rgba(148, 163, 184, 0.28)"
        card_shadow = "0 24px 56px rgba(0, 0, 0, 0.48), 0 8px 18px rgba(0, 0, 0, 0.25)"
        card_glow = "rgba(250, 204, 21, 0.26)"
        frame_border = "rgba(148, 163, 184, 0.25)"
        metric_label_color = "#d5deea"
        metric_value_color = "#ffffff"
        metric_delta_color = "#f8fafc"
        tab_bg = "rgba(17, 24, 39, 0.66)"
        tab_active_bg = "linear-gradient(120deg, rgba(234, 179, 8, 0.30), rgba(16, 185, 129, 0.28))"
        tab_hover_bg = "rgba(30, 41, 59, 0.80)"
        tab_border = "rgba(148, 163, 184, 0.25)"
        tab_active_border = "rgba(250, 204, 21, 0.40)"
        tab_text_color = "#d7e0ec"
        tab_active_text_color = "#ffffff"
        input_bg = "rgba(15, 23, 42, 0.78)"
        input_border = "rgba(148, 163, 184, 0.30)"
        input_focus = "rgba(250, 204, 21, 0.42)"
        accent_start = "#facc15"
        accent_end = "#10b981"
        link_color = "#a7f3d0"
        link_hover = "#fde047"
        chart_wrap_bg = "rgba(17, 24, 39, 0.56)"
        chart_wrap_border = "rgba(148, 163, 184, 0.25)"
    else:
        text_color = "#0f172a"
        muted_color = "#475569"
        bg_main = (
            "radial-gradient(1200px 620px at 94% -10%, rgba(250, 204, 21, 0.28) 0%, rgba(250, 204, 21, 0) 50%),"
            "radial-gradient(1000px 540px at -8% 20%, rgba(52, 211, 153, 0.24) 0%, rgba(52, 211, 153, 0) 46%),"
            "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(248, 250, 252, 0.94) 100%)"
        header_bg = "rgba(255, 255, 255, 0.78)"
        header_border = "rgba(15, 23, 42, 0.10)"
        card_bg = "rgba(255, 255, 255, 0.66)"
        card_border = "rgba(15, 23, 42, 0.10)"
        card_shadow = "0 26px 52px rgba(2, 6, 23, 0.12), 0 8px 18px rgba(2, 6, 23, 0.08)"
        card_glow = "rgba(16, 185, 129, 0.20)"
        frame_border = "rgba(15, 23, 42, 0.11)"
        metric_label_color = "#334155"
        metric_value_color = "#0f172a"
        metric_delta_color = "#0f172a"
        tab_bg = "rgba(255, 255, 255, 0.72)"
        tab_active_bg = "linear-gradient(120deg, rgba(254, 240, 138, 0.74), rgba(167, 243, 208, 0.72))"
        tab_hover_bg = "rgba(255, 255, 255, 0.90)"
        tab_border = "rgba(15, 23, 42, 0.12)"
        tab_active_border = "rgba(16, 185, 129, 0.44)"
        tab_text_color = "#334155"
        tab_active_text_color = "#0f172a"
        input_bg = "rgba(255, 255, 255, 0.80)"
        input_border = "rgba(15, 23, 42, 0.14)"
        input_focus = "rgba(16, 185, 129, 0.36)"
        accent_start = "#eab308"
        accent_end = "#10b981"
        link_color = "#0f766e"
        link_hover = "#a16207"
        chart_wrap_bg = "rgba(255, 255, 255, 0.58)"
        chart_wrap_border = "rgba(15, 23, 42, 0.11)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

html, body, .stApp {{
  font-family: 'Nunito Sans', sans-serif;
  color: {text_color};
}}

.stApp {{
  background: {bg_main};
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

h1, h2, h3, h4, h5, h6 {{
  letter-spacing: -0.01em;
  font-weight: 700;
}}

h1 {{
  font-weight: 800;
  background-image: linear-gradient(110deg, {accent_start} 0%, {accent_end} 56%, {text_color} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}

p, span, label, li {{
  color: {text_color};
}}

a {{
  color: {link_color};
  text-decoration-color: {link_color};
}}

a:hover {{
  color: {link_hover};
  text-decoration-color: {link_hover};
}}

section[data-testid="stSidebar"] {{
  background: {bg_sidebar};
  border-right: 1px solid {frame_border};
  backdrop-filter: blur(9px);
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
  backdrop-filter: blur(12px);
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
  background: {input_bg};
  border: 1px solid {input_border};
  color: {text_color};
  border-radius: 10px;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {{
  border-color: {input_focus} !important;
  box-shadow: 0 0 0 2px {input_focus} !important;
}}

div[data-testid="stAlert"],
div[data-testid="stDataFrame"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  padding: 0.45rem 0.82rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {{
  color: {metric_label_color} !important;
  font-weight: 700 !important;
}}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
  color: {metric_value_color} !important;
  letter-spacing: -0.01em;
}}

div[data-testid="stMetricDelta"] {{
  color: {metric_delta_color} !important;
  font-size: 0.72rem !important;
  line-height: 1.05 !important;
}}

div[data-testid="stPlotlyChart"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {chart_wrap_bg};
  border: 1px solid {chart_wrap_border};
  border-radius: 16px;
  padding: 0.35rem 0.45rem 0.2rem 0.45rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem;
  padding-bottom: 0.1rem;
}}

.stTabs [data-baseweb="tab"] {{
  background: {tab_bg};
  border: 1px solid {tab_border};
  color: {tab_text_color} !important;
  border-radius: 11px 11px 0 0;
  padding-top: 0.42rem;
  padding-bottom: 0.48rem;
  transition: all 0.18s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
  background: {tab_hover_bg};
  border-color: {tab_active_border};
  transform: translateY(-1px);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: {tab_active_bg} !important;
  color: {tab_active_text_color} !important;
  border-color: {tab_active_border} !important;
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.18);
}}

div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] {{
  border: 1px solid {frame_border};
  border-radius: 14px;
}}

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {{
  color: {muted_color} !important;
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
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 10px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
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


def country_region_continent(country: Any) -> tuple[str, str]:
    text = clean_text(country)
    if text is None:
        return "Unknown", "Unknown"
    mapped = COUNTRY_REGION_CONTINENT.get(text)
    if mapped is not None:
        return mapped
    return COUNTRY_REGION_CONTINENT_LOWER.get(text.lower(), ("Unknown", "Unknown"))


def apply_region_continent_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return df

    out = df.copy()
    rc = out["country"].apply(country_region_continent)
    out["region"] = rc.apply(lambda t: t[0])
    out["continent"] = rc.apply(lambda t: t[1])
    return out


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


def parse_oi_sheet_year(sheet_name: Any) -> int | None:
    text = clean_text(sheet_name)
    if text is None:
        return None
    m = re.fullmatch(r"OI[\s_-]*(\d{4})", text, flags=re.IGNORECASE)
    if m is None:
        return None
    year = int(m.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def normalize_sheet_key(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_sheet_name(sheet_names: list[str], target_name: str) -> str | None:
    target_clean = clean_text(target_name)
    if target_clean is None:
        return None

    for name in sheet_names:
        if clean_text(name) == target_clean:
            return str(name)

    target_casefold = target_clean.casefold()
    for name in sheet_names:
        name_clean = clean_text(name)
        if name_clean is not None and name_clean.casefold() == target_casefold:
            return str(name)

    target_key = normalize_sheet_key(target_clean)
    for name in sheet_names:
        if normalize_sheet_key(name) == target_key:
            return str(name)

    return None


def parse_economy_sheet(workbook: Path) -> pd.DataFrame:
    try:
        sheet_names = pd.ExcelFile(workbook).sheet_names
    except Exception as exc:
        raise ValueError(f"Could not open workbook '{workbook.name}': {exc}") from exc

    economy_sheet = resolve_sheet_name(sheet_names, "Vestas Economy")
    if economy_sheet is None:
        available = ", ".join(sheet_names[:12]) if sheet_names else "(none)"
        raise ValueError(
            f"Workbook '{workbook.name}' does not contain a 'Vestas Economy' sheet. "
            f"Available sheets: {available}"
        )

    economy = pd.read_excel(workbook, sheet_name=economy_sheet)
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
    oi_sheets = [
        (s, parse_oi_sheet_year(s))
        for s in xl.sheet_names
    ]
    oi_sheets = [(s, y) for s, y in oi_sheets if y is not None]
    oi_sheets = sorted(oi_sheets, key=lambda pair: pair[1])

    order_rows: list[dict[str, Any]] = []
    platform_rows: list[dict[str, Any]] = []
    unannounced_rows: list[dict[str, Any]] = []

    for sheet, year in oi_sheets:
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
            region, continent = country_region_continent(country)
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
                "region": region,
                "continent": continent,
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
                        "region": region,
                        "continent": continent,
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
                            "region": region,
                            "continent": continent,
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
    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)
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
    app_dir = Path(__file__).resolve().parent
    parent_dir = app_dir.parent
    search_roots = [Path("."), app_dir, parent_dir]
    preferred_names = [
        "Vestas_economical_data_start_2026.xlsx",
        "Vestas_data_new.xlsx",
        "vestas_data_new.xlsx",
    ]

    for root in search_roots:
        for name in preferred_names:
            workbook = root / name
            if not workbook.exists() or workbook.name.startswith("~$"):
                continue
            try:
                sheet_names = pd.ExcelFile(workbook).sheet_names
            except Exception:
                continue
            if resolve_sheet_name(sheet_names, "Vestas Economy") is not None:
                return workbook

    patterns = ("*.xlsx", "*.xlsm", "*.xls")
    files: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        for pattern in patterns:
            for file_path in sorted(root.glob(pattern)):
                if file_path.name.startswith("~$"):
                    continue
                key = str(file_path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                files.append(file_path)
    if not files:
        return None

    def file_priority(path: Path) -> tuple[int, float, str]:
        name = path.name.lower()
        if "vestas" in name:
            rank = 0
        elif any(token in name for token in ("nordex", "suzlon", "sgre", "siemens")):
            rank = 2
        else:
            rank = 1
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (rank, -mtime, name)

    for workbook in sorted(files, key=file_priority):
        try:
            sheet_names = pd.ExcelFile(workbook).sheet_names
        except Exception:
            continue
        if resolve_sheet_name(sheet_names, "Vestas Economy") is not None:
            return workbook

    return None


def empty_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def parse_stock_monthly_json(stock_path: Path) -> pd.DataFrame:
    if not stock_path.exists():
        return empty_stock_frame()

    try:
        raw = pd.read_json(stock_path)
    except Exception:
        return empty_stock_frame()

    if raw.empty:
        return empty_stock_frame()

    rename_map: dict[str, str] = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        if "date" in key:
            rename_map[col] = "date"
        elif "adj" in key and "close" in key:
            rename_map[col] = "adj_close"
        elif "open" in key:
            rename_map[col] = "open"
        elif "high" in key:
            rename_map[col] = "high"
        elif "low" in key:
            rename_map[col] = "low"
        elif "close" in key:
            rename_map[col] = "close"
        elif "volume" in key:
            rename_map[col] = "volume"

    stock = raw.rename(columns=rename_map).copy()
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required_cols:
        if col not in stock.columns:
            stock[col] = np.nan
    stock = stock[required_cols].copy()

    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    stock = stock.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    stock = stock.sort_values("date")
    stock["date"] = stock["date"].dt.normalize()
    stock["volume"] = stock["volume"].fillna(0.0)
    return stock


def parse_market_monthly_json(market_path: Path) -> pd.DataFrame:
    if not market_path.exists():
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    try:
        raw = pd.read_json(market_path)
    except Exception:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    if raw.empty:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    market = raw.copy()
    if "date" not in market.columns:
        for c in market.columns:
            if "date" in str(c).lower():
                market = market.rename(columns={c: "date"})
                break
    if "date" not in market.columns:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    for col in ["steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]:
        if col not in market.columns:
            market[col] = np.nan
        market[col] = pd.to_numeric(market[col], errors="coerce")

    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["date"]).sort_values("date")
    return market[["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]].copy()


@st.cache_data(show_spinner=False)
def load_data(cache_key: str, workbook_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_file = PARSED_DATA_FILE
    if cache_file.exists():
        try:
            payload = pd.read_pickle(cache_file)
            if isinstance(payload, dict) and payload.get("cache_key") == cache_key:
                return (
                    payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
                    payload.get("orders", pd.DataFrame()),
                    payload.get("platforms", pd.DataFrame()),
                    payload.get("unannounced", pd.DataFrame()),
                )
        except Exception:
            pass

    workbook = Path(workbook_path)
    economy = parse_economy_sheet(workbook)
    orders, platforms, unannounced = parse_oi_sheets(workbook)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "workbook_path": str(workbook.resolve()),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "economy": economy,
            "orders": orders,
            "platforms": platforms,
            "unannounced": unannounced,
        }
        pd.to_pickle(payload, cache_file)
    except Exception:
        pass

    return economy, orders, platforms, unannounced


@st.cache_data(show_spinner=False)
def load_stock_data(cache_key: str, stock_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_stock_monthly_json(Path(stock_path))


@st.cache_data(show_spinner=False)
def load_market_data(cache_key: str, market_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_market_monthly_json(Path(market_path))


def load_data_from_cache_only() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not PARSED_DATA_FILE.exists():
        return None
    try:
        payload = pd.read_pickle(PARSED_DATA_FILE)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return (
        payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
        payload.get("orders", pd.DataFrame()),
        payload.get("platforms", pd.DataFrame()),
        payload.get("unannounced", pd.DataFrame()),
    )


def parse_news_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if text is None:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return pd.Timestamp(dt).tz_convert("UTC").tz_localize(None) if dt.tzinfo else pd.Timestamp(dt)
    except Exception:
        pass
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)


def normalize_news_title(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_news_summary(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def short_summary(value: Any, max_chars: int = 320) -> str | None:
    text = normalize_news_summary(value)
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[:max_chars].strip()
    return f"{clipped}..."


def is_vestas_relevant(
    title: str | None,
    summary: str | None,
    link: str | None,
    source_name: str | None = None,
) -> bool:
    source_txt = (source_name or "").lower()
    full = " ".join([title or "", summary or "", link or ""]).lower()

    if any(bad in full for bad in VESTAS_NEWS_BLACKLIST):
        return False

    # Official Vestas channels are already company-specific.
    if source_txt.startswith("vestas (official"):
        return True

    title_txt = title or ""
    if not VESTAS_TITLE_RE.search(title_txt):
        return False

    # Filter broad roundup headlines where Vestas is one of many names.
    title_low = title_txt.lower()
    if title_low.count("|") >= 2 and not title_low.startswith("vestas"):
        return False

    return any(k in full for k in VESTAS_KEYWORDS)


def vestas_news_focus(news_type: Any) -> str:
    text = (clean_text(news_type) or "").lower()
    if "announcement" in text or "investor" in text:
        return "financial"
    if "press" in text:
        return "company"
    return "company"


def parse_vestas_company_news_archive() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    url = "https://www.vestas.com/en/media/company-news/_jcr_content/root/container/container/newsarchive.newsarchive.json"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return rows

    news_items = payload.get("news")
    if not isinstance(news_items, list) or not news_items:
        return rows

    for item in news_items:
        title = normalize_news_title(item.get("title"))
        path = clean_text(item.get("path"))
        if title is None or path is None:
            continue
        link = path if path.startswith("http") else f"https://www.vestas.com{path}"
        summary = normalize_news_summary(item.get("type"))
        published = parse_news_datetime(item.get("publishDate"))
        key = (title.strip().lower(), link.strip().lower())
        if key in seen:
            continue
        if not is_vestas_relevant(title, summary, link, source_name="Vestas (Official News)"):
            continue
        seen.add(key)
        rows.append(
            {
                "source": "Vestas (Official News)",
                "focus": vestas_news_focus(item.get("type")),
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )
    return rows


def parse_rss_feed(feed_url: str, source_name: str, focus: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        resp = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return rows

    items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for node in items:
        title = normalize_news_title(node.findtext("title"))
        link = clean_text(node.findtext("link"))
        summary = normalize_news_summary(node.findtext("description"))
        pub_raw = (
            node.findtext("pubDate")
            or node.findtext("{http://purl.org/dc/elements/1.1/}date")
            or node.findtext("date")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_vestas_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )

    for node in atom_entries:
        title = normalize_news_title(node.findtext("{http://www.w3.org/2005/Atom}title"))
        link = None
        for link_node in node.findall("{http://www.w3.org/2005/Atom}link"):
            href = clean_text(link_node.attrib.get("href"))
            rel = clean_text(link_node.attrib.get("rel"))
            if href is None:
                continue
            if rel in {None, "alternate"}:
                link = href
                break
            if link is None:
                link = href

        pub_raw = (
            node.findtext("{http://www.w3.org/2005/Atom}updated")
            or node.findtext("{http://www.w3.org/2005/Atom}published")
        )
        summary = normalize_news_summary(
            node.findtext("{http://www.w3.org/2005/Atom}summary")
            or node.findtext("{http://www.w3.org/2005/Atom}content")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_vestas_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )
    return rows


def parse_vestas_orders_page() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    url = "https://www.vestas.com/en/investor/announcements/wind-turbines-orders"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return rows

    pattern = re.compile(
        r"<tr[^>]*>\s*<td[^>]*>\s*(\d{2}-\d{2}-\d{2})\s*</td>\s*"
        r"<td[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>\s*([^<]+?)\s*</a>",
        flags=re.IGNORECASE,
    )
    for d_str, link, title in pattern.findall(page):
        title_n = normalize_news_title(title)
        link_n = clean_text(link)
        published = pd.to_datetime(d_str, format="%d-%m-%y", errors="coerce")
        if title_n is None or link_n is None:
            continue
        if not is_vestas_relevant(title_n, None, link_n, source_name="Vestas (Official Orders)"):
            continue
        rows.append(
            {
                "source": "Vestas (Official Orders)",
                "focus": "company",
                "title": title_n,
                "link": link_n,
                "summary": None,
                "published": published,
            }
        )
    return rows


@st.cache_data(show_spinner=False, ttl=1800)
def load_latest_news() -> pd.DataFrame:
    rows: list[dict[str, Any]] = parse_vestas_company_news_archive()
    rows.extend(parse_vestas_orders_page())
    for feed in NEWS_FEEDS:
        rows.extend(parse_rss_feed(feed["url"], feed["source"], feed["focus"]))

    if not rows:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])

    news = pd.DataFrame(rows)
    news = news[
        news.apply(
            lambda r: is_vestas_relevant(
                clean_text(r.get("title")),
                clean_text(r.get("summary")),
                clean_text(r.get("link")),
                clean_text(r.get("source")),
            ),
            axis=1,
        )
    ].copy()
    if news.empty:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["title_key"] = news["title"].str.strip().str.lower()
    news["link_key"] = news["link"].str.strip().str.lower()
    news = news.sort_values(["published", "title"], ascending=[False, True])
    news = news.drop_duplicates(subset=["link_key"], keep="first")
    news = news.drop_duplicates(subset=["title_key"], keep="first")
    news = news.drop(columns=["title_key", "link_key"])

    max_per_source = 8
    selected_idx: list[int] = []
    source_counts: dict[str, int] = {}
    for idx, row in news.iterrows():
        src = str(row.get("source", "Unknown"))
        count = source_counts.get(src, 0)
        if count >= max_per_source:
            continue
        selected_idx.append(idx)
        source_counts[src] = count + 1
        if len(selected_idx) >= 20:
            break

    if len(selected_idx) < 20:
        for idx in news.index:
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= 20:
                break

    return news.loc[selected_idx].copy()


def empty_patents_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PATENT_COLUMNS)


def normalize_patents_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_patents_frame()

    out = frame.copy()
    for col in PATENT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PATENT_COLUMNS].copy()
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce")
    out["priority_date"] = pd.to_datetime(out["priority_date"], errors="coerce")
    out = out.drop_duplicates(subset=["publication_number", "title"], keep="first")
    out = out.sort_values(["publication_date", "title"], ascending=[False, True]).head(10)
    return out


def load_patents_fallback() -> pd.DataFrame:
    if not PATENTS_FALLBACK_FILE.exists():
        return empty_patents_frame()
    try:
        payload = json.loads(PATENTS_FALLBACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return empty_patents_frame()
    if isinstance(payload, dict):
        records = payload.get("records", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    if not isinstance(records, list) or not records:
        return empty_patents_frame()
    return normalize_patents_frame(pd.DataFrame(records))


def extract_wipo_patents(page: str) -> pd.DataFrame:
    def html_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"<[^>]+>", " ", value)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    row_blocks = re.findall(
        r"<tr[^>]*data-ri=\"\d+\"[^>]*>.*?</tr>",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not row_blocks:
        row_blocks = re.findall(
            r"<div[^>]*class=\"[^\"]*ps-patent-result[^\"]*\"[\s\S]*?</div>\s*</div>",
            page,
            flags=re.IGNORECASE,
        )

    rows: list[dict[str, Any]] = []
    for row_html in row_blocks:
        pub_no_match = re.search(
            r"ps-patent-result--title--patent-number[^>]*>(.*?)</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title_match = re.search(
            r"ps-patent-result--title--title[^>]*>(.*?)</span>\s*</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if title_match is None:
            title_match = re.search(
                r"ps-patent-result--title--title[^>]*>(.*?)</span>",
                row_html,
                flags=re.IGNORECASE | re.DOTALL,
            )
        pub_date_match = re.search(
            r"resultListTableColumnPubDate\"[^>]*>(.*?)</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        link_match = re.search(
            r"<a[^>]*href=\"([^\"]*detail\.jsf[^\"]+)\"",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        applicant_match = re.search(
            r"ps-patent-result--applicant[^>]*>(.*?)</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        inventor_match = re.search(
            r"ps-patent-result--inventor[^>]*>(.*?)</span>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        abstract_match = re.search(
            r"ps-patent-result--abstract[^>]*>(.*?)</div>",
            row_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        publication_number = html_text(pub_no_match.group(1)) if pub_no_match else None
        title = normalize_news_title(html_text(title_match.group(1)) if title_match else None)
        applicant = html_text(applicant_match.group(1)) if applicant_match else None
        inventor = html_text(inventor_match.group(1)) if inventor_match else None
        summary = normalize_news_summary(html_text(abstract_match.group(1)) if abstract_match else None)
        pub_date_raw = html_text(pub_date_match.group(1)) if pub_date_match else None
        publication_date = pd.to_datetime(pub_date_raw, format="%d.%m.%Y", errors="coerce")
        if pd.isna(publication_date):
            publication_date = pd.to_datetime(pub_date_raw, errors="coerce")

        if title is None or publication_number is None:
            continue
        if applicant is not None and "vestas" not in applicant.lower():
            continue

        link = None
        if link_match:
            href = html.unescape(link_match.group(1))
            href = re.sub(r";jsessionid=[^?]+", "", href)
            link = href if href.startswith("http") else f"https://patentscope.wipo.int/search/en/{href.lstrip('/')}"

        rows.append(
            {
                "title": title,
                "link": link,
                "publication_number": publication_number,
                "publication_date": publication_date,
                "filing_date": pd.NaT,
                "priority_date": pd.NaT,
                "inventor": inventor,
                "assignee": applicant,
                "summary": summary,
            }
        )

    if not rows:
        return empty_patents_frame()
    return normalize_patents_frame(pd.DataFrame(rows))


def fetch_wipo_patents() -> pd.DataFrame:
    for url in PATENTSCOPE_SEARCH_URLS:
        try:
            resp = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            patents = extract_wipo_patents(resp.text)
            if not patents.empty:
                return patents
        except Exception:
            continue
    return empty_patents_frame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_latest_patents() -> pd.DataFrame:
    live = fetch_wipo_patents()
    if not live.empty:
        return live
    return load_patents_fallback()


def latest_data_handled_text(
    economy: pd.DataFrame,
    orders: pd.DataFrame,
    unannounced: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
) -> str:
    economy_max_year = int(economy["year"].max()) if economy is not None and not economy.empty else None
    orders_max_date = pd.to_datetime(orders["order_date"], errors="coerce").max() if orders is not None and not orders.empty else pd.NaT
    orders_max_year = int(orders["sheet_year"].max()) if orders is not None and not orders.empty else None
    stock_max = pd.to_datetime(stock_monthly["date"], errors="coerce").max() if stock_monthly is not None and not stock_monthly.empty else pd.NaT
    market_max = pd.to_datetime(market_monthly["date"], errors="coerce").max() if market_monthly is not None and not market_monthly.empty else pd.NaT
    unann_max_year = int(unannounced["sheet_year"].max()) if unannounced is not None and not unannounced.empty else None

    orders_txt = "-"
    if pd.notna(orders_max_date):
        orders_txt = orders_max_date.strftime("%Y-%m-%d")
    elif orders_max_year is not None:
        orders_txt = str(orders_max_year)

    stock_txt = stock_max.strftime("%Y-%m-%d") if pd.notna(stock_max) else "-"
    market_txt = market_max.strftime("%Y-%m-%d") if pd.notna(market_max) else "-"
    econ_txt = str(economy_max_year) if economy_max_year is not None else "-"
    unann_txt = str(unann_max_year) if unann_max_year is not None else "-"

    return (
        f"Latest handled data -> Economy: {econ_txt}; Orders: {orders_txt}; "
        f"Unannounced: {unann_txt}; Stock: {stock_txt}; Steel/Copper: {market_txt}. "
        "Not a live feed; updates occur when source files/feeds are refreshed."
    )


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
    delivery_base = orders.dropna(subset=["delivery_days"]).copy()
    delivery_base_orders = delivery_base["order_id"].nunique()
    delivery_base_mw = delivery_base["size_mw"].sum()

    turbines_est = pd.to_numeric(platforms["turbines_qty"], errors="coerce").sum()

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Orders", int_fmt(total_orders))
    m2.metric(
        "Ordered MW",
        mw_fmt(total_mw),
        delta=f"hereof unannounced {unannounced_mw:,.0f} MW",
        delta_color="off",
    )
    m3.metric("Platform-Mapped MW", mw_fmt(total_platform_mw))
    m4.metric(
        "Turbines (est.)",
        int_fmt(turbines_est) if pd.notna(turbines_est) and turbines_est > 0 else "-",
        delta="from platform-mapped rows",
        delta_color="off",
    )
    m5.metric(
        "Avg Order Size",
        mw_fmt(avg_order_mw),
        delta=f"based on {announced_mw / 1000.0:,.1f} GW announced",
        delta_color="off",
    )
    m6.metric(
        "Avg Delivery Time",
        days_fmt(avg_delivery_days),
        delta=f"based on {delivery_base_mw / 1000.0:,.1f} GW / {delivery_base_orders:,.0f} orders",
        delta_color="off",
    )
    m7.metric("Countries", int_fmt(countries))


def render_information_page() -> None:
    st.subheader("Information")
    st.markdown("**Data source**")
    st.write("This dashboard is build on public available data.")

    st.markdown("**How to use and navigate**")
    st.write(
        "Use the sidebar filters to narrow years, countries, service schemes, platforms, and minimum order size."
    )
    st.write(
        "Navigate tabs from high-level trends (Overall/Yearly/Quarterly) to deep dives (Across Years, Platform, "
        "Turbine Explorer, Country, Customer Intelligence, Sankey, Delivery, Correlations)."
    )
    st.write(
        "The Turbine Explorer holds a full catalog of every turbine variant in the order book - specs, swept area, "
        "specific power, sales lifecycle, and per-model commercial detail."
    )

    st.markdown("**Disclaimer**")
    st.write(
        "This dashboard is provided for informational purposes only. Data is compiled from public sources and may contain gaps, delays, or inaccuracies."
    )
    st.write(
        "Always verify critical figures with official company reporting before using the information for financial, legal, strategic, or investment decisions."
    )
    st.write(
        'The data and dashboard are provided "as is" without warranties of any kind. I make no guarantees of accuracy, completeness, or fitness for a particular purpose.'
    )
    st.write(
        "Use at your own risk. I am not responsible or liable for any losses or damages arising from use of the data or dashboard."
    )

    st.markdown("**Sources and attribution**")
    st.write(
        "All figures are compiled manually from Vestas public investor communications (press releases/announcements) and annual reports."
    )
    st.write("Each datapoint should be verified against the original source documents.")

    st.markdown("**No affiliation / no endorsement**")
    st.write(
        "This is an unofficial, independent dashboard and is not affiliated with, endorsed by, or sponsored by Vestas."
    )

    st.markdown("**IP and rights**")
    st.write(
        "The dataset in this repository is my own compilation (selection/structure) based on publicly available sources."
    )
    st.write("Underlying source documents remain the property of their respective owners.")

    st.markdown("**About me**")
    st.write(
        "Thomas Buhl is a wind-energy engineering leader with 20+ years across research and industry, including professor/department-head and director/VP roles."
    )
    st.write(
        "He combines deep technical expertise in wind turbine design and optimization with international people leadership and strategy execution."
    )


def render_latest_news() -> None:
    st.subheader("Latest News")
    st.caption("Showing the 20 latest items directly related to Vestas.")
    st.caption("Sources include official Vestas news plus selected financial and wind-industry feeds.")
    st.caption("Use links to open the original source articles.")

    news = load_latest_news()
    if news.empty:
        st.info("No news items could be loaded right now. Try again later.")
        return

    news = news.copy()
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["published_label"] = news["published"].dt.strftime("%Y-%m-%d %H:%M UTC")
    news["published_label"] = news["published_label"].fillna("Unknown date")

    for idx, row in enumerate(news.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled"
        link = clean_text(row.link) or ""
        source = clean_text(row.source) or "Unknown source"
        focus = clean_text(row.focus) or "news"
        when = clean_text(row.published_label) or "Unknown date"
        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        summary = short_summary(row.summary, max_chars=360)
        if summary is None or len(summary) < 80:
            summary = (
                f"This is a Vestas-related {focus} update from {source}. "
                "The source feed did not provide a detailed excerpt, so open the original link for full context."
            )
        st.write(summary)
        st.caption(f"{source} | {focus} | {when}")


def render_latest_patents() -> None:
    st.subheader("Latest Patents")
    st.caption("Showing 10 most recent Vestas patent publications from WIPO PATENTSCOPE (with local fallback snapshot).")
    st.caption("Inventor names are listed from filings; employment is inferred from Vestas applicant records.")

    patents = load_latest_patents()
    if patents.empty:
        st.info("No patent records could be loaded right now. Try again later.")
        return

    view = patents.copy()
    view["publication_label"] = pd.to_datetime(view["publication_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["filing_label"] = pd.to_datetime(view["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["priority_label"] = pd.to_datetime(view["priority_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["publication_label"] = view["publication_label"].fillna("-")
    view["filing_label"] = view["filing_label"].fillna("-")
    view["priority_label"] = view["priority_label"].fillna("-")

    for idx, row in enumerate(view.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled patent record"
        link = clean_text(row.link) or ""
        publication_number = clean_text(row.publication_number) or "Unknown publication number"
        inventor = clean_text(row.inventor) or "Unknown inventor"
        assignee = clean_text(row.assignee) or "Unknown assignee"
        summary = short_summary(row.summary, max_chars=360)
        if summary is None:
            summary = "No abstract snippet available in the source response."

        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        st.write(summary)
        st.caption(
            f"{publication_number} | Published: {row.publication_label} | Filed: {row.filing_label} | Priority: {row.priority_label}"
        )
        st.caption(f"Inventor(s): {inventor} | Assignee: {assignee}")


def render_overall_economics(economy: pd.DataFrame, stock_monthly: pd.DataFrame, market_monthly: pd.DataFrame) -> None:
    st.subheader("Overall Economics")
    if economy.empty:
        st.info("No parsable data found in `Vestas Economy`.")
        return

    years = pd.to_numeric(economy["year"], errors="coerce").dropna().astype(int).tolist()
    y_min = int(min(years))
    max_year_candidates = [int(max(years))]
    for date_frame in (stock_monthly, market_monthly):
        if date_frame is not None and not date_frame.empty and "date" in date_frame.columns:
            date_years = pd.to_datetime(date_frame["date"], errors="coerce").dt.year.dropna()
            if not date_years.empty:
                max_year_candidates.append(int(date_years.max()))
    y_max = max(max_year_candidates)
    if y_min == y_max:
        year_range = (y_min, y_max)
        st.caption(f"Economy year range: {y_min}")
    else:
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

    ebit_metric = find_metric(["ebit"])
    service_sales_metric = find_metric(["salesofservice"]) or find_metric(["sales", "service"])
    employees_metric = find_metric(["average", "employees"]) or find_metric(["eoy", "employees"])
    pivot = econ.pivot_table(index="year", columns="metric", values="value", aggfunc="mean")

    def ratio_series(numerator: str | None, denominator: str | None, scale: float = 1.0) -> pd.Series | None:
        if numerator is None or denominator is None:
            return None
        if numerator not in pivot.columns or denominator not in pivot.columns:
            return None
        series = (pivot[numerator] / pivot[denominator]) * scale
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        return series if not series.empty else None

    ebit_margin = ratio_series(ebit_metric, revenue_metric, 100.0)
    gross_margin = ratio_series(gross_profit_metric, revenue_metric, 100.0)
    service_share = ratio_series(service_sales_metric, revenue_metric, 100.0)
    book_to_bill = ratio_series(order_intake_mw_metric, deliveries_mw_metric)
    rev_per_employee = ratio_series(revenue_metric, employees_metric, 1000.0)

    if any(s is not None for s in (ebit_margin, gross_margin, service_share, book_to_bill, rev_per_employee)):
        st.markdown("**Profitability and Order-Quality Ratios (derived)**")
        r1, r2 = st.columns(2)
        with r1:
            fig_margins = go.Figure()
            for series, name in (
                (gross_margin, "Gross margin (%)"),
                (ebit_margin, "EBIT margin (%)"),
                (service_share, "Service share of revenue (%)"),
            ):
                if series is None:
                    continue
                fig_margins.add_trace(
                    go.Scatter(x=series.index, y=series.values, mode="lines+markers", name=name, line=dict(width=3))
                )
            if fig_margins.data:
                fig_margins.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.6)")
                fig_margins.update_layout(
                    template=plotly_template(),
                    title="Margin Development",
                    yaxis_title="% of revenue",
                    height=420,
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                st.plotly_chart(fig_margins, width="stretch")
            else:
                st.info("Margin inputs (EBIT, gross profit, revenue) not found in the economy sheet.")

        with r2:
            fig_ratio = go.Figure()
            if book_to_bill is not None:
                fig_ratio.add_trace(
                    go.Scatter(
                        x=book_to_bill.index,
                        y=book_to_bill.values,
                        mode="lines+markers",
                        name="Book-to-bill (MW intake / MW delivered)",
                        line=dict(width=3),
                    )
                )
                fig_ratio.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="rgba(209,73,91,0.8)",
                    annotation_text="Backlog neutral (1.0)",
                    annotation_position="bottom right",
                )
            if rev_per_employee is not None:
                fig_ratio.add_trace(
                    go.Scatter(
                        x=rev_per_employee.index,
                        y=rev_per_employee.values,
                        mode="lines+markers",
                        name="Revenue per employee (kEUR)",
                        yaxis="y2",
                        line=dict(width=3, dash="dot"),
                    )
                )
            if fig_ratio.data:
                fig_ratio.update_layout(
                    template=plotly_template(),
                    title="Book-to-Bill and Productivity",
                    yaxis_title="Book-to-bill ratio",
                    yaxis2=dict(title="kEUR / employee", overlaying="y", side="right", showgrid=False),
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=10, r=10, t=80, b=10),
                )
                st.plotly_chart(fig_ratio, width="stretch")
            else:
                st.info("Book-to-bill inputs (order intake MW, deliveries MW) not found in the economy sheet.")
        st.caption(
            "Derived from the economy sheet: book-to-bill above 1.0 means the order backlog is growing; "
            "the gap between gross and EBIT margin shows fixed-cost and one-off pressure."
        )

    st.markdown("**Vestas Stock Price (Monthly OHLC)**")
    if stock_monthly is None or stock_monthly.empty:
        st.info("No local stock file found. Expected: `data/vestas_stock_monthly.json`.")
    else:
        stock = stock_monthly.copy()
        stock["year"] = stock["date"].dt.year
        stock = stock[stock["year"].between(year_range[0], year_range[1])].copy()
        if stock.empty:
            st.info("No stock data available for the selected economy year range.")
        else:
            overlay_options = [f"Economy: {m}" for m in sorted(metric_options)]
            market_series = {}
            steel_label = "Market: Steel price (HRC futures, USD/short ton)"
            copper_label = "Market: Copper price (HG futures, USD/lb)"
            if market_monthly is not None and not market_monthly.empty:
                market_series = {
                    steel_label: "steel_hrc_usd_per_short_ton",
                    copper_label: "copper_hg_usd_per_lb",
                }
                for label, col in list(market_series.items()):
                    if col not in market_monthly.columns or market_monthly[col].dropna().empty:
                        market_series.pop(label, None)
                overlay_options.extend(list(market_series.keys()))

            overlay_default: list[str] = []
            if "Economy: revenue (mEUR)" in overlay_options:
                overlay_default.append("Economy: revenue (mEUR)")
            if steel_label in overlay_options:
                overlay_default.append(steel_label)

            selected_overlays = st.multiselect(
                "Overlay on right axis (Y2/Y3)",
                options=overlay_options,
                default=overlay_default,
                key="stock_overlay_series",
            )

            fig_stock = go.Figure(
                data=[
                    go.Candlestick(
                        x=stock["date"],
                        open=stock["open"],
                        high=stock["high"],
                        low=stock["low"],
                        close=stock["close"],
                        name="VWS.CO",
                    )
                ]
            )
            y2_used = False
            y3_used = False
            for item in selected_overlays:
                if item.startswith("Economy: "):
                    metric_name = item.replace("Economy: ", "", 1)
                    metric_series = econ[econ["metric"] == metric_name][["year", "value"]].copy()
                    if metric_series.empty:
                        continue
                    metric_series["date"] = pd.to_datetime(metric_series["year"].astype(str) + "-12-31", errors="coerce")
                    metric_series = metric_series.dropna(subset=["date", "value"])
                    metric_series = metric_series[metric_series["date"].dt.year.between(year_range[0], year_range[1])]
                    if metric_series.empty:
                        continue
                    fig_stock.add_trace(
                        go.Scatter(
                            x=metric_series["date"],
                            y=metric_series["value"],
                            mode="lines+markers",
                            name=metric_name,
                            yaxis="y2",
                        )
                    )
                    y2_used = True
                elif item in market_series:
                    col = market_series[item]
                    market_f = market_monthly.copy()
                    market_f["date"] = pd.to_datetime(market_f["date"], errors="coerce")
                    market_f[col] = pd.to_numeric(market_f[col], errors="coerce")
                    market_f = market_f.dropna(subset=["date", col])
                    market_f = market_f[market_f["date"].dt.year.between(year_range[0], year_range[1])]
                    if market_f.empty:
                        continue
                    target_axis = "y3" if item == steel_label else "y2"
                    fig_stock.add_trace(
                        go.Scatter(
                            x=market_f["date"],
                            y=market_f[col],
                            mode="lines",
                            name=item.replace("Market: ", ""),
                            yaxis=target_axis,
                        )
                    )
                    if target_axis == "y3":
                        y3_used = True
                    else:
                        y2_used = True

            yaxis2_cfg: dict[str, Any] = {
                "title": "Overlay metrics (Y2)",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "visible": y2_used,
            }
            if y3_used:
                yaxis2_cfg["anchor"] = "free"
                yaxis2_cfg["position"] = 0.9

            fig_stock.update_layout(
                template=plotly_template(),
                title="Vestas (VWS.CO) Monthly Price - Open/High/Low/Close",
                yaxis_title="Price",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False,
                height=470,
                yaxis2=yaxis2_cfg,
                yaxis3=dict(
                    title="Steel price (Y3, USD/short ton)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    visible=y3_used,
                    anchor="free",
                    position=0.995,
                ),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_stock, width="stretch")
            st.caption("Source: Yahoo Finance monthly history for VWS.CO, HRC=F (steel), and HG=F (copper).")

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

    g1, g2 = st.columns(2)
    with g1:
        growth = yearly[["order_year", "total_with_unannounced_mw"]].copy()
        growth["yoy_pct"] = growth["total_with_unannounced_mw"].pct_change() * 100.0
        growth = growth.dropna(subset=["yoy_pct"])
        if growth.empty:
            st.info("Not enough years selected for growth rates.")
        else:
            growth["direction"] = np.where(growth["yoy_pct"] >= 0, "Growth", "Decline")
            fig_g = px.bar(
                growth,
                x="order_year",
                y="yoy_pct",
                color="direction",
                color_discrete_map={"Growth": "#10B981", "Decline": "#D1495B"},
                template=plotly_template(),
                title="Year-over-Year Order Intake Growth (total MW incl. unannounced)",
                labels={"order_year": "Year", "yoy_pct": "YoY change (%)", "direction": ""},
                height=410,
            )
            fig_g.add_hline(y=0, line_color="rgba(148,163,184,0.6)")
            fig_g.update_layout(showlegend=False, yaxis=dict(ticksuffix="%"), margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_g, width="stretch")

    with g2:
        size_dist = orders.dropna(subset=["size_mw"]).copy()
        if size_dist.empty:
            st.info("No order size data for distribution view.")
        else:
            fig_box = px.box(
                size_dist,
                x="order_year",
                y="size_mw",
                template=plotly_template(),
                title="Order Size Distribution per Year (log scale)",
                labels={"order_year": "Year", "size_mw": "Order size (MW)"},
                height=410,
                log_y=True,
            )
            fig_box.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_box, width="stretch")
            st.caption("Widening boxes and rising medians show the shift toward fewer but much larger orders.")

    continent_year = (
        orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["continent", "order_year"])
    )
    if not continent_year.empty:
        continent_year["cum_sold_mw"] = continent_year.groupby("continent")["size_mw"].cumsum()
        yearly_totals = (
            continent_year.groupby("order_year", as_index=False)["size_mw"]
            .sum()
            .rename(columns={"size_mw": "year_total_mw"})
        )
        continent_share = continent_year.merge(yearly_totals, on="order_year", how="left")
        continent_share = continent_share[continent_share["year_total_mw"] > 0].copy()
        continent_share["market_share_pct"] = 100.0 * continent_share["size_mw"] / continent_share["year_total_mw"]

        c5, c6 = st.columns(2)
        with c5:
            fig_continent_cum = px.area(
                continent_year,
                x="order_year",
                y="cum_sold_mw",
                color="continent",
                template=plotly_template(),
                labels={"order_year": "Year", "cum_sold_mw": "Accumulated MW", "continent": ""},
                title="Accumulated Sold MW by Continent",
                height=430,
            )
            fig_continent_cum.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_continent_cum, width="stretch")

        with c6:
            fig_continent_share = px.bar(
                continent_share,
                x="order_year",
                y="market_share_pct",
                color="continent",
                barmode="stack",
                template=plotly_template(),
                labels={"order_year": "Year", "market_share_pct": "Market share (%)", "continent": ""},
                title="Continent Market Share by Year",
                hover_data={"size_mw": ":,.0f", "year_total_mw": ":,.0f", "market_share_pct": ":.1f"},
                height=430,
            )
            fig_continent_share.update_layout(
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_continent_share, width="stretch")

        st.caption("Accumulated chart uses announced order MW by continent.")

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
        rolling = quarter_df[["year_quarter", "total_mw"]].copy()
        rolling["rolling_4q_mw"] = rolling["total_mw"].rolling(4, min_periods=2).mean()
        fig1.add_trace(
            go.Scatter(
                x=rolling["year_quarter"],
                y=rolling["rolling_4q_mw"],
                mode="lines",
                name="4-quarter avg",
                line=dict(width=3, color="#D1495B"),
            )
        )
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")
        st.caption("The red line is the trailing 4-quarter average - the underlying intake momentum without quarter noise.")

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

        c_geo1, c_geo2 = st.columns(2)
        with c_geo1:
            continent_year = (
                orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if continent_year.empty:
                st.info("No continent data available for current filters.")
            else:
                fig_cont = px.bar(
                    continent_year,
                    x="order_year",
                    y="size_mw",
                    color="continent",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Continent MW Trends Across Years",
                    height=430,
                )
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_cont, width="stretch")

        with c_geo2:
            top_regions = top_n_by_mw(orders[orders["region"] != "Unknown"], "region", "size_mw", 10)
            region_year = (
                orders[orders["region"].isin(top_regions)]
                .groupby(["order_year", "region"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if region_year.empty:
                st.info("No region data available for current filters.")
            else:
                fig_reg = px.area(
                    region_year,
                    x="order_year",
                    y="size_mw",
                    color="region",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Top Regions MW Trends Across Years",
                    height=430,
                )
                fig_reg.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_reg, width="stretch")

        geo_tree = (
            orders.assign(
                continent=orders["continent"].fillna("Unknown"),
                region=orders["region"].fillna("Unknown"),
                country=orders["country"].fillna("Unknown"),
            )
            .groupby(["continent", "region", "country"], as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not geo_tree.empty:
            fig_geo_tree = px.sunburst(
                geo_tree,
                path=["continent", "region", "country"],
                values="size_mw",
                color="continent",
                template=plotly_template(),
                title="Country/Region/Continent Mix (Ordered MW)",
                height=600,
            )
            fig_geo_tree.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_geo_tree, width="stretch")

        table = (
            orders.groupby(["country", "region", "continent"], as_index=False)
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

        d1, d2 = st.columns(2)
        with d1:
            platform_share = (
                p_base.groupby("platform", as_index=False)["slot_mw"]
                .sum()
                .sort_values("slot_mw", ascending=False)
            )
            if not platform_share.empty:
                top_share = platform_share.head(10).copy()
                other_mw = float(platform_share["slot_mw"].sum() - top_share["slot_mw"].sum())
                if other_mw > 0:
                    top_share = pd.concat(
                        [top_share, pd.DataFrame([{"platform": "Other", "slot_mw": other_mw}])],
                        ignore_index=True,
                    )
                fig_pie_platform = px.pie(
                    top_share,
                    names="platform",
                    values="slot_mw",
                    hole=0.52,
                    template=plotly_template(),
                    title="Platform Share (Ordered MW)",
                    height=430,
                )
                fig_pie_platform.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie_platform.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_pie_platform, width="stretch")

        with d2:
            region_share = (
                p_base.groupby("region", as_index=False)["slot_mw"]
                .sum()
                .sort_values("slot_mw", ascending=False)
            )
            if not region_share.empty:
                fig_pie_region = px.pie(
                    region_share,
                    names="region",
                    values="slot_mw",
                    hole=0.52,
                    template=plotly_template(),
                    title="Region Share Across Platforms (MW)",
                    height=430,
                )
                fig_pie_region.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie_region.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_pie_region, width="stretch")

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


TURBINE_GENERATION_ORDER = [
    "Legacy (kW era)",
    "Classic MW (V80-V90)",
    "2 MW platform",
    "3 MW platform",
    "4 MW platform",
    "EnVentus",
    "Offshore",
    "Platform-level order",
]

TURBINE_GENERATION_COLORS = {
    "Legacy (kW era)": "#94A3B8",
    "Classic MW (V80-V90)": "#64748B",
    "2 MW platform": "#0EA5E9",
    "3 MW platform": "#10B981",
    "4 MW platform": "#F59E0B",
    "EnVentus": "#D1495B",
    "Offshore": "#6D5BD0",
    "Platform-level order": "#CBD5E1",
}

# Indicative public profile per rotor family. Hub heights / wind classes are
# typical catalog values from public Vestas materials, not order-specific data.
TURBINE_FAMILY_INFO: dict[str, dict[str, Any]] = {
    "V52": {
        "segment": "Onshore",
        "wind_class": "IEC IA / IIA (high-medium wind)",
        "hub_heights": "44-74 m (typ.)",
        "introduced": "~2000",
        "note": "850 kW workhorse of the early 2000s with thousands of units installed worldwide.",
    },
    "V60": {
        "segment": "Onshore",
        "wind_class": "IEC IIIA (low wind)",
        "hub_heights": "site-specific",
        "introduced": "~2011",
        "note": "Stretched-rotor 850 kW variant developed primarily for lower-wind sites in China.",
    },
    "V80": {
        "segment": "Onshore + early offshore",
        "wind_class": "IEC IA / IIA",
        "hub_heights": "60-100 m (typ.)",
        "introduced": "~1999",
        "note": "Early 2 MW machine; offshore variants powered pioneering projects such as Horns Rev I.",
    },
    "V82": {
        "segment": "Onshore",
        "wind_class": "IEC IIIA (low wind)",
        "hub_heights": "70-80 m (typ.)",
        "introduced": "~2002",
        "note": "Low-wind 1.65 MW machine that joined the portfolio via the NEG Micon merger (2004).",
    },
    "V90": {
        "segment": "Onshore + early offshore",
        "wind_class": "IEC IA-IIIA depending on variant",
        "hub_heights": "80-105 m (typ.)",
        "introduced": "~2003",
        "note": "Highly successful family; the V90-3.0 also served early offshore farms (e.g., Barrow, Thanet).",
    },
    "V100": {
        "segment": "Onshore",
        "wind_class": "IEC IIA / IIIA (low-medium wind)",
        "hub_heights": "80-95 m (typ.)",
        "introduced": "~2009",
        "note": "2 MW platform low-wind rotor; particularly strong in the Americas.",
    },
    "V105": {
        "segment": "Onshore",
        "wind_class": "IEC IA (high wind)",
        "hub_heights": "72-95 m (typ.)",
        "introduced": "~2013",
        "note": "High-wind sibling of the 3 MW platform.",
    },
    "V110": {
        "segment": "Onshore",
        "wind_class": "IEC IIIA (low wind)",
        "hub_heights": "80-125 m (typ.)",
        "introduced": "~2013",
        "note": "US PTC-era best seller of the 2 MW platform.",
    },
    "V112": {
        "segment": "Onshore + offshore variant",
        "wind_class": "IEC IIA / IIIA",
        "hub_heights": "84-119 m (typ.)",
        "introduced": "~2010",
        "note": "First member of the 3 MW platform; an offshore variant ran at Kårehamn.",
    },
    "V116": {
        "segment": "Onshore",
        "wind_class": "IEC IIB / S",
        "hub_heights": "80-94 m (typ.)",
        "introduced": "~2017",
        "note": "2 MW platform extension announced alongside the V120 for medium-wind sites.",
    },
    "V117": {
        "segment": "Onshore",
        "wind_class": "IEC IB / IIA + typhoon-class variants",
        "hub_heights": "80-141 m (typ.)",
        "introduced": "~2012",
        "note": "Medium/high-wind rotor; typhoon-rated variants serve markets such as Japan.",
    },
    "V120": {
        "segment": "Onshore",
        "wind_class": "IEC IIIB (low wind)",
        "hub_heights": "up to ~139 m",
        "introduced": "~2017",
        "note": "Low-wind 2 MW platform stretch aimed at India and US repowering.",
    },
    "V126": {
        "segment": "Onshore",
        "wind_class": "IEC IIIA / IIIB (low wind)",
        "hub_heights": "87-166 m (incl. large steel towers)",
        "introduced": "~2012",
        "note": "Low-wind 3 MW platform rotor with some of the tallest tower options in the fleet.",
    },
    "V135": {
        "segment": "Onshore",
        "wind_class": "IEC S",
        "hub_heights": "site-specific",
        "introduced": "~2023",
        "note": "4 MW platform variant for special site conditions.",
    },
    "V136": {
        "segment": "Onshore",
        "wind_class": "IEC IIB / IIIA / S",
        "hub_heights": "82-166 m (typ.)",
        "introduced": "~2015",
        "note": "Low/medium-wind rotor; 4.x MW uprates arrived from 2017 onward.",
    },
    "V150": {
        "segment": "Onshore",
        "wind_class": "IEC IIIB / S (low wind)",
        "hub_heights": "105-166 m (typ.)",
        "introduced": "~2017",
        "note": "Best-selling low-wind model in this dataset; EnVentus 5.6-6.0 MW variants from 2019.",
    },
    "V155": {
        "segment": "Onshore",
        "wind_class": "IEC S (low wind)",
        "hub_heights": "~120 m",
        "introduced": "~2020",
        "note": "Tailored for India and similar mid/low-wind markets.",
    },
    "V162": {
        "segment": "Onshore",
        "wind_class": "IEC S / IIIB",
        "hub_heights": "119-169 m (typ.)",
        "introduced": "~2019",
        "note": "Flagship rotor of the modular EnVentus platform.",
    },
    "V163": {
        "segment": "Onshore",
        "wind_class": "IEC S (low wind)",
        "hub_heights": "up to ~166 m",
        "introduced": "~2022",
        "note": "4 MW platform evolution for low-wind sites.",
    },
    "V164": {
        "segment": "Offshore",
        "wind_class": "IEC S (B)",
        "hub_heights": "105-140 m (site-specific)",
        "introduced": "~2014",
        "note": "MHI Vestas offshore platform; the most powerful turbine in the world at launch (8-10 MW).",
    },
    "V172": {
        "segment": "Onshore",
        "wind_class": "IEC S (low-medium wind)",
        "hub_heights": "up to ~175 m",
        "introduced": "~2022",
        "note": "Largest-rotor EnVentus onshore turbine to date.",
    },
    "V174": {
        "segment": "Offshore",
        "wind_class": "IEC S",
        "hub_heights": "site-specific",
        "introduced": "~2019",
        "note": "Offshore rotor stretch of the V164 platform (9.5 MW class).",
    },
    "V236": {
        "segment": "Offshore",
        "wind_class": "IEC S (B), typhoon option",
        "hub_heights": "~145-150 m (site-specific)",
        "introduced": "prototype 2022, serial ~2024",
        "note": "Offshore flagship with a 43,700+ m2 swept area; among the world's most powerful turbines.",
    },
}


def turbine_family(platform: Any) -> str:
    text = clean_text(platform) or "Unknown"
    match = re.match(r"^(V\d{2,3})-", text.upper())
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+(\.\d+)?MW", text.upper()):
        return "Platform-level"
    return "Other"


def turbine_generation(family: str, rating: float) -> str:
    if family in {"V164", "V174", "V236"}:
        return "Offshore"
    if family in {"V52", "V60"}:
        return "Legacy (kW era)"
    if family in {"V80", "V82", "V90"}:
        return "Classic MW (V80-V90)"
    if family in {"V100", "V110", "V116", "V120"}:
        return "2 MW platform"
    if family in {"V162", "V172"}:
        return "EnVentus"
    if family == "V150" and not math.isnan(rating) and rating >= 5.5:
        return "EnVentus"
    if family in {"V105", "V112"}:
        return "3 MW platform"
    if family in {"V117", "V126", "V136"}:
        if not math.isnan(rating) and rating >= 4.0:
            return "4 MW platform"
        return "3 MW platform"
    if family in {"V135", "V150", "V155", "V163"}:
        return "4 MW platform"
    if family == "Platform-level":
        return "Platform-level order"
    if not math.isnan(rating):
        if rating < 1.0:
            return "Legacy (kW era)"
        if rating < 2.7:
            return "2 MW platform"
        if rating < 4.0:
            return "3 MW platform"
        if rating < 5.5:
            return "4 MW platform"
        if rating < 8.0:
            return "EnVentus"
        return "Offshore"
    return "Platform-level order"


def build_turbine_catalog(platforms: pd.DataFrame) -> pd.DataFrame:
    base = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
    if base.empty:
        return pd.DataFrame()

    base["rotor_m"] = pd.to_numeric(base["rotor_m"], errors="coerce")
    base["mw_rating"] = pd.to_numeric(base["mw_rating"], errors="coerce")
    base["turbines_qty"] = pd.to_numeric(base["turbines_qty"], errors="coerce")
    base["order_date"] = pd.to_datetime(base["order_date"], errors="coerce")

    def top_label(series: pd.Series) -> str:
        vals = series.dropna()
        vals = vals[~vals.isin(["Unknown"])]
        if vals.empty:
            return "-"
        return vals.value_counts().idxmax()

    catalog = (
        base.groupby("platform", as_index=False)
        .agg(
            rotor_m=("rotor_m", "median"),
            mw_rating=("mw_rating", "median"),
            units=("turbines_qty", "sum"),
            total_mw=("slot_mw", "sum"),
            orders=("order_id", "nunique"),
            countries=("country", "nunique"),
            customers=("customer", lambda s: s[~s.isin(["Unknown"])].nunique()),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
            top_country=("country", top_label),
            avg_delivery_days=("delivery_days", "mean"),
        )
        .copy()
    )

    # The variant name (e.g. V136-3.45MW) is the authoritative spec; raw sheet
    # cells occasionally hold uprated or slot-total values, so they only fill gaps.
    parsed = catalog["platform"].map(parse_platform_specs)
    name_rotor = parsed.map(lambda t: t[0])
    name_rating = parsed.map(lambda t: t[1])
    catalog["rotor_m"] = name_rotor.fillna(catalog["rotor_m"])
    catalog["mw_rating"] = name_rating.fillna(catalog["mw_rating"])

    catalog["family"] = catalog["platform"].map(turbine_family)
    catalog["generation"] = [
        turbine_generation(fam, rating if pd.notna(rating) else np.nan)
        for fam, rating in zip(catalog["family"], catalog["mw_rating"], strict=False)
    ]
    catalog["swept_area_m2"] = np.pi * (catalog["rotor_m"] / 2.0) ** 2
    catalog["specific_power_w_m2"] = catalog["mw_rating"] * 1_000_000.0 / catalog["swept_area_m2"]
    catalog["avg_order_mw"] = catalog["total_mw"] / catalog["orders"]
    total = catalog["total_mw"].sum()
    catalog["mw_share_pct"] = 100.0 * catalog["total_mw"] / total if total > 0 else 0.0
    catalog["first_year"] = catalog["first_order"].dt.year
    catalog["last_year"] = catalog["last_order"].dt.year
    catalog["active_years"] = catalog["last_year"] - catalog["first_year"] + 1
    return catalog.sort_values("total_mw", ascending=False)


def render_turbine_explorer(platforms: pd.DataFrame, orders: pd.DataFrame) -> None:
    st.subheader("Turbine Explorer")
    st.caption(
        "Every turbine variant found in the order book, enriched with engineering metrics "
        "(swept area, specific power) and an indicative public family profile. Respects sidebar filters."
    )

    catalog = build_turbine_catalog(platforms)
    if catalog.empty:
        st.info("No platform-mapped turbine rows after filtering.")
        return

    named = catalog[~catalog["generation"].eq("Platform-level order")].copy()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Turbine variants", int_fmt(len(catalog)))
    k2.metric("Rotor families", int_fmt(named["family"].nunique()))
    k3.metric("Turbines (est.)", int_fmt(catalog["units"].sum()))
    k4.metric("Platform-mapped MW", mw_fmt(catalog["total_mw"].sum()))
    if named["mw_rating"].notna().any():
        flagship = named.loc[named["mw_rating"].idxmax()]
        k5.metric("Largest rating", f"{flagship['mw_rating']:.1f} MW", delta=flagship["platform"], delta_color="off")
    best = catalog.iloc[0]
    k6.metric("Top seller (MW)", best["platform"], delta=f"{best['total_mw']:,.0f} MW", delta_color="off")

    view_tabs = st.tabs(["Model Catalog", "Technology Map", "Product Lifecycle", "Model Detail"])

    with view_tabs[0]:
        generations = [g for g in TURBINE_GENERATION_ORDER if g in set(catalog["generation"])]
        gen_filter = st.multiselect(
            "Filter by platform generation",
            options=generations,
            default=[],
            key="turbine_catalog_generation",
        )
        table = catalog[catalog["generation"].isin(gen_filter)] if gen_filter else catalog

        display = table[
            [
                "platform",
                "generation",
                "rotor_m",
                "mw_rating",
                "swept_area_m2",
                "specific_power_w_m2",
                "units",
                "total_mw",
                "mw_share_pct",
                "orders",
                "countries",
                "customers",
                "first_year",
                "last_year",
                "top_country",
                "avg_order_mw",
            ]
        ].copy()
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=560,
            column_config={
                "platform": st.column_config.TextColumn("Model", pinned=True),
                "generation": st.column_config.TextColumn("Generation"),
                "rotor_m": st.column_config.NumberColumn("Rotor (m)", format="%.0f"),
                "mw_rating": st.column_config.NumberColumn("Rating (MW)", format="%.2f"),
                "swept_area_m2": st.column_config.NumberColumn("Swept area (m2)", format="%.0f"),
                "specific_power_w_m2": st.column_config.NumberColumn("Specific power (W/m2)", format="%.0f"),
                "units": st.column_config.NumberColumn("Units (est.)", format="%.0f"),
                "total_mw": st.column_config.ProgressColumn(
                    "Ordered MW",
                    format="%.0f",
                    min_value=0.0,
                    max_value=float(catalog["total_mw"].max()),
                ),
                "mw_share_pct": st.column_config.NumberColumn("MW share (%)", format="%.1f"),
                "orders": st.column_config.NumberColumn("Orders", format="%.0f"),
                "countries": st.column_config.NumberColumn("Countries", format="%.0f"),
                "customers": st.column_config.NumberColumn("Known customers", format="%.0f"),
                "first_year": st.column_config.NumberColumn("First order", format="%d"),
                "last_year": st.column_config.NumberColumn("Last order", format="%d"),
                "top_country": st.column_config.TextColumn("Top country"),
                "avg_order_mw": st.column_config.NumberColumn("Avg order (MW)", format="%.0f"),
            },
        )
        st.download_button(
            "Download turbine catalog (CSV)",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="vestas_turbine_catalog.csv",
            mime="text/csv",
        )
        st.caption(
            "Units are estimated from order-level turbine counts. Generic rows such as '2MW'/'4MW' are "
            "orders announced at platform level without a named variant."
        )

    with view_tabs[1]:
        tech = named.dropna(subset=["rotor_m", "mw_rating"]).copy()
        if tech.empty:
            st.info("No rotor/rating data available for the current filters.")
        else:
            fig = px.scatter(
                tech,
                x="rotor_m",
                y="mw_rating",
                size="total_mw",
                color="generation",
                color_discrete_map=TURBINE_GENERATION_COLORS,
                category_orders={"generation": TURBINE_GENERATION_ORDER},
                hover_name="platform",
                hover_data={
                    "rotor_m": ":.0f",
                    "mw_rating": ":.2f",
                    "specific_power_w_m2": ":.0f",
                    "units": ":,.0f",
                    "total_mw": ":,.0f",
                    "first_year": True,
                    "last_year": True,
                    "generation": False,
                },
                size_max=52,
                template=plotly_template(),
                title="Rotor Diameter vs Rated Power (bubble = ordered MW)",
                labels={"rotor_m": "Rotor diameter (m)", "mw_rating": "Rated power (MW)"},
                height=620,
            )
            rotor_span = np.linspace(max(40.0, tech["rotor_m"].min() * 0.9), tech["rotor_m"].max() * 1.06, 60)
            max_rating = tech["mw_rating"].max() * 1.18
            for sp in (250, 350, 450):
                iso = sp * np.pi * (rotor_span / 2.0) ** 2 / 1_000_000.0
                mask = iso <= max_rating
                if not mask.any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=rotor_span[mask],
                        y=iso[mask],
                        mode="lines",
                        line=dict(color="rgba(148,163,184,0.55)", width=1, dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig.add_annotation(
                    x=float(rotor_span[mask][-1]),
                    y=float(iso[mask][-1]),
                    text=f"{sp} W/m2",
                    showarrow=False,
                    font=dict(size=11, color="rgba(148,163,184,0.9)"),
                    xanchor="left",
                )
            fig.update_yaxes(range=[0, max_rating])
            fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Dotted isolines show constant specific power. Lower specific power = larger rotor per MW, "
                "built for lower-wind sites and higher capacity factors."
            )

            sp_base = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
            sp_base["rotor_m"] = pd.to_numeric(sp_base["rotor_m"], errors="coerce")
            sp_base["mw_rating"] = pd.to_numeric(sp_base["mw_rating"], errors="coerce")
            sp_base = sp_base.dropna(subset=["rotor_m", "mw_rating"])
            sp_base = sp_base[sp_base["rotor_m"] > 0]
            if not sp_base.empty:
                sp_base["specific_power"] = sp_base["mw_rating"] * 1_000_000.0 / (np.pi * (sp_base["rotor_m"] / 2.0) ** 2)
                sp_year = (
                    sp_base.assign(weighted=sp_base["specific_power"] * sp_base["slot_mw"])
                    .groupby("order_year", as_index=False)
                    .agg(weighted=("weighted", "sum"), mw=("slot_mw", "sum"))
                )
                sp_year["avg_specific_power"] = sp_year["weighted"] / sp_year["mw"]
                fig_sp = px.line(
                    sp_year,
                    x="order_year",
                    y="avg_specific_power",
                    markers=True,
                    template=plotly_template(),
                    title="MW-Weighted Average Specific Power of Ordered Turbines",
                    labels={"order_year": "Order year", "avg_specific_power": "Specific power (W/m2)"},
                    height=400,
                )
                fig_sp.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_sp, width="stretch")
                st.caption(
                    "The long-run decline in specific power reflects the industry shift toward "
                    "low-wind rotors that maximise energy capture per installed MW."
                )

    with view_tabs[2]:
        life = catalog.dropna(subset=["first_order", "last_order"]).copy()
        if life.empty:
            st.info("No order dates available to build the lifecycle view.")
        else:
            life["lifecycle_end"] = life["last_order"] + pd.Timedelta(days=60)
            life = life.sort_values("first_order")
            fig_life = px.timeline(
                life,
                x_start="first_order",
                x_end="lifecycle_end",
                y="platform",
                color="generation",
                color_discrete_map=TURBINE_GENERATION_COLORS,
                category_orders={"generation": TURBINE_GENERATION_ORDER},
                hover_data={
                    "total_mw": ":,.0f",
                    "units": ":,.0f",
                    "orders": ":,.0f",
                    "first_year": True,
                    "last_year": True,
                    "lifecycle_end": False,
                },
                template=plotly_template(),
                title="Sales Lifecycle per Model (first to last order in dataset)",
                height=max(520, 24 * len(life) + 140),
            )
            fig_life.update_yaxes(autorange="reversed", title="")
            fig_life.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_life, width="stretch")
            st.caption(
                "Bars span from the first to the last order observed for each variant - a direct view of "
                "product generation turnover and how long each model stays commercially active."
            )

    with view_tabs[3]:
        model_options = catalog["platform"].tolist()
        selected_model = st.selectbox("Choose a turbine model", options=model_options, key="turbine_detail_model")
        row = catalog[catalog["platform"] == selected_model].iloc[0]
        family_info = TURBINE_FAMILY_INFO.get(row["family"], {})

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Rotor diameter", f"{row['rotor_m']:.0f} m" if pd.notna(row["rotor_m"]) else "-")
        d2.metric("Rated power", f"{row['mw_rating']:.2f} MW" if pd.notna(row["mw_rating"]) else "-")
        d3.metric(
            "Swept area",
            f"{row['swept_area_m2']:,.0f} m2" if pd.notna(row["swept_area_m2"]) else "-",
        )
        d4.metric(
            "Specific power",
            f"{row['specific_power_w_m2']:,.0f} W/m2" if pd.notna(row["specific_power_w_m2"]) else "-",
        )
        d5.metric("Generation", row["generation"])

        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Ordered MW", mw_fmt(row["total_mw"]))
        e2.metric("Units (est.)", int_fmt(row["units"]) if pd.notna(row["units"]) and row["units"] > 0 else "-")
        e3.metric("Orders", int_fmt(row["orders"]))
        e4.metric("Countries", int_fmt(row["countries"]))
        e5.metric(
            "Active span",
            f"{int(row['first_year'])}-{int(row['last_year'])}" if pd.notna(row["first_year"]) else "-",
        )

        if family_info:
            st.markdown(
                f"**Indicative {row['family']} family profile** (public catalog data - verify against official spec sheets)  \n"
                f"Segment: {family_info.get('segment', '-')} | Wind class: {family_info.get('wind_class', '-')} | "
                f"Typical hub heights: {family_info.get('hub_heights', '-')} | Introduced: {family_info.get('introduced', '-')}  \n"
                f"{family_info.get('note', '')}"
            )
        else:
            st.caption("No curated family profile available for this entry.")

        model_rows = platforms[platforms["platform"] == selected_model].copy()
        c1, c2 = st.columns(2)
        with c1:
            yearly_mw = model_rows.groupby("order_year", as_index=False)["slot_mw"].sum()
            fig_y = px.bar(
                yearly_mw,
                x="order_year",
                y="slot_mw",
                template=plotly_template(),
                title=f"{selected_model}: Ordered MW per Year",
                labels={"order_year": "Year", "slot_mw": "MW"},
                height=420,
            )
            fig_y.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_y, width="stretch")

        with c2:
            country_mw = (
                model_rows.groupby("country", as_index=False)["slot_mw"]
                .sum()
                .sort_values("slot_mw", ascending=False)
                .head(12)
            )
            fig_c = px.bar(
                country_mw,
                x="slot_mw",
                y="country",
                orientation="h",
                template=plotly_template(),
                title=f"{selected_model}: Top Countries",
                labels={"slot_mw": "MW", "country": ""},
                height=420,
            )
            fig_c.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_c, width="stretch")

        cust = (
            model_rows[model_rows["customer"] != "Unknown"]
            .groupby("customer", as_index=False)
            .agg(mw=("slot_mw", "sum"), orders=("order_id", "nunique"))
            .sort_values("mw", ascending=False)
            .head(10)
        )
        if not cust.empty:
            st.markdown(f"**Top known customers for {selected_model}**")
            st.dataframe(
                cust.rename(columns={"customer": "Customer", "mw": "MW", "orders": "Orders"}),
                width="stretch",
                hide_index=True,
            )


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
    rc = map_base["country"].apply(country_region_continent)
    map_base["region"] = rc.apply(lambda t: t[0])
    map_base["continent"] = rc.apply(lambda t: t[1])

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

    map_style = st.selectbox(
        "Map style",
        options=["Bubble map", "Choropleth map"],
        index=0,
        key="country_map_style",
    )

    map_geo["orders"] = pd.to_numeric(map_geo["orders"], errors="coerce")

    if map_style == "Bubble map":
        c5, c6 = st.columns(2)
        with c5:
            bubble_size_metric = st.selectbox(
                "Bubble size metric",
                options=list(size_metric_labels.keys()),
                format_func=lambda k: size_metric_labels[k],
                key="country_map_bubble_size_metric",
            )
        with c6:
            bubble_color_metric = st.selectbox(
                "Bubble color metric",
                options=list(color_metric_labels.keys()),
                format_func=lambda k: color_metric_labels[k],
                key="country_map_bubble_color_metric",
            )

        map_geo[bubble_size_metric] = pd.to_numeric(map_geo[bubble_size_metric], errors="coerce")
        map_geo[bubble_color_metric] = pd.to_numeric(map_geo[bubble_color_metric], errors="coerce")

        bubble_data = map_geo.dropna(subset=[bubble_size_metric]).copy()
        bubble_data = bubble_data[bubble_data[bubble_size_metric] > 0]
        if bubble_data[bubble_color_metric].notna().any():
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()
        else:
            bubble_color_metric = "orders"
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()

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
                    "region": True,
                    "continent": True,
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
                height=600,
            )
            fig_bubble.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_bubble.update_layout(
                coloraxis_colorbar_title=color_metric_labels[bubble_color_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_bubble, width="stretch")
    else:
        choropleth_metric = st.selectbox(
            "Choropleth metric",
            options=list(map_metric_labels.keys()),
            format_func=lambda k: map_metric_labels[k],
            key="country_map_choropleth_metric",
        )
        map_geo[choropleth_metric] = pd.to_numeric(map_geo[choropleth_metric], errors="coerce")
        choropleth_data = map_geo.dropna(subset=[choropleth_metric]).copy()
        if choropleth_metric in {"total_mw", "platform_overlay_mw", "orders"}:
            choropleth_data = choropleth_data[choropleth_data[choropleth_metric] > 0]

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
                    "region": True,
                    "continent": True,
                    "dominant_platform": True,
                    "dominant_service_scheme": True,
                    "iso3": False,
                },
                template=plotly_template(),
                color_continuous_scale="YlGnBu",
                title=f"Zoomable Choropleth: {map_metric_labels[choropleth_metric]}",
                height=600,
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_map.update_layout(
                coloraxis_colorbar_title=map_metric_labels[choropleth_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_map, width="stretch")


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

    c5, c6 = st.columns(2)
    with c5:
        continent_mw = (
            o.groupby("continent", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"))
            .sort_values("total_mw", ascending=False)
        )
        fig5 = px.bar(
            continent_mw,
            x="continent",
            y="total_mw",
            color="continent",
            template=plotly_template(),
            title="Across Countries: Continent Totals",
            labels={"total_mw": "MW", "continent": ""},
            height=430,
        )
        fig5.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig5, width="stretch")

    with c6:
        region_stats = (
            o.groupby("region", as_index=False)
            .agg(
                total_mw=("size_mw", "sum"),
                avg_service_time=("service_time_years", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
            .sort_values("total_mw", ascending=False)
            .head(12)
        )
        region_stats = region_stats.dropna(subset=["avg_service_time", "avg_delivery_days"])
        if region_stats.empty:
            st.info("No complete region-level service/delivery data for current filters.")
        else:
            fig6 = px.scatter(
                region_stats,
                x="avg_service_time",
                y="avg_delivery_days",
                size="total_mw",
                color="region",
                template=plotly_template(),
                title="Across Countries: Region Service vs Delivery",
                labels={"avg_service_time": "Avg service time (years)", "avg_delivery_days": "Avg delivery days"},
                height=430,
            )
            fig6.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig6, width="stretch")

    geo_tree = (
        o.assign(
            continent=o["continent"].fillna("Unknown"),
            region=o["region"].fillna("Unknown"),
            country=o["country"].fillna("Unknown"),
        )
        .groupby(["continent", "region", "country"], as_index=False)["size_mw"]
        .sum()
        .sort_values("size_mw", ascending=False)
    )
    if not geo_tree.empty:
        fig_tree = px.sunburst(
            geo_tree,
            path=["continent", "region", "country"],
            values="size_mw",
            color="continent",
            template=plotly_template(),
            title="Across Countries: Continent -> Region -> Country (Ordered MW)",
            height=640,
        )
        fig_tree.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_tree, width="stretch")

    c7, c8 = st.columns(2)
    with c7:
        continent_share = (
            o.groupby("continent", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not continent_share.empty:
            fig7 = px.pie(
                continent_share,
                names="continent",
                values="size_mw",
                hole=0.52,
                template=plotly_template(),
                title="Across Countries: Continent Share (MW)",
                height=430,
            )
            fig7.update_traces(textposition="inside", textinfo="percent+label")
            fig7.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig7, width="stretch")

    with c8:
        country_share = (
            o.groupby("country", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not country_share.empty:
            top_country_share = country_share.head(12).copy()
            other_mw = float(country_share["size_mw"].sum() - top_country_share["size_mw"].sum())
            if other_mw > 0:
                top_country_share = pd.concat(
                    [top_country_share, pd.DataFrame([{"country": "Other", "size_mw": other_mw}])],
                    ignore_index=True,
                )
            fig8 = px.pie(
                top_country_share,
                names="country",
                values="size_mw",
                hole=0.52,
                template=plotly_template(),
                title="Across Countries: Top Country Share (MW)",
                height=430,
            )
            fig8.update_traces(textposition="inside", textinfo="percent+label")
            fig8.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig8, width="stretch")


def render_customer_intelligence(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Customer Intelligence")
    st.caption(
        "Who buys, how concentrated the order book is, and how much business comes from "
        "repeat customers. Based on announced orders with a disclosed customer; respects sidebar filters."
    )

    if orders.empty:
        st.info("No order rows after filtering.")
        return

    total_mw = float(orders["size_mw"].sum())
    known = orders[orders["customer"] != "Unknown"].copy()
    if known.empty:
        st.info("No orders with a known customer in the current selection.")
        return
    known["size_mw"] = pd.to_numeric(known["size_mw"], errors="coerce").fillna(0.0)
    known_mw = float(known["size_mw"].sum())

    cust_stats = (
        known.groupby("customer", as_index=False)
        .agg(
            total_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            first_year=("order_year", "min"),
            last_year=("order_year", "max"),
            countries=("country", "nunique"),
            avg_order_mw=("size_mw", "mean"),
        )
        .sort_values("total_mw", ascending=False)
    )
    cust_stats["years_active"] = cust_stats["last_year"] - cust_stats["first_year"] + 1
    cust_stats["mw_share_pct"] = 100.0 * cust_stats["total_mw"] / known_mw if known_mw > 0 else 0.0

    repeat_customers = cust_stats[cust_stats["orders"] >= 2]
    repeat_mw_share = 100.0 * repeat_customers["total_mw"].sum() / known_mw if known_mw > 0 else 0.0
    top10_share = float(cust_stats.head(10)["total_mw"].sum() / known_mw * 100.0) if known_mw > 0 else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Known customers", int_fmt(len(cust_stats)))
    k2.metric(
        "Disclosed-customer MW",
        mw_fmt(known_mw),
        delta=f"{100.0 * known_mw / total_mw:,.0f}% of announced MW" if total_mw > 0 else None,
        delta_color="off",
    )
    k3.metric(
        "Repeat-buyer MW share",
        f"{repeat_mw_share:,.0f}%",
        delta=f"{len(repeat_customers):,} customers with 2+ orders",
        delta_color="off",
    )
    k4.metric("Top-10 concentration", f"{top10_share:,.0f}%", delta="of disclosed MW", delta_color="off")
    top_account = cust_stats.iloc[0]
    k5.metric("Largest account", top_account["customer"], delta=mw_fmt(top_account["total_mw"]), delta_color="off")

    first_year_map = cust_stats.set_index("customer")["first_year"]
    known["customer_status"] = np.where(
        known["order_year"] > known["customer"].map(first_year_map),
        "Returning customer",
        "New customer",
    )

    c1, c2 = st.columns(2)
    with c1:
        status_year = (
            known.groupby(["order_year", "customer_status"], as_index=False)["size_mw"].sum().sort_values("order_year")
        )
        fig_status = px.bar(
            status_year,
            x="order_year",
            y="size_mw",
            color="customer_status",
            barmode="stack",
            template=plotly_template(),
            color_discrete_map={"New customer": "#F59E0B", "Returning customer": "#10B981"},
            title="New vs Returning Customer MW by Year",
            labels={"order_year": "Year", "size_mw": "MW", "customer_status": ""},
            height=440,
        )
        fig_status.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_status, width="stretch")
        st.caption("A customer counts as 'new' in the first year it appears anywhere in the dataset.")

    with c2:
        conc_rows: list[dict[str, Any]] = []
        for year, grp in known.groupby("order_year"):
            mw_by_cust = grp.groupby("customer")["size_mw"].sum().sort_values(ascending=False)
            year_mw = float(mw_by_cust.sum())
            if year_mw <= 0:
                continue
            conc_rows.append(
                {
                    "order_year": int(year),
                    "Top 1": 100.0 * mw_by_cust.head(1).sum() / year_mw,
                    "Top 5": 100.0 * mw_by_cust.head(5).sum() / year_mw,
                    "Top 10": 100.0 * mw_by_cust.head(10).sum() / year_mw,
                }
            )
        conc = pd.DataFrame(conc_rows)
        if not conc.empty:
            conc_long = conc.melt(id_vars="order_year", var_name="bucket", value_name="share_pct")
            fig_conc = px.line(
                conc_long,
                x="order_year",
                y="share_pct",
                color="bucket",
                markers=True,
                template=plotly_template(),
                title="Customer Concentration per Year (share of disclosed MW)",
                labels={"order_year": "Year", "share_pct": "Share (%)", "bucket": ""},
                height=440,
            )
            fig_conc.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"), margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_conc, width="stretch")
            st.caption("Falling lines = a broader customer base; rising lines = growing key-account dependency.")

    top_n = st.slider("Key accounts to track", 5, 20, 10, key="customer_intel_top_n")
    top_customers = cust_stats.head(top_n)["customer"].tolist()
    cum = (
        known[known["customer"].isin(top_customers)]
        .groupby(["order_year", "customer"], as_index=False)["size_mw"]
        .sum()
        .sort_values("order_year")
    )
    if not cum.empty:
        cum["cumulative_mw"] = cum.groupby("customer")["size_mw"].cumsum()
        fig_cum = px.line(
            cum,
            x="order_year",
            y="cumulative_mw",
            color="customer",
            markers=True,
            template=plotly_template(),
            title=f"Cumulative MW Bought by Top {top_n} Accounts",
            labels={"order_year": "Year", "cumulative_mw": "Cumulative MW", "customer": ""},
            height=480,
        )
        fig_cum.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_cum, width="stretch")

    st.markdown("**Key account table**")
    st.dataframe(
        cust_stats.head(40)[
            ["customer", "total_mw", "mw_share_pct", "orders", "avg_order_mw", "countries", "first_year", "last_year", "years_active"]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "customer": st.column_config.TextColumn("Customer", pinned=True),
            "total_mw": st.column_config.ProgressColumn(
                "Total MW",
                format="%.0f",
                min_value=0.0,
                max_value=float(cust_stats["total_mw"].max()),
            ),
            "mw_share_pct": st.column_config.NumberColumn("MW share (%)", format="%.1f"),
            "orders": st.column_config.NumberColumn("Orders", format="%.0f"),
            "avg_order_mw": st.column_config.NumberColumn("Avg order (MW)", format="%.0f"),
            "countries": st.column_config.NumberColumn("Countries", format="%.0f"),
            "first_year": st.column_config.NumberColumn("First order", format="%d"),
            "last_year": st.column_config.NumberColumn("Last order", format="%d"),
            "years_active": st.column_config.NumberColumn("Years active", format="%d"),
        },
    )

    st.markdown("**Customer drill-down**")
    selected_customer = st.selectbox(
        "Choose a customer",
        options=cust_stats["customer"].tolist(),
        key="customer_intel_detail",
    )
    cust_orders = known[known["customer"] == selected_customer]
    cust_platforms = platforms[(platforms["customer"] == selected_customer) & (platforms["platform"] != "Unknown")]

    cd1, cd2 = st.columns(2)
    with cd1:
        yearly = cust_orders.groupby("order_year", as_index=False)["size_mw"].sum()
        fig_cy = px.bar(
            yearly,
            x="order_year",
            y="size_mw",
            template=plotly_template(),
            title=f"{selected_customer}: Ordered MW per Year",
            labels={"order_year": "Year", "size_mw": "MW"},
            height=400,
        )
        fig_cy.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_cy, width="stretch")

    with cd2:
        if cust_platforms.empty:
            geo = cust_orders.groupby("country", as_index=False)["size_mw"].sum().sort_values("size_mw", ascending=False)
            fig_cp = px.bar(
                geo,
                x="size_mw",
                y="country",
                orientation="h",
                template=plotly_template(),
                title=f"{selected_customer}: Countries",
                labels={"size_mw": "MW", "country": ""},
                height=400,
            )
        else:
            mix = cust_platforms.groupby("platform", as_index=False)["slot_mw"].sum().sort_values("slot_mw", ascending=False).head(12)
            fig_cp = px.bar(
                mix,
                x="slot_mw",
                y="platform",
                orientation="h",
                template=plotly_template(),
                title=f"{selected_customer}: Turbine Models Bought",
                labels={"slot_mw": "MW", "platform": ""},
                height=400,
            )
        fig_cp.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_cp, width="stretch")


SANKEY_COLORS = {
    "continent": "#2F8FCE",
    "region": "#10B981",
    "country": "#F59E0B",
    "platform": "#D1495B",
    "year": "#6D5BD0",
    "rotor": "#06B6D4",
    "rating": "#F97316",
    "customer": "#EAB308",
    "service": "#14B8A6",
    "duration": "#A855F7",
    "other": "#64748B",
}

SANKEY_YEAR_ORDER = ["2008-2012", "2013-2016", "2017-2020", "2021-2023", "2024-2026"]
SANKEY_ROTOR_ORDER = ["<100 m rotor", "100-129 m rotor", "130-149 m rotor", "150-169 m rotor", "170+ m rotor", "Rotor unknown"]
SANKEY_RATING_ORDER = ["<2.5 MW", "2.5-3.9 MW", "4.0-5.9 MW", "6.0-9.9 MW", "10+ MW", "MW unknown"]
SANKEY_SERVICE_ORDER = ["AOM4000", "AOM5000", "Unknown/other service"]
SANKEY_DURATION_ORDER = ["<10 years service", "10-19 years service", "20-29 years service", "30+ years service", "Service length unknown"]


def sankey_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(100,116,139,{alpha})"
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return f"rgba(100,116,139,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def sankey_label(value: Any, max_len: int = 30) -> str:
    text = clean_text(value) or "Unknown"
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "Unknown"
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "..."


def sankey_top_or_other(frame: pd.DataFrame, label_col: str, value_col: str, top_n: int, other_label: str) -> pd.Series:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return pd.Series(dtype=object)
    labels = frame[label_col].fillna("Unknown").astype(str)
    totals = pd.DataFrame({"label": labels, "value": frame[value_col]}).groupby("label", as_index=False)["value"].sum()
    keep = set(totals.sort_values("value", ascending=False).head(top_n)["label"])
    return labels.where(labels.isin(keep), other_label)


def sankey_platform_family(value: Any) -> str:
    text = clean_text(value) or "Unknown"
    match = re.search(r"\bV(\d{2,3})[-/]", text, flags=re.IGNORECASE)
    if match:
        return f"V{match.group(1)} family"
    if text.lower() in {"unknown", "nan", "none"}:
        return "Unknown platform"
    return sankey_label(text, 24)


def sankey_year_band(year: Any) -> str:
    if pd.isna(year):
        return "Unknown year"
    value = int(float(year))
    if value <= 2012:
        return "2008-2012"
    if value <= 2016:
        return "2013-2016"
    if value <= 2020:
        return "2017-2020"
    if value <= 2023:
        return "2021-2023"
    return "2024-2026"


def sankey_rotor_bucket(rotor: Any) -> str:
    if pd.isna(rotor):
        return "Rotor unknown"
    value = float(rotor)
    if value < 100:
        return "<100 m rotor"
    if value < 130:
        return "100-129 m rotor"
    if value < 150:
        return "130-149 m rotor"
    if value < 170:
        return "150-169 m rotor"
    return "170+ m rotor"


def sankey_rating_bucket(rating: Any) -> str:
    if pd.isna(rating):
        return "MW unknown"
    value = float(rating)
    if value < 2.5:
        return "<2.5 MW"
    if value < 4.0:
        return "2.5-3.9 MW"
    if value < 6.0:
        return "4.0-5.9 MW"
    if value < 10.0:
        return "6.0-9.9 MW"
    return "10+ MW"


def sankey_service_group(value: Any) -> str:
    text = (clean_text(value) or "").upper()
    if "AOM4000" in text or "4000" in text:
        return "AOM4000"
    if "AOM5000" in text or "5000" in text:
        return "AOM5000"
    return "Unknown/other service"


def sankey_service_duration(value: Any) -> str:
    if pd.isna(value):
        return "Service length unknown"
    years = float(value)
    if years < 10:
        return "<10 years service"
    if years < 20:
        return "10-19 years service"
    if years < 30:
        return "20-29 years service"
    return "30+ years service"


def sankey_axis_positions(count: int) -> list[float]:
    if count <= 1:
        return [0.5]
    if count == 2:
        return [0.28, 0.72]
    # Keep the last node away from the lower edge; Plotly may otherwise
    # nudge large Sankey nodes and break explicit chronological ordering.
    top, bottom = 0.02, 0.82
    return [top + (bottom - top) * i / (count - 1) for i in range(count)]


def build_sankey_figure(
    frame: pd.DataFrame,
    steps: list[tuple[str, str]],
    value_col: str,
    title: str,
    stage_colors: dict[str, str],
    stage_orders: dict[str, list[str]] | None = None,
    stage_y_positions: dict[str, list[float]] | None = None,
) -> go.Figure | None:
    if frame.empty or value_col not in frame.columns:
        return None

    data = frame.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col])
    data = data[data[value_col] > 0]
    if data.empty:
        return None

    stage_orders = stage_orders or {}
    stage_y_positions = stage_y_positions or {}
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_index: dict[tuple[str, str], int] = {}
    stage_label_order: dict[str, list[str]] = {}
    stage_count = max(1, len(steps) - 1)

    for stage_position, (column, stage) in enumerate(steps):
        if column not in data.columns:
            return None
        labels = data[column].map(sankey_label)
        totals = pd.DataFrame({"label": labels, "value": data[value_col]}).groupby("label", as_index=False)["value"].sum()
        order = stage_orders.get(stage)
        if order:
            order_map = {label: idx for idx, label in enumerate(order)}
            totals["order_rank"] = totals["label"].map(order_map).fillna(len(order) + 1)
            totals = totals.sort_values(["order_rank", "label"])
        else:
            totals = totals.sort_values(["value", "label"], ascending=[False, True])
        ordered_labels = totals["label"].tolist()
        stage_label_order[stage] = ordered_labels
        y_values = stage_y_positions.get(stage, sankey_axis_positions(len(ordered_labels)))
        if len(y_values) < len(ordered_labels):
            y_values = sankey_axis_positions(len(ordered_labels))
        x_value = stage_position / stage_count if stage_count else 0.5
        for y_value, label in zip(y_values, ordered_labels, strict=False):
            key = (stage, label)
            node_index[key] = len(node_labels)
            node_labels.append(label)
            node_colors.append(sankey_rgba(stage_colors.get(stage, SANKEY_COLORS["other"]), 0.94))
            node_x.append(x_value)
            node_y.append(y_value)

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []

    for (src_col, src_stage), (dst_col, dst_stage) in zip(steps[:-1], steps[1:], strict=False):
        pair = data[[src_col, dst_col, value_col]].copy()
        pair[src_col] = pair[src_col].map(sankey_label)
        pair[dst_col] = pair[dst_col].map(sankey_label)
        grouped = pair.groupby([src_col, dst_col], as_index=False)[value_col].sum()
        grouped = grouped[grouped[value_col] > 0]
        src_order = {label: idx for idx, label in enumerate(stage_label_order.get(src_stage, []))}
        dst_order = {label: idx for idx, label in enumerate(stage_label_order.get(dst_stage, []))}
        grouped["src_rank"] = grouped[src_col].map(src_order).fillna(9999)
        grouped["dst_rank"] = grouped[dst_col].map(dst_order).fillna(9999)
        grouped = grouped.sort_values(["src_rank", "dst_rank", value_col], ascending=[True, True, False])

        for row in grouped.itertuples(index=False):
            src_label = getattr(row, src_col)
            dst_label = getattr(row, dst_col)
            value = float(getattr(row, value_col))
            src_key = (src_stage, src_label)
            dst_key = (dst_stage, dst_label)
            if src_key not in node_index or dst_key not in node_index:
                continue
            sources.append(node_index[src_key])
            targets.append(node_index[dst_key])
            values.append(value)
            link_colors.append(sankey_rgba(stage_colors.get(src_stage, SANKEY_COLORS["other"]), 0.28))

    if not values:
        return None

    paper_bg = "#07111F"
    font_color = "#F8FAFC"
    node_line = "rgba(255,255,255,0.55)"

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=16,
                    thickness=20,
                    line=dict(color=node_line, width=0.8),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    hovertemplate="%{source.label} -> %{target.label}<br>%{value:,.0f} MW<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=24, color=font_color)),
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        font=dict(color=font_color, size=15, family="Nunito Sans, Segoe UI, sans-serif"),
        margin=dict(l=14, r=14, t=70, b=16),
        height=760,
    )
    return fig


def render_sankey_flows(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Sankey Flows")
    st.caption("Flow views use the current sidebar filters. Values are MW from announced order data.")

    flow_tabs = st.tabs(["Geography to Platform", "Technology Migration", "Customers and Service"])

    with flow_tabs[0]:
        data = platforms.copy()
        if "slot_mw" not in data.columns or data.empty:
            st.info("No platform MW rows available for the selected filters.")
        else:
            data["slot_mw"] = pd.to_numeric(data["slot_mw"], errors="coerce")
            data = data.dropna(subset=["slot_mw"])
            data = data[data["slot_mw"] > 0].copy()
            if data.empty:
                st.info("No platform MW rows available for the selected filters.")
            else:
                for column in ["continent", "region", "country", "platform"]:
                    data[column] = data[column].fillna("Unknown").replace("", "Unknown")
                data = data[data["continent"] != "Unknown"].copy()
                data["platform_family"] = data["platform"].map(sankey_platform_family)
                data["country_top"] = sankey_top_or_other(data, "country", "slot_mw", 18, "Other countries")
                data["platform_top"] = sankey_top_or_other(data, "platform_family", "slot_mw", 14, "Other platform families")
                fig = build_sankey_figure(
                    data,
                    [
                        ("continent", "continent"),
                        ("region", "region"),
                        ("country_top", "country"),
                        ("platform_top", "platform"),
                    ],
                    "slot_mw",
                    "Vestas Ordered MW Flow: Continent -> Region -> Country -> Platform Family",
                    {
                        "continent": SANKEY_COLORS["continent"],
                        "region": SANKEY_COLORS["region"],
                        "country": SANKEY_COLORS["country"],
                        "platform": SANKEY_COLORS["platform"],
                    },
                )
                if fig is None:
                    st.info("Not enough complete rows for this Sankey view.")
                else:
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"Based on {data['slot_mw'].sum() / 1000:,.1f} GW of platform-linked orders.")

    with flow_tabs[1]:
        data = platforms.copy()
        if "slot_mw" not in data.columns or data.empty:
            st.info("No platform MW rows available for the selected filters.")
        else:
            data["slot_mw"] = pd.to_numeric(data["slot_mw"], errors="coerce")
            data = data.dropna(subset=["slot_mw"])
            data = data[data["slot_mw"] > 0].copy()
            if data.empty:
                st.info("No platform MW rows available for the selected filters.")
            else:
                data["year_band"] = data["order_year"].map(sankey_year_band)
                data["rotor_bucket"] = data["rotor_m"].map(sankey_rotor_bucket)
                data["rating_bucket"] = data["mw_rating"].map(sankey_rating_bucket)
                data["platform_family"] = data["platform"].map(sankey_platform_family)
                data["platform_top"] = sankey_top_or_other(data, "platform_family", "slot_mw", 12, "Other platform families")
                fig = build_sankey_figure(
                    data,
                    [
                        ("year_band", "year"),
                        ("rotor_bucket", "rotor"),
                        ("rating_bucket", "rating"),
                        ("platform_top", "platform"),
                    ],
                    "slot_mw",
                    "Technology Migration: Order Era -> Rotor Class -> MW Rating -> Platform Family",
                    {
                        "year": SANKEY_COLORS["year"],
                        "rotor": SANKEY_COLORS["rotor"],
                        "rating": SANKEY_COLORS["rating"],
                        "platform": SANKEY_COLORS["platform"],
                    },
                    {
                        "year": SANKEY_YEAR_ORDER,
                        "rotor": SANKEY_ROTOR_ORDER,
                        "rating": SANKEY_RATING_ORDER,
                    },
                    {
                        "year": [0.04, 0.23, 0.42, 0.61, 0.80],
                    },
                )
                if fig is None:
                    st.info("Not enough complete rows for this Sankey view.")
                else:
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Year bands are fixed in chronological order from oldest to newest.")

    with flow_tabs[2]:
        data = orders.copy()
        if "size_mw" not in data.columns or data.empty:
            st.info("No order MW rows available for the selected filters.")
        else:
            data["size_mw"] = pd.to_numeric(data["size_mw"], errors="coerce")
            data = data.dropna(subset=["size_mw"])
            data = data[data["size_mw"] > 0].copy()
            data["customer"] = data["customer"].fillna("Unknown").replace("", "Unknown")
            data = data[data["customer"].str.lower() != "unknown"].copy()
            if data.empty:
                st.info("No known-customer rows available for the selected filters.")
            else:
                for column in ["continent", "service_scheme"]:
                    data[column] = data[column].fillna("Unknown").replace("", "Unknown")
                data["continent_top"] = sankey_top_or_other(data, "continent", "size_mw", 6, "Other continents")
                data["customer_top"] = sankey_top_or_other(data, "customer", "size_mw", 16, "Other known customers")
                data["service_group"] = data["service_scheme"].map(sankey_service_group)
                data["service_length_bucket"] = data["service_time_years"].map(sankey_service_duration)
                fig = build_sankey_figure(
                    data,
                    [
                        ("continent_top", "continent"),
                        ("customer_top", "customer"),
                        ("service_group", "service"),
                        ("service_length_bucket", "duration"),
                    ],
                    "size_mw",
                    "Known Customers: Continent -> Customer -> Service Scheme -> Service Length",
                    {
                        "continent": SANKEY_COLORS["continent"],
                        "customer": SANKEY_COLORS["customer"],
                        "service": SANKEY_COLORS["service"],
                        "duration": SANKEY_COLORS["duration"],
                    },
                    {
                        "service": SANKEY_SERVICE_ORDER,
                        "duration": SANKEY_DURATION_ORDER,
                    },
                )
                if fig is None:
                    st.info("Not enough complete rows for this Sankey view.")
                else:
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"Known-customer basis: {data['size_mw'].sum() / 1000:,.1f} GW.")


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

    c5, c6 = st.columns(2)
    with c5:
        scheduled = orders.dropna(subset=["order_year", "delivery_year", "size_mw"]).copy()
        if scheduled.empty:
            st.info("No orders with both order and delivery year for the backlog view.")
        else:
            ordered_per_year = scheduled.groupby("order_year")["size_mw"].sum()
            delivered_per_year = scheduled.groupby("delivery_year")["size_mw"].sum()
            year_axis = range(
                int(min(ordered_per_year.index.min(), delivered_per_year.index.min())),
                int(max(ordered_per_year.index.max(), delivered_per_year.index.max())) + 1,
            )
            backlog = pd.DataFrame(index=list(year_axis))
            backlog["cum_ordered"] = ordered_per_year.reindex(backlog.index).fillna(0.0).cumsum()
            backlog["cum_delivered"] = delivered_per_year.reindex(backlog.index).fillna(0.0).cumsum()
            backlog["open_mw"] = backlog["cum_ordered"] - backlog["cum_delivered"]
            fig_bl = go.Figure()
            fig_bl.add_trace(
                go.Scatter(
                    x=backlog.index,
                    y=backlog["open_mw"],
                    mode="lines+markers",
                    fill="tozeroy",
                    name="Open MW (ordered, not yet delivered)",
                    line=dict(width=3, color="#6D5BD0"),
                    fillcolor="rgba(109,91,208,0.25)",
                )
            )
            fig_bl.update_layout(
                template=plotly_template(),
                title="Implied Delivery Pipeline (orders with a delivery schedule)",
                yaxis_title="MW",
                xaxis_title="Year",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_bl, width="stretch")
            st.caption(
                f"Directional view only: covers the {len(scheduled):,} announced orders "
                "({:.0f}% of MW) where a delivery year is disclosed.".format(
                    100.0 * scheduled["size_mw"].sum() / max(orders["size_mw"].sum(), 1e-9)
                )
            )

    with c6:
        days = pd.to_numeric(orders["delivery_days"], errors="coerce").dropna()
        days = days[days >= 0]
        if days.empty:
            st.info("No delivery-day records for the histogram.")
        else:
            fig_hist = px.histogram(
                days.to_frame("delivery_days"),
                x="delivery_days",
                nbins=40,
                template=plotly_template(),
                title="Delivery Lead-Time Distribution",
                labels={"delivery_days": "Days from order to delivery"},
                height=430,
            )
            median_days = float(days.median())
            fig_hist.add_vline(
                x=median_days,
                line_dash="dash",
                line_color="#D1495B",
                annotation_text=f"Median {median_days:,.0f} d",
                annotation_position="top right",
            )
            fig_hist.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_hist, width="stretch")


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

    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)

    st.sidebar.header("Global Filters")
    y_min, y_max = int(orders["order_year"].min()), int(orders["order_year"].max())
    year_range = st.sidebar.slider("Order year range", y_min, y_max, (y_min, y_max))

    continents = sorted([c for c in orders["continent"].dropna().unique().tolist() if c != "Unknown"])
    regions = sorted([r for r in orders["region"].dropna().unique().tolist() if r != "Unknown"])
    countries = sorted(orders["country"].dropna().unique().tolist())
    services = sorted(orders["service_scheme"].dropna().unique().tolist())
    platform_options = sorted(platforms["platform"].dropna().unique().tolist()) if not platforms.empty else []

    selected_continents = st.sidebar.multiselect("Continents", options=continents, default=[])
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=[])
    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=[])
    selected_services = st.sidebar.multiselect("Service schemes", options=services, default=[])
    selected_platforms = st.sidebar.multiselect("Platforms", options=platform_options, default=[])

    min_order_mw = float(np.nanmin(orders["size_mw"])) if orders["size_mw"].notna().any() else 0.0
    max_order_mw = float(np.nanmax(orders["size_mw"])) if orders["size_mw"].notna().any() else 1.0
    mw_floor = st.sidebar.slider("Minimum order MW", min_value=float(min_order_mw), max_value=float(max_order_mw), value=float(min_order_mw))

    o = orders[
        orders["order_year"].between(year_range[0], year_range[1]) & (orders["size_mw"].fillna(0) >= mw_floor)
    ].copy()

    if selected_continents:
        o = o[o["continent"].isin(selected_continents)]
    if selected_regions:
        o = o[o["region"].isin(selected_regions)]
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
        excel_files = [
            f.name
            for pattern in ("*.xlsx", "*.xlsm", "*.xls")
            for f in Path(".").glob(pattern)
            if not f.name.startswith("~$")
        ]
        if excel_files:
            st.warning(
                "No workbook with a 'Vestas Economy' sheet was found in this folder. "
                f"Excel files detected: {', '.join(sorted(excel_files))}"
            )
        cached = load_data_from_cache_only()
        if cached is None:
            st.error(
                "No valid Vestas workbook found and no parsed cache available "
                "(`data_cache/vestas_parsed_data.pkl`)."
            )
            return
        economy, orders, platforms, unannounced = cached
    else:
        stat = workbook.stat()
        cache_key = f"{workbook.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
        try:
            economy, orders, platforms, unannounced = load_data(cache_key, str(workbook))
        except ValueError as exc:
            cached = load_data_from_cache_only()
            if cached is None:
                st.error(str(exc))
                return
            st.warning(f"{exc}. Falling back to cached parsed dataset.")
            economy, orders, platforms, unannounced = cached

    if STOCK_MONTHLY_FILE.exists():
        stock_stat = STOCK_MONTHLY_FILE.stat()
        stock_cache_key = f"{STOCK_MONTHLY_FILE.resolve()}::{stock_stat.st_mtime_ns}::{stock_stat.st_size}"
    else:
        stock_cache_key = "missing_stock_file"
    stock_monthly = load_stock_data(stock_cache_key, str(STOCK_MONTHLY_FILE))
    if MARKET_MONTHLY_FILE.exists():
        market_stat = MARKET_MONTHLY_FILE.stat()
        market_cache_key = f"{MARKET_MONTHLY_FILE.resolve()}::{market_stat.st_mtime_ns}::{market_stat.st_size}"
    else:
        market_cache_key = "missing_market_file"
    market_monthly = load_market_data(market_cache_key, str(MARKET_MONTHLY_FILE))
    st.caption(latest_data_handled_text(economy, orders, unannounced, stock_monthly, market_monthly))

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
            "Turbine Explorer",
            "Country Analytics",
            "Customer Intelligence",
            "Sankey Flows",
            "Delivery and Capacity",
            "Correlations",
            "Latest News",
            "Latest Patents",
            "Information",
        ]
    )

    with tabs[0]:
        render_overall_economics(economy, stock_monthly, market_monthly)
    with tabs[1]:
        render_yearly_overview(orders_f, platforms_f, unannounced_f)
    with tabs[2]:
        render_quarterly_analytics(orders_f, unannounced_f)
    with tabs[3]:
        render_across_years(orders_f, platforms_f)
    with tabs[4]:
        render_platform_lens(platforms_f)
    with tabs[5]:
        render_turbine_explorer(platforms_f, orders_f)
    with tabs[6]:
        render_country_lens(orders_f, platforms_f)
    with tabs[7]:
        render_customer_intelligence(orders_f, platforms_f)
    with tabs[8]:
        render_sankey_flows(orders_f, platforms_f)
    with tabs[9]:
        render_delivery_capacity(orders_f)
    with tabs[10]:
        render_correlations(orders_f)
    with tabs[11]:
        render_latest_news()
    with tabs[12]:
        render_latest_patents()
    with tabs[13]:
        render_information_page()


if __name__ == "__main__":
    main()
