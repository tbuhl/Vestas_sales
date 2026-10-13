"""Links between order flow and company economics: pricing, backlog cover, conversion, lead/lag."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import eco_series, eco_wide


def economics_table(economy: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annual KPIs derived from the Vestas Economy sheet (index = year)."""
    wide = eco_wide(economy)
    if wide.empty:
        return pd.DataFrame()

    def s(*names: str) -> pd.Series:
        return eco_series(wide, *names)

    t = pd.DataFrame(index=wide.index)
    t["revenue_meur"] = s("revenue (mEUR)")
    t["gross_profit_meur"] = s("gross profit")
    t["ebit_meur"] = s("EBIT")
    t["sales_projects_meur"] = s("sales of projects")
    t["sales_service_meur"] = s("sales of service")
    t["shipped_mw"] = s("Produced and shipped wind turbine (MW)")
    t["order_intake_mw"] = s("Order inkake (MW)", "Order intake (MW)")
    t["order_intake_bneur"] = s("Order intake (bnEUR)")
    t["backlog_wt_bneur"] = s("Order backlog - wind turbines (bnEUR)")
    t["backlog_service_bneur"] = s("Order backlog - service (bnEUR)")
    t["backlog_wt_mw"] = s("Order backlog - windturbines (MW)")
    t["employees"] = s("EOY employees")
    t["reported_asp_meur_mw"] = s("average MW price [mEUR/MW]")

    t["gross_margin"] = t["gross_profit_meur"] / t["revenue_meur"]
    t["ebit_margin"] = t["ebit_meur"] / t["revenue_meur"]
    t["service_share"] = t["sales_service_meur"] / t["revenue_meur"]
    # Realised project price: project sales per MW shipped (a proxy - recognition and shipment timing differ).
    t["project_asp_meur_mw"] = t["sales_projects_meur"] / t["shipped_mw"]
    t["book_to_bill_mw"] = t["order_intake_mw"] / t["shipped_mw"]
    t["cover_wt_years"] = t["backlog_wt_bneur"] * 1000.0 / t["sales_projects_meur"]
    t["cover_service_years"] = t["backlog_service_bneur"] * 1000.0 / t["sales_service_meur"]
    t["cover_mw_years"] = t["backlog_wt_mw"] / t["shipped_mw"]
    # Share of the opening turbine backlog recognised as project sales during the year.
    t["backlog_conversion"] = t["sales_projects_meur"] / t["backlog_wt_bneur"].shift(1).mul(1000.0)
    t["revenue_per_employee_keur"] = t["revenue_meur"] * 1000.0 / t["employees"]

    if market is not None and not market.empty:
        m = market.copy()
        m["year"] = pd.to_datetime(m["date"]).dt.year
        annual = m.groupby("year")[["steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]].mean()
        t["steel_usd_per_ton"] = annual["steel_hrc_usd_per_short_ton"]
        t["copper_usd_per_lb"] = annual["copper_hg_usd_per_lb"]
    return t


def lag_correlations(table: pd.DataFrame, first_year: int = 2005, max_lag: int = 2) -> pd.DataFrame:
    """Correlation of annual growth rates: does order intake lead shipments / revenue?

    Rows are (driver -> outcome) pairs; columns are lags in years (outcome shifted forward by the lag).
    """
    pairs = [
        ("Order intake MW", "order_intake_mw", "Shipped MW", "shipped_mw"),
        ("Order intake MW", "order_intake_mw", "Revenue", "revenue_meur"),
        ("Order intake bnEUR", "order_intake_bneur", "Revenue", "revenue_meur"),
        ("Order intake bnEUR", "order_intake_bneur", "Project sales", "sales_projects_meur"),
    ]
    t = table[table.index >= first_year]
    rows = []
    for driver_label, driver, outcome_label, outcome in pairs:
        if driver not in t or outcome not in t:
            continue
        d = t[driver].pct_change(fill_method=None)
        o = t[outcome].pct_change(fill_method=None)
        row = {"pair": f"{driver_label} -> {outcome_label}"}
        for lag in range(0, max_lag + 1):
            both = pd.concat([d, o.shift(-lag)], axis=1).dropna()
            row[f"lag {lag}y"] = float(both.corr().iloc[0, 1]) if len(both) >= 6 else np.nan
            row[f"n (lag {lag}y)"] = int(len(both))
        rows.append(row)
    return pd.DataFrame(rows)


def steel_sensitivity(table: pd.DataFrame, first_year: int = 2008) -> dict[str, float]:
    """Correlation of annual changes in steel price with changes in realised project price (same year and +1y)."""
    if "steel_usd_per_ton" not in table or "project_asp_meur_mw" not in table:
        return {}
    t = table[table.index >= first_year]
    steel = t["steel_usd_per_ton"].pct_change(fill_method=None)
    asp = t["project_asp_meur_mw"].pct_change(fill_method=None)
    out = {}
    for lag in (0, 1):
        both = pd.concat([steel, asp.shift(-lag)], axis=1).dropna()
        out[f"lag{lag}"] = float(both.corr().iloc[0, 1]) if len(both) >= 6 else float("nan")
        out[f"n{lag}"] = int(len(both))
    return out
