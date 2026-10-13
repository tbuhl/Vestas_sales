"""One-shot computation of every analytic the intelligence tabs need (so each rerun is cheap)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import economics, outlook, quality, structure
from .common import (
    as_of_date,
    load_installed_base,
    prepare_orders,
    prepare_platforms,
)


@dataclass
class Intel:
    """Precomputed analytics for the full (unfiltered) dataset."""

    as_of: pd.Timestamp
    orders: pd.DataFrame
    platforms: pd.DataFrame
    unannounced: pd.DataFrame
    economy: pd.DataFrame
    recon: pd.DataFrame
    ytd: pd.DataFrame
    ltm: dict[str, Any]
    projection: dict[str, Any]
    qe_overall: dict[str, float]
    qe_by_year: pd.DataFrame
    quarter_share: pd.DataFrame
    month_share: pd.Series
    concentration: pd.DataFrame
    mix: pd.DataFrame
    rating_mix: pd.DataFrame
    leading: pd.DataFrame
    service: pd.DataFrame
    lead_time: pd.DataFrame
    radar: pd.DataFrame
    markets: pd.DataFrame
    econ: pd.DataFrame
    lag_corr: pd.DataFrame
    steel: dict[str, float]
    anomalies: pd.DataFrame
    completeness: pd.DataFrame
    installed_base: dict[str, Any] | None = None
    ib_table: pd.DataFrame | None = None
    fleet: pd.DataFrame | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def build_intel(
    economy: pd.DataFrame,
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    unannounced: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> Intel:
    o = prepare_orders(orders)
    p = prepare_platforms(platforms)
    as_of = as_of_date(o)
    ib = load_installed_base()
    econ_tbl = economics.economics_table(economy, market)
    return Intel(
        as_of=as_of,
        orders=o,
        platforms=p,
        unannounced=unannounced,
        economy=economy,
        recon=outlook.reconcile_intake(o, unannounced, economy),
        ytd=outlook.ytd_table(o, as_of),
        ltm=outlook.ltm_summary(o, as_of),
        projection=outlook.project_full_year(o, unannounced, economy, as_of),
        qe_overall=outlook.quarter_end_overall(o),
        qe_by_year=outlook.quarter_end_share_by_year(o),
        quarter_share=outlook.quarter_share_table(o),
        month_share=outlook.month_in_quarter_share(o),
        concentration=structure.concentration_by_year(o, as_of),
        mix=structure.mix_by_year(p),
        rating_mix=structure.rating_bucket_mix(p),
        leading=structure.leading_platforms(p),
        service=structure.service_by_year(o),
        lead_time=structure.lead_time_by_year(o),
        radar=structure.account_radar(o, as_of),
        markets=structure.market_status(o, as_of),
        econ=econ_tbl,
        lag_corr=economics.lag_correlations(econ_tbl) if not econ_tbl.empty else pd.DataFrame(),
        steel=economics.steel_sensitivity(econ_tbl) if not econ_tbl.empty else {},
        anomalies=quality.anomaly_summary(o, p, unannounced, economy, stock, market),
        completeness=quality.completeness_by_year(o),
        installed_base=ib,
        ib_table=structure.installed_base_table(o, ib) if ib else None,
        fleet=structure.fleet_vintage(ib) if ib else None,
    )
