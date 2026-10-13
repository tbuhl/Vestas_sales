"""Data coverage and integrity checks, so users know how far to trust each analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import UNKNOWN, as_of_date
from .outlook import duplicate_unannounced_years, reconcile_intake


FIELD_LABELS = {
    "customer": "Customer named",
    "country": "Specific country",
    "scheme": "Service scheme",
    "tenure": "Service tenure",
    "delivery": "Delivery date",
    "platform": "Platform identified",
}


def completeness_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Share of each year's announced MW for which a field is disclosed (year x field)."""
    rows = []
    for year, g in orders.groupby("order_year"):
        total = g["size_mw"].sum()
        if total <= 0:
            continue

        def share(mask: pd.Series) -> float:
            return float(g.loc[mask, "size_mw"].sum() / total)

        rows.append(
            {
                "year": int(year),
                FIELD_LABELS["customer"]: share(g["customer"] != UNKNOWN),
                FIELD_LABELS["country"]: share(~g["multi_country"]),
                FIELD_LABELS["scheme"]: share(g["service_scheme"] != UNKNOWN),
                FIELD_LABELS["tenure"]: share(g["service_time_years"].notna()),
                FIELD_LABELS["delivery"]: share(g["delivery_year"].notna()),
                FIELD_LABELS["platform"]: share(g["primary_platform"] != UNKNOWN),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _platform_gap_groups(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orders whose platform MW differs from order size by >5%: (5-10% band, >10% band)."""
    rel = (orders["mw_gap"].abs() / orders["size_mw"]).where(orders["size_mw"] > 0)
    flagged = rel > 0.05
    big = orders["mw_gap"].abs() > 1.0
    return orders[flagged & big & (rel <= 0.10)], orders[flagged & big & (rel > 0.10)]


def anomaly_summary(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    unannounced: pd.DataFrame,
    economy: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per check: severity, how many records/years are affected and what it means."""
    as_of = as_of_date(orders)
    rows: list[dict[str, Any]] = []

    def add(check: str, severity: str, affected: Any, detail: str) -> None:
        rows.append({"Check": check, "Severity": severity, "Affected": str(affected), "Detail": detail})

    recon = reconcile_intake(orders, unannounced, economy)
    dups = duplicate_unannounced_years(unannounced)
    if dups:
        years = sorted(dups)
        over = recon.loc[[y for y in years if y in recon.index], :]
        over = over[over["status"] == "Overshoots reported"]
        text = f"Quarterly unannounced MW are identical in {', '.join(map(str, years))}."
        if not over.empty:
            parts = [f"{int(y)} overshoots reported intake by {-r['gap']:,.0f} MW" for y, r in over.iterrows()]
            text += " " + "; ".join(parts) + " - likely a copy/paste in the source workbook."
        add("Unannounced MW copied across years", "High" if not over.empty else "Medium", len(years), text)

    bad = recon[recon["status"].isin(["Overshoots reported", "Below reported"])]
    bad = bad[~bad.index.isin(list(dups))]
    if not bad.empty:
        add(
            "Announced + unannounced vs reported intake",
            "Medium",
            len(bad),
            "Years outside +/-3% of reported order intake: " + ", ".join(f"{y} ({r['coverage']:.0%})" for y, r in bad.iterrows()),
        )
    recon_years = recon[recon["status"] == "Reconciled"].index.tolist()
    if recon_years:
        add(
            "Reconciled years",
            "OK",
            len(recon_years),
            f"Announced + unannounced MW matches reported order intake within +/-3% in {len(recon_years)} years "
            f"({min(recon_years)}-{max(recon_years)}), which supports the completeness of the order list.",
        )

    neg = orders[orders["delivery_days"] < 0]
    if len(neg):
        add("Delivery before order date", "Low", len(neg), f"{len(neg)} orders have a delivery date earlier than the order date (excluded from lead-time medians where noted).")

    small, large = _platform_gap_groups(orders)
    if len(small):
        add(
            "Platform MW differs slightly from order size",
            "Info",
            len(small),
            f"{len(small)} orders ({small['size_mw'].sum():,.0f} MW) have platform-level MW within 5-10% of the order size; net difference {small['mw_gap'].sum():,.0f} MW. "
            "Usually expected: the order is sold at a turbine's nameplate rating (e.g. 4.5 MW) but the site runs a different power mode (e.g. 4.3 MW).",
        )
    if len(large):
        add(
            "Platform MW differs a lot from order size",
            "Medium",
            len(large),
            f"{len(large)} orders ({large['size_mw'].sum():,.0f} MW) have platform-level MW more than 10% away from the order size "
            f"({(large['platform_mw_sum'] / large['size_mw']).min():.0%}-{(large['platform_mw_sum'] / large['size_mw']).max():.0%} of it). "
            "Too large for a power-mode difference: probably an incomplete platform split or an inconsistent entry. Platform-based charts use platform-slot MW, order-based charts use order MW.",
        )

    multi = orders[orders["multi_country"] & (orders["country_clean"] != UNKNOWN)]
    if len(multi):
        labels = ", ".join(sorted(multi["country_clean"].unique()))
        add(
            "Orders without a single country",
            "Low",
            len(multi),
            f"{len(multi)} orders ({multi['size_mw'].sum():,.0f} MW) carry a region label instead of a country ({labels}); "
            "excluded from country-level analytics.",
        )

    twins = orders[orders.duplicated(["order_date", "country", "customer", "size_mw"], keep=False)]
    if len(twins):
        add("Possible duplicate order rows", "Low", len(twins), f"{len(twins)} rows share date, country, customer and size - often twin tranches, but worth a look.")

    recent = orders[orders["order_year"] >= as_of.year - 2]
    for col, label in (("service_scheme", "Service scheme"), ("customer", "Customer")):
        share_unknown = float(recent.loc[recent[col] == UNKNOWN, "size_mw"].sum() / recent["size_mw"].sum())
        full_unknown = float(orders.loc[orders[col] == UNKNOWN, "size_mw"].sum() / orders["size_mw"].sum())
        sev = "High" if share_unknown >= 0.4 else "Medium" if share_unknown >= 0.25 else "Low"
        add(
            f"{label} undisclosed",
            sev,
            f"{share_unknown:.0%} of MW since {as_of.year - 2}",
            f"{label} is unknown for {share_unknown:.0%} of announced MW since {as_of.year - 2} ({full_unknown:.0%} over the full history).",
        )

    if economy is not None and not economy.empty:
        ec = economy.sort_values(["metric", "year"]).reset_index(drop=True)
        repeats = []
        for metric, g in ec.groupby("metric"):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                if g.loc[i, "year"] == g.loc[i - 1, "year"] + 1 and g.loc[i, "value"] == g.loc[i - 1, "value"] and abs(g.loc[i, "value"]) >= 50:
                    repeats.append(f"{metric} {int(g.loc[i - 1, 'year'])}/{int(g.loc[i, 'year'])} = {g.loc[i, 'value']:,.0f}")
        if repeats:
            add(
                "Identical economics value in consecutive years",
                "Medium",
                len(repeats),
                "; ".join(repeats) + ". Possibly copied rather than reported - derived ratios that use these years (e.g. price per MW shipped, book-to-bill) are less reliable.",
            )

    mismatch = orders[orders["sheet_year"] != orders["order_year"]]
    if len(mismatch):
        add("Order filed on a different year's sheet", "Low", len(mismatch), "Order date year differs from the workbook sheet year.")

    for label, frame in (("Stock", stock), ("Steel/copper", market)):
        if frame is not None and not frame.empty:
            last = pd.Timestamp(frame["date"].max())
            months = (as_of.year - last.year) * 12 + as_of.month - last.month
            if months >= 3:
                add(f"{label} series are stale", "Info", f"{months} months", f"{label} data ends {last.date()}; refresh data/*.json to extend overlays.")
    return pd.DataFrame(rows)


def anomaly_rows(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detail tables behind the anomaly checks."""
    cols = ["order_id", "order_date", "country", "customer", "size_mw", "primary_platform"]
    out: dict[str, pd.DataFrame] = {}
    neg = orders[orders["delivery_days"] < 0][cols + ["delivery_days"]]
    if len(neg):
        out["Delivery before order date"] = neg.reset_index(drop=True)
    small, large = _platform_gap_groups(orders)
    for name, frame in (("Platform MW differs a lot from order size", large), ("Platform MW differs slightly from order size", small)):
        if len(frame):
            f = frame[cols + ["platform_mw_sum", "mw_gap"]].copy()
            f["platform / order"] = (f["platform_mw_sum"] / f["size_mw"]).round(2)
            out[name] = f.sort_values("mw_gap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    multi = orders[orders["multi_country"] & (orders["country_clean"] != UNKNOWN)][cols]
    if len(multi):
        out["Orders without a single country"] = multi.reset_index(drop=True)
    twins = orders[orders.duplicated(["order_date", "country", "customer", "size_mw"], keep=False)][cols]
    if len(twins):
        out["Possible duplicate order rows"] = twins.sort_values(["order_date", "country"]).reset_index(drop=True)
    return out
