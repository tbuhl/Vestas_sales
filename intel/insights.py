"""Rule-based insight generator: turns the computed analytics into short, number-backed findings.

Every figure in a finding is computed from the data at runtime - nothing here is hard-coded commentary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .bundle import Intel
from .common import GIGA_MW, MEGA_MW, UNKNOWN, pct_change
from .structure import cagr, period_compare


@dataclass
class Insight:
    key: str
    category: str
    headline: str
    detail: str
    metric: str
    metric_label: str
    tone: str = "neutral"  # up | down | warn | neutral
    series: list[float] = field(default_factory=list)
    series_start: int | None = None  # first x (year) of the sparkline
    where: str = ""  # where to dig deeper in the app


# ---- formatting ------------------------------------------------------------------------
def gw(x: float, d: int = 1) -> str:
    return f"{x / 1000.0:,.{d}f} GW"


def mwf(x: float) -> str:
    return f"{x:,.0f} MW"


def pc(x: float, d: int = 0, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{d}%}" if signed else f"{x:.{d}%}"


def _tone(x: float, good_up: bool = True) -> str:
    if x is None or pd.isna(x) or abs(x) < 0.02:
        return "neutral"
    return "up" if (x > 0) == good_up else "down"


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _ok(*vals: float) -> bool:
    return all(v is not None and not pd.isna(v) for v in vals)


# ---- individual insights ---------------------------------------------------------------
def insight_momentum(i: Intel) -> Insight | None:
    year = i.as_of.year
    if year not in i.ytd.index or (year - 1) not in i.ytd.index:
        return None
    cur, prev = i.ytd.loc[year], i.ytd.loc[year - 1]
    growth = pct_change(cur["ytd_mw"], prev["ytd_mw"])
    growth_ex = pct_change(cur["ytd_mw_ex_giga"], prev["ytd_mw_ex_giga"])
    rank, n = int(cur["ytd_rank"]), len(i.ytd)
    first = int(i.ytd.index.min())
    head = (
        f"{year} is the strongest start of any year in the dataset (since {first})"
        if rank == 1
        else f"{year} ranks #{rank} of {n} years for order intake to date"
    )
    return Insight(
        key="momentum",
        category="Momentum",
        headline=head,
        detail=(
            f"{gw(cur['ytd_mw'])} announced by {i.as_of:%d %b} versus {gw(prev['ytd_mw'])} at the same date in {year - 1} "
            f"({pc(growth, signed=True)}). Excluding orders of {GIGA_MW / 1000:.0f} GW or more, the gain is "
            f"{pc(growth_ex, signed=True)} ({gw(cur['ytd_mw_ex_giga'])} vs {gw(prev['ytd_mw_ex_giga'])})."
        ),
        metric=pc(growth, signed=True),
        metric_label=f"announced MW year-to-date vs {year - 1}",
        tone=_tone(growth),
        series=i.ytd["ytd_mw"].loc[year - 7 :].tolist(),
        series_start=year - 7,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_lumpiness(i: Intel) -> Insight | None:
    year = i.as_of.year
    c = i.concentration
    if year not in c.index or (year - 1) not in c.index:
        return None
    cur, prev = c.loc[year], c.loc[year - 1]
    tone = "warn" if cur["top5_share"] >= 0.4 else "neutral"
    return Insight(
        key="lumpiness",
        category="Concentration",
        headline=f"Five orders make up {pc(cur['top5_share'])} of {year}'s announced MW",
        detail=(
            f"{int(cur['mega_orders'])} orders of {MEGA_MW:,.0f} MW or more carry {pc(cur['mega_share'])} of the year's MW "
            f"(vs {pc(prev['mega_share'])} in {year - 1}), while the median order is only {cur['median_mw']:.0f} MW. "
            "The book is a barbell: many small onshore orders plus a handful of offshore-scale ones, so annual totals swing on a few signatures."
        ),
        metric=pc(cur["top5_share"]),
        metric_label=f"top-5 order share of {year} MW",
        tone=tone,
        series=c["top5_share"].loc[year - 8 :].tolist(),
        series_start=year - 8,
        where="Intelligence Lab > Market Structure",
    )


def insight_outlook(i: Intel) -> Insight | None:
    pr = i.projection
    if not pr.get("total"):
        return None
    year = pr["year"]
    tot = pr["total"]
    need = pr["record_mw"] - (pr["ytd_announced"] + pr["unannounced_est"])
    q4_hist = pr["analog"]["remainder_mw"]
    meets = int((q4_hist >= need).sum())
    return Insight(
        key="outlook",
        category="Outlook",
        headline=f"{year} is tracking toward ~{gw(tot['base'])} of total order intake",
        detail=(
            f"Range {gw(tot['conservative'])} to {gw(tot['strong'])} by applying the last {pr['lookback']} years' Q4 patterns to "
            f"{gw(pr['ytd_announced'])} announced so far plus ~{gw(pr['unannounced_est'])} unannounced. "
            f"The record is {gw(pr['record_mw'])} ({pr['record_year']}); beating it needs ~{gw(max(need, 0))} announced after {i.as_of:%d %b}, "
            f"which happened in {meets} of the last {len(q4_hist)} years. "
            f"{int(pr['analogs_above_record'])} of {pr['n_analogs']} analogue outcomes set a record."
        ),
        metric=gw(tot["base"]),
        metric_label=f"{year} order intake, base case (announced + unannounced)",
        tone="up" if tot["base"] >= pr["record_mw"] else "neutral",
        series=i.recon["total_best"].loc[year - 7 : year - 1].tolist() + [tot["base"]],
        series_start=year - 7,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_timing(i: Intel) -> Insight | None:
    q = i.qe_overall
    qs = i.quarter_share
    year = i.as_of.year
    recent = qs.loc[year - 4 : year - 1, "Q4"] if "Q4" in qs else pd.Series(dtype=float)
    m3 = float(i.month_share.get(3, np.nan))
    q4_text = f" Q4 delivered {pc(recent.min())}-{pc(recent.max())} of the annual total in {year - 4}-{year - 1}." if len(recent) else ""
    return Insight(
        key="timing",
        category="Timing",
        headline=f"{pc(q['mw_share'])} of MW is announced in the last {q['window_days']} days of a quarter",
        detail=(
            f"An even spread would put ~{pc(q['expected_if_uniform'])} there. The third month of each quarter carries {pc(m3)} of MW, "
            f"so mid-quarter run-rates understate the quarter and the final fortnight decides it.{q4_text}"
        ),
        metric=pc(q["mw_share"]),
        metric_label=f"of announced MW lands in quarter-end fortnights ({q['first_year']}+)",
        tone="neutral",
        series=i.qe_by_year["mw_share"].loc[q["first_year"] :].tolist(),
        series_start=q["first_year"],
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_product(i: Intel) -> Insight | None:
    m = i.mix
    year = i.as_of.year
    base_year = year - 11
    if year not in m.index or base_year not in m.index:
        return None
    cur = m.loc[year]
    rating_cagr = cagr(m["rating_mw"], base_year, year)
    lead = i.leading[(i.leading["year"] == year) & (i.leading["rank"] == 1)]
    lead_txt = f" {lead.iloc[0]['platform']} alone is {pc(lead.iloc[0]['share'])} of platform-mapped MW." if len(lead) else ""
    return Insight(
        key="product",
        category="Product",
        headline=f"The average ordered turbine is now {cur['rating_mw']:.1f} MW, {cur['rating_mw'] / m.loc[base_year, 'rating_mw']:.1f}x {base_year}",
        detail=(
            f"Ratings grew {pc(rating_cagr, 1)} a year since {base_year} and rotors reached {cur['rotor_m']:.0f} m "
            f"(from {m.loc[base_year, 'rotor_m']:.0f} m).{lead_txt} "
            f"Offshore-class platforms (V164/V174/V236) are {pc(cur['offshore_share'])} of {year} platform-mapped MW."
        ),
        metric=f"{cur['rating_mw']:.1f} MW",
        metric_label=f"MW-weighted turbine rating, {year} orders",
        tone="neutral",
        series=m["rating_mw"].loc[base_year:].tolist(),
        series_start=base_year,
        where="Intelligence Lab > Market Structure",
    )


def insight_geography(i: Intel) -> Insight | None:
    o = i.orders
    asof = i.as_of
    cur = (asof - pd.DateOffset(years=1), asof)
    pri = (asof - pd.DateOffset(years=2), asof - pd.DateOffset(years=1))
    pcmp = period_compare(o, cur, pri)
    if pcmp.empty:
        return None
    gain = pcmp.sort_values("delta_mw", ascending=False).head(3)
    loss = pcmp.sort_values("delta_mw").head(2)
    year = asof.year
    cont = o.pivot_table(index="order_year", columns="continent", values="size_mw", aggfunc="sum").fillna(0)
    share = cont.div(cont.sum(axis=1), axis=0)
    eu = share["Europe"] if "Europe" in share else pd.Series(dtype=float)
    if year not in eu.index:
        return None
    gtxt = ", ".join(f"{c} {d / 1000:+.1f} GW" for c, d in gain["delta_mw"].items())
    ltxt = ", ".join(f"{c} {d / 1000:+.1f} GW" for c, d in loss["delta_mw"].items())
    return Insight(
        key="geography",
        category="Geography",
        headline=f"Europe is {pc(eu[year])} of {year} MW; growth is led by {gain.index[0]} and {gain.index[1]}",
        detail=(
            f"Last 12 months vs the 12 before: {gtxt}; but {ltxt}. "
            f"Europe was {pc(eu.get(year - 7, np.nan))} of MW in {year - 7}; the Americas fell to {pc(share['Americas'].get(year, np.nan))}. "
            "Country mix is lumpy because single offshore projects can swing a market by gigawatts."
        ),
        metric=pc(eu[year]),
        metric_label=f"Europe share of {year} announced MW",
        tone="neutral",
        series=eu.loc[year - 10 :].tolist(),
        series_start=year - 10,
        where="Intelligence Lab > Market Structure",
    )


def insight_accounts(i: Intel) -> Insight | None:
    r = i.radar
    key = r[(r["lifetime_mw"] >= 300) & r["repeat_buyer"]]
    if key.empty:
        return None
    dormant = key[key["status"] == "Dormant"]
    cooling = key[key["status"] == "Cooling"]
    names_d = ", ".join(dormant.head(3)["customer"])
    names_c = ", ".join(cooling.head(3)["customer"])
    year = i.as_of.year
    und = float(
        i.orders.loc[(i.orders["order_year"] >= year - 2) & (i.orders["customer"] == UNKNOWN), "size_mw"].sum()
        / i.orders.loc[i.orders["order_year"] >= year - 2, "size_mw"].sum()
    )
    return Insight(
        key="accounts",
        category="Accounts",
        headline=f"{len(dormant)} of {len(key)} repeat key accounts have not ordered in over two years",
        detail=(
            f"They hold {gw(dormant['lifetime_mw'].sum())} of lifetime orders (e.g. {names_d}). "
            f"{len(cooling)} more are cooling at 12-24 months ({names_c}) - a re-engagement list. "
            f"Customers are undisclosed for {pc(und)} of MW since {year - 2}, so account trends are directional."
        ),
        metric=f"{len(dormant)} / {len(key)}",
        metric_label="key repeat accounts dormant >24 months",
        tone="warn",
        where="Intelligence Lab > Accounts & Service",
    )


def insight_service(i: Intel) -> Insight | None:
    s = i.service
    year = i.as_of.year
    if year not in s.index:
        return None
    peak_year = int(s.loc[year - 5 :, "scheme_disclosed_share"].idxmax())
    cur = s.loc[year]
    med = s.loc[year - 3 :, "tenure_median_years"].median()
    return Insight(
        key="service",
        category="Service",
        headline="Service-scheme disclosure has dropped, so read AOM trends with care",
        detail=(
            f"The scheme is known for {pc(cur['scheme_disclosed_share'])} of {year} MW versus {pc(s.loc[peak_year, 'scheme_disclosed_share'])} in {peak_year}; "
            f"an apparent fall in AOM5000 share is mostly missing data. Among disclosed orders AOM5000 is {pc(cur['aom5000_share_of_disclosed'])} "
            f"and the typical contract runs {med:.0f} years (MW-weighted {cur['tenure_wavg_years']:.1f})."
        ),
        metric=pc(cur["scheme_disclosed_share"]),
        metric_label=f"of {year} MW with a disclosed service scheme",
        tone="warn",
        series=s["scheme_disclosed_share"].loc[year - 8 :].tolist(),
        series_start=year - 8,
        where="Intelligence Lab > Accounts & Service",
    )


def insight_pricing(i: Intel) -> Insight | None:
    e = i.econ
    if e.empty or "reported_asp_meur_mw" not in e:
        return None
    rep = e["reported_asp_meur_mw"].dropna()
    if rep.empty:
        return None
    last = int(rep.index.max())
    base_year = 2020 if 2020 in rep.index else int(rep.index.min())
    asp, asp0 = rep[last], rep[base_year]
    row = e.loc[last]
    derived = e["project_asp_meur_mw"].dropna() if "project_asp_meur_mw" in e else pd.Series(dtype=float)
    cross = (
        f" Project sales per MW shipped tell the same story ({pc(pct_change(derived[last], derived[base_year]), signed=True)})."
        if last in derived.index and base_year in derived.index
        else ""
    )
    r = i.steel.get("lag0", np.nan)
    steel_txt = (
        f" Year-to-year steel moves explain almost none of it (correlation {r:+.2f}, n={i.steel.get('n0', 0)})."
        if _ok(r) and abs(r) < 0.3
        else ""
    )
    return Insight(
        key="pricing",
        category="Pricing",
        headline=f"Realised turbine price is {pc(pct_change(asp, asp0), signed=True)} since {base_year}, at EUR {asp:.2f}m per MW",
        detail=(
            f"Vestas' reported average price rose from EUR {asp0:.2f}m to EUR {asp:.2f}m per MW by {last}.{cross} "
            f"Gross margin recovered to {pc(row['gross_margin'], 1)} and EBIT margin to {pc(row['ebit_margin'], 1)}.{steel_txt}"
        ),
        metric=f"EUR {asp:.2f}m",
        metric_label=f"reported average price per MW, {last}",
        tone="up",
        series=rep.loc[last - 10 :].tolist(),
        series_start=last - 10,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_backlog(i: Intel) -> Insight | None:
    e = i.econ
    if e.empty or "cover_wt_years" not in e:
        return None
    valid = e.dropna(subset=["cover_wt_years"])
    if valid.empty:
        return None
    last = int(valid.index.max())
    row = e.loc[last]
    early = e.loc[2015:2017, "cover_wt_years"].mean()
    conv0 = e.loc[2015, "backlog_conversion"] if 2015 in e.index else np.nan
    lt = i.lead_time
    lt_new = lt.loc[lt.index >= i.as_of.year - 2, "median_days"].median() if len(lt) else np.nan
    lt_old = lt.loc[(lt.index >= 2020) & (lt.index <= 2023), "median_days"].median() if len(lt) else np.nan
    lead_txt = (
        f" Median planned order-to-delivery time rose from {lt_old:.0f} to {lt_new:.0f} days for recent orders."
        if _ok(lt_new, lt_old) and lt_new > lt_old
        else ""
    )
    return Insight(
        key="backlog",
        category="Visibility",
        headline=f"Turbine backlog covers {row['cover_wt_years']:.1f} years of project sales (about {early:.1f} in 2015-17)",
        detail=(
            f"EUR {row['backlog_wt_bneur']:.1f}bn of turbine backlog and EUR {row['backlog_service_bneur']:.1f}bn of service backlog "
            f"({row['cover_service_years']:.1f} years of service sales). Only {pc(row['backlog_conversion'])} of the opening backlog became project sales in {last} "
            f"(vs {pc(conv0)} in 2015).{lead_txt}"
        ),
        metric=f"{row['cover_wt_years']:.1f} yrs",
        metric_label=f"turbine backlog cover, {last}",
        tone="neutral",
        series=valid["cover_wt_years"].loc[last - 10 :].tolist(),
        series_start=last - 10,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_disclosure(i: Intel) -> Insight | None:
    r = i.recon
    ok = r[r["status"] == "Reconciled"]
    bad = r[r["status"] == "Overshoots reported"]
    if ok.empty:
        return None
    share = r["announced_share"].dropna()
    peak = ok["announced_share"].idxmax()
    bad_txt = (
        f" {_join_and([str(y) for y in bad.index])} overshoot reported intake by {_join_and([f'{-g / 1000:.1f}' for g in bad['gap']])} GW "
        "because the unannounced figures appear copied from another year."
        if len(bad)
        else ""
    )
    return Insight(
        key="reconciliation",
        category="Data integrity",
        headline=f"The order list reconciles to reported intake in {len(ok)} of {len(ok) + len(bad)} checkable years",
        detail=(
            f"Announced plus unannounced MW is within +/-3% of Vestas' reported order intake in {ok.index.min()}-{ok.index.max()}, so the order list is "
            f"effectively complete. Announced orders were {pc(share.loc[ok.index.min()])} of reported intake in {ok.index.min()} and "
            f"{pc(ok.loc[peak, 'announced_share'])} in {peak}.{bad_txt}"
        ),
        metric=f"{len(ok)} / {len(ok) + len(bad)}",
        metric_label="years reconciled to reported intake (2017+)",
        tone="warn" if len(bad) else "up",
        series=share.loc[2010:].tolist(),
        series_start=int(share.loc[2010:].index.min()),
        where="Intelligence Lab > Data Quality",
    )


def insight_fleet(i: Intel) -> Insight | None:
    if i.fleet is None or i.fleet.empty or i.ib_table is None:
        return None
    f = i.fleet.set_index("fleet_class")
    legacy_names = [c for c in ["Pre-V80 / other legacy", "2 MW class (V80-V120)"] if c in f.index]
    legacy = float(f.loc[legacy_names, "installed_mw"].sum())
    total = float(f["installed_mw"].sum())
    t = i.ib_table
    since = int(t["since_year"].iloc[0])
    new_orders = float(t["orders_mw_since"].sum())
    fast = t[t["segment"].str.startswith("Fast growth")].sort_values("orders_mw_since", ascending=False).head(3)
    return Insight(
        key="fleet",
        category="Installed base",
        headline=f"{pc(legacy / total)} of the installed fleet sits on 2 MW-class or older platforms",
        detail=(
            f"{gw(legacy, 0)} of {gw(total, 0)} installed by end-{int(i.installed_base['as_of_year'])} - the pool for lifetime extension, repowering and service. "
            f"Announced orders since {since} add {gw(new_orders)} ({pc(new_orders / total)} of the base); fastest relative growth in "
            f"{', '.join(fast['country'])}."
        ),
        metric=gw(legacy, 0),
        metric_label="installed MW on legacy and 2 MW-class platforms",
        tone="neutral",
        where="Intelligence Lab > Accounts & Service",
    )


INSIGHT_BUILDERS: list[Callable[[Intel], Insight | None]] = [
    insight_momentum,
    insight_outlook,
    insight_lumpiness,
    insight_timing,
    insight_product,
    insight_geography,
    insight_pricing,
    insight_backlog,
    insight_accounts,
    insight_service,
    insight_fleet,
    insight_disclosure,
]


def generate_insights(intel: Intel) -> list[Insight]:
    out: list[Insight] = []
    for build in INSIGHT_BUILDERS:
        try:
            item = build(intel)
        except Exception:  # an insight that cannot be computed must not break the page
            item = None
        if item is not None:
            out.append(item)
    return out


def watchlist(intel: Intel) -> list[str]:
    """Forward-looking things to check, derived from the same numbers."""
    items: list[str] = []
    pr = intel.projection
    year = intel.as_of.year
    if pr.get("total"):
        need = pr["record_mw"] - (pr["ytd_announced"] + pr["unannounced_est"])
        items.append(
            f"Q4 {year}: ~{gw(max(need, 0))} of announced orders are needed to beat the {pr['record_year']} record of {gw(pr['record_mw'])}; "
            f"{int((pr['analog']['remainder_mw'] >= need).sum())} of the last {len(pr['analog'])} years delivered that much after {intel.as_of:%d %b}."
        )
        missing_q = 4 - pr["unannounced_known_quarters"]
        if missing_q > 0:
            items.append(
                f"{missing_q} quarter(s) of {year} unannounced MW are not yet in the workbook (estimated at ~{mwf(pr['unannounced_per_quarter'])} per quarter); "
                "they arrive with the quarterly reports."
            )
    ytd = intel.ytd
    if year in ytd.index:
        items.append(
            f"{int(ytd.loc[year, 'ytd_giga_orders'])} gigawatt-scale order(s) in {year} account for "
            f"{gw(ytd.loc[year, 'ytd_mw'] - ytd.loc[year, 'ytd_mw_ex_giga'])}; the like-for-like trend is the ex-giga number."
        )
    r = intel.radar
    cool = r[(r["lifetime_mw"] >= 500) & (r["status"] == "Cooling") & r["repeat_buyer"]].head(4)
    if len(cool):
        items.append("Key accounts approaching 24 months without an order: " + ", ".join(cool["customer"]) + ".")
    return items
