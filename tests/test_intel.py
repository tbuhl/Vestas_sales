"""Tests for the intelligence layer.

Each analytic is checked against an independent brute-force calculation on the cached dataset, so the tests keep
passing when the data is refreshed. Tests tagged "data sanity" assert properties of the data, not of the code.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intel import economics, insights, outlook, quality, structure
from intel.common import GIGA_MW, MEGA_MW, UNKNOWN, as_of_date, load_installed_base, platform_family, rating_bucket, same_day

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- helpers
def test_platform_family_and_buckets():
    assert platform_family("V150-4.5MW") == "V150"
    assert platform_family("V236-15.0MW") == "V236"
    assert platform_family("3MW") == UNKNOWN and platform_family(None) == UNKNOWN
    assert rating_bucket(2.0) == "<2.5 MW" and rating_bucket(4.2) == "4.0-5.9 MW" and rating_bucket(15) == "10+ MW"
    assert rating_bucket(np.nan) == UNKNOWN


def test_same_day_handles_leap_day():
    assert same_day(2023, pd.Timestamp("2024-02-29")) == pd.Timestamp("2023-02-28")
    assert same_day(2020, pd.Timestamp("2026-10-01")) == pd.Timestamp("2020-10-01")


def test_prepare_orders_adds_columns(orders):
    for col in ["country_clean", "multi_country", "days_to_quarter_end", "in_quarter_end_window", "is_mega", "is_giga"]:
        assert col in orders
    assert (orders["country_clean"] != "United Kingdom - Offshore").all() and (orders["country_clean"] != "Wales").all()
    assert orders["days_to_quarter_end"].between(0, 92).all()
    assert (orders["is_mega"] == (orders["size_mw"] >= MEGA_MW)).all()


# ---------------------------------------------------------------- reconciliation
def test_reconcile_identities(orders, raw):
    r = outlook.reconcile_intake(orders, raw["unannounced"], raw["economy"])
    assert np.allclose(r["announced"], orders.groupby("order_year")["size_mw"].sum().reindex(r.index).fillna(0))
    has = r["reported_oi"].notna()
    assert np.allclose(r.loc[has, "gap"], (r["reported_oi"] - r["listed_total"])[has])
    assert np.allclose(r.loc[has, "total_best"], r.loc[has, "reported_oi"])
    expected = (r["reported_oi"] - r["announced"])[has]
    expected = expected[expected > 0]
    assert np.allclose(r.loc[expected.index, "implied_unannounced"], expected)


def test_duplicate_unannounced_detected(raw):
    dups = outlook.duplicate_unannounced_years(raw["unannounced"])
    vec = {y: tuple(g.sort_values("quarter")["unannounced_mw"]) for y, g in raw["unannounced"].groupby("sheet_year") if len(g) == 4}
    expected = {y for y, v in vec.items() if sum(v == w for w in vec.values()) > 1}
    assert set(dups) == expected


def test_recent_years_reconcile(orders, raw):
    """data sanity: once unannounced MW is listed, announced + unannounced matches reported intake within 3%."""
    r = outlook.reconcile_intake(orders, raw["unannounced"], raw["economy"])
    recent = r[(r.index >= 2019) & r["reported_oi"].notna()]
    assert len(recent) >= 5
    assert (recent["coverage"].sub(1).abs() <= 0.03).all()


# ---------------------------------------------------------------- momentum
def test_ytd_matches_bruteforce(orders):
    as_of = as_of_date(orders)
    t = outlook.ytd_table(orders, as_of)
    for year in (as_of.year, as_of.year - 1, as_of.year - 3):
        cut = same_day(year, as_of)
        sel = orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]
        assert t.loc[year, "ytd_mw"] == pytest.approx(sel["size_mw"].sum())
        assert t.loc[year, "ytd_orders"] == len(sel)
        assert t.loc[year, "ytd_mw_ex_giga"] == pytest.approx(sel.loc[sel["size_mw"] < GIGA_MW, "size_mw"].sum())
    assert t.loc[as_of.year, "ytd_rank"] >= 1


def test_ltm_windows_do_not_overlap(orders):
    as_of = as_of_date(orders)
    s = outlook.ltm_summary(orders, as_of)
    start = as_of - pd.DateOffset(years=1)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]["size_mw"].sum()
    prev = orders[(orders["order_date"] > as_of - pd.DateOffset(years=2)) & (orders["order_date"] <= start)]["size_mw"].sum()
    assert s["current"]["mw"] == pytest.approx(cur) and s["prior"]["mw"] == pytest.approx(prev)


def test_cumulative_curves_end_at_annual_totals(orders):
    as_of = as_of_date(orders)
    c = outlook.cumulative_curves(orders, as_of, first_year=2019)
    for year in (2019, 2022, as_of.year):
        final = c[c["year"] == year]["cum_mw"].iloc[-1]
        cut = as_of if year == as_of.year else pd.Timestamp(year, 12, 31)
        assert final == pytest.approx(orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]["size_mw"].sum())
    assert c.groupby("year")["cum_mw"].apply(lambda s: s.is_monotonic_increasing).all()


# ---------------------------------------------------------------- projection
def test_projection_structure(orders, raw):
    pr = outlook.project_full_year(orders, raw["unannounced"], raw["economy"], lookback=6)
    assert len(pr["analog"]) == 6
    a, t = pr["announced"], pr["total"]
    assert a["conservative"] <= a["base"] <= a["strong"]
    assert all(t[k] == pytest.approx(a[k] + pr["unannounced_est"]) for k in a)
    assert a["conservative"] >= pr["ytd_announced"]  # remainder MW is never negative
    assert 0 <= pr["analogs_above_record"] <= pr["n_analogs"] == 12
    row = pr["analog"].iloc[0]
    assert row["est_runrate"] == pytest.approx(pr["ytd_announced"] * row["ratio"])
    assert row["est_addon"] == pytest.approx(pr["ytd_announced"] + row["remainder_mw"])


# ---------------------------------------------------------------- timing
def test_quarter_end_shares_are_fractions(orders):
    by = outlook.quarter_end_share_by_year(orders)
    assert by["mw_share"].between(0, 1).all() and by["orders_share"].between(0, 1).all()
    overall = outlook.quarter_end_overall(orders)
    assert overall["mw_share"] > overall["expected_if_uniform"]  # the clustering the dashboard reports
    recent = orders[orders["order_year"] >= 2013]
    brute = recent.loc[recent["days_to_quarter_end"] < 14, "size_mw"].sum() / recent["size_mw"].sum()
    assert overall["mw_share"] == pytest.approx(brute)


def test_intra_quarter_profile_is_a_cdf(orders):
    p = outlook.intra_quarter_profile(orders)
    assert p["mean"].is_monotonic_increasing and p["mean"].iloc[-1] == pytest.approx(1.0)
    assert (p["p25"] <= p["p75"] + 1e-12).all()


def test_quarter_and_month_shares_sum_to_one(orders):
    q = outlook.quarter_share_table(orders)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert outlook.month_in_quarter_share(orders).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- structure
def test_concentration_invariants(orders):
    c = structure.concentration_by_year(orders)
    ordered = (c["top1_share"] <= c["top5_share"] + 1e-12) & (c["top5_share"] <= c["top10_share"] + 1e-12) & (c["top10_share"] <= 1 + 1e-12)
    assert ordered.all()
    assert c["mega_share"].between(0, 1).all()
    year = int(c.index[-1])
    assert c.loc[year, "orders"] == (orders["order_year"] == year).sum()
    assert c.loc[year, "mega_orders"] == ((orders["order_year"] == year) & (orders["size_mw"] >= MEGA_MW)).sum()


def test_period_compare_is_consistent(orders):
    as_of = as_of_date(orders)
    cur = (as_of - pd.DateOffset(years=1), as_of)
    prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
    p = structure.period_compare(orders, cur, prior)
    assert np.allclose(p["delta_mw"], p["mw_current"] - p["mw_prior"])
    assert not set(p.index) & {"EU", "North America", "Unknown", "Eastern Europe"}
    assert p["share_current"].sum() == pytest.approx(1.0)


def test_account_radar_rules(orders):
    as_of = as_of_date(orders)
    r = structure.account_radar(orders, as_of)
    named = orders[orders["customer"] != UNKNOWN]
    assert r["lifetime_mw"].sum() == pytest.approx(named["size_mw"].sum())
    assert (r.loc[r["days_since_last"] <= 365, "status"] == "Active").all()
    assert (r.loc[(r["days_since_last"] > 365) & (r["days_since_last"] <= 730), "status"] == "Cooling").all()
    assert (r.loc[r["days_since_last"] > 730, "status"] == "Dormant").all()
    assert (r.loc[~r["repeat_buyer"], "order_dates"] == 1).all()
    assert np.allclose(r["ltm_delta_mw"], r["ltm_mw"] - r["prior_ltm_mw"])


def test_market_status_covers_all_countries(orders):
    m = structure.market_status(orders)
    assert set(m["status"]) <= {"Active", "Cooling", "Dormant", "Emerging"}
    assert m["lifetime_mw"].sum() == pytest.approx(orders[~orders["multi_country"]]["size_mw"].sum())


def test_mix_and_rating_shares(platforms):
    mix = structure.mix_by_year(platforms)
    assert mix["offshore_share"].dropna().between(0, 1).all()
    rm = structure.rating_bucket_mix(platforms)
    assert np.allclose(rm.sum(axis=1), 1.0)
    assert set(platforms.loc[platforms["offshore_class"], "family"]) <= {"V164", "V174", "V236"}


def test_service_shares_are_bounded(orders):
    s = structure.service_by_year(orders)
    for col in ["scheme_disclosed_share", "aom5000_share_of_disclosed", "tenure_disclosed_share"]:
        assert s[col].dropna().between(0, 1).all()
    assert (s["aom5000_share_of_disclosed"] + s["aom4000_share_of_disclosed"]).dropna().le(1 + 1e-9).all()


def test_installed_base_mapping(orders):
    os.chdir(ROOT)
    ib = load_installed_base()
    if ib is None:
        pytest.skip("installed base file missing")
    t = structure.installed_base_table(orders, ib)
    assert t["installed_mw"].sum() == pytest.approx(sum(c["installed_mw"] for c in ib["countries"]))
    assert "United States" in set(t["country"]) and "USA" not in set(t["country"])
    since = ib["as_of_year"] + 1
    brute = orders[(orders["order_year"] >= since) & ~orders["multi_country"]]["size_mw"].sum()
    assert t["orders_mw_since"].sum() == pytest.approx(brute)
    fleet = structure.fleet_vintage(ib)
    assert fleet["installed_mw"].sum() == pytest.approx(sum(x["installed_mw"] for x in ib["turbine_types"]))
    assert fleet["share"].sum() == pytest.approx(1.0)


def test_entity_profile_matches_direct_sums(orders, platforms):
    prof = structure.entity_profile(orders, platforms, "Country", "Germany")
    sub = orders[orders["country_clean"] == "Germany"]
    assert prof["mw"] == pytest.approx(sub["size_mw"].sum()) and prof["orders"] == len(sub)
    fam = structure.entity_profile(orders, platforms, "Platform family", "V150")
    assert fam["mw"] == pytest.approx(platforms.loc[platforms["family"] == "V150", "slot_mw"].sum())
    yearly = structure.entity_yearly(orders, platforms, "Country", "Germany")
    assert yearly.sum() == pytest.approx(prof["mw"])
    breakdown = structure.entity_breakdown(orders, platforms, "Country", "Germany", "Customer")
    assert breakdown.sum() <= prof["mw"] + 1e-6


# ---------------------------------------------------------------- economics
def test_economics_table_definitions(raw):
    t = economics.economics_table(raw["economy"])
    y = t.dropna(subset=["sales_projects_meur", "shipped_mw"]).index[-1]
    assert t.loc[y, "project_asp_meur_mw"] == pytest.approx(t.loc[y, "sales_projects_meur"] / t.loc[y, "shipped_mw"])
    assert t.loc[y, "gross_margin"] == pytest.approx(t.loc[y, "gross_profit_meur"] / t.loc[y, "revenue_meur"])
    lag = economics.lag_correlations(t)
    values = lag[[c for c in lag.columns if c.startswith("lag")]].stack().dropna()
    assert (values.abs() <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------- quality + insights
def test_quality_flags_known_issues(intel):
    checks = " | ".join(intel.anomalies["Check"])
    assert "Unannounced MW copied across years" in checks
    assert "Identical economics value in consecutive years" in checks
    assert set(intel.anomalies["Severity"]) <= {"High", "Medium", "Low", "OK", "Info"}
    assert intel.anomalies["Affected"].map(type).eq(str).all()  # Arrow-safe column


def test_completeness_is_bounded(intel):
    assert intel.completeness.stack().between(0, 1).all()


def test_anomaly_rows_are_subsets_of_orders(orders):
    for frame in quality.anomaly_rows(orders).values():
        assert set(frame["order_id"]) <= set(orders["order_id"])


def test_insights_generate_and_are_clean(intel):
    items = insights.generate_insights(intel)
    assert len(items) >= 10
    keys = [i.key for i in items]
    assert len(keys) == len(set(keys))
    for i in items:
        text = f"{i.headline} {i.detail} {i.metric} {i.metric_label}"
        assert i.headline and i.detail and i.metric and i.where
        assert not re.search(r"(nan|NaN|None|inf|n/a)", text), (i.key, text)
        assert i.tone in {"up", "down", "warn", "neutral"}
    assert insights.watchlist(intel)


def test_each_insight_builder_runs_without_swallowed_errors(intel):
    """generate_insights() hides exceptions; call each builder directly so a regression cannot hide."""
    for build in insights.INSIGHT_BUILDERS:
        assert build(intel) is not None, build.__name__
