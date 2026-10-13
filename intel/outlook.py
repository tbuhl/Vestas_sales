"""Momentum, reconciliation, projection and order-timing analytics (pure pandas)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    GIGA_MW,
    QUARTER_END_WINDOW_DAYS,
    as_of_date,
    eco_series,
    eco_wide,
    pct_change,
    same_day,
)

RECONCILED_TOLERANCE = 0.03  # +/-3% between announced+unannounced and reported order intake counts as reconciled


# --------------------------------------------------------------------------------------
# Reconciliation of announced + unannounced MW against Vestas' reported order intake
# --------------------------------------------------------------------------------------
def duplicate_unannounced_years(unannounced: pd.DataFrame) -> dict[int, list[int]]:
    """Years whose four quarterly unannounced values are identical to another year's.

    Returns {year: [other years with the same vector]}.
    """
    if unannounced is None or unannounced.empty:
        return {}
    vectors: dict[int, tuple[float, ...]] = {}
    for year, grp in unannounced.groupby("sheet_year"):
        q = grp.drop_duplicates("quarter").set_index("quarter")["unannounced_mw"].reindex([1, 2, 3, 4])
        if q.notna().all():
            vectors[int(year)] = tuple(float(v) for v in q.to_numpy())
    result: dict[int, list[int]] = {}
    for year, vec in vectors.items():
        twins = [y for y, v in vectors.items() if y != year and v == vec]
        if twins:
            result[year] = sorted(twins)
    return result


def reconcile_intake(orders: pd.DataFrame, unannounced: pd.DataFrame, economy: pd.DataFrame) -> pd.DataFrame:
    """Year-by-year bridge: announced + listed unannounced MW versus reported order intake (MW)."""
    years = list(range(int(orders["order_year"].min()), int(orders["order_year"].max()) + 1))
    out = pd.DataFrame(index=pd.Index(years, name="year"))
    out["announced"] = orders.groupby("order_year")["size_mw"].sum().reindex(years).fillna(0.0)

    if unannounced is not None and not unannounced.empty:
        un = unannounced.drop_duplicates(["sheet_year", "quarter"])
        out["unannounced_listed"] = un.groupby("sheet_year")["unannounced_mw"].sum().reindex(years)
        out["quarters_listed"] = un.groupby("sheet_year")["quarter"].nunique().reindex(years).fillna(0).astype(int)
    else:
        out["unannounced_listed"] = np.nan
        out["quarters_listed"] = 0

    wide = eco_wide(economy)
    reported = eco_series(wide, "Order inkake (MW)", "Order intake (MW)")
    out["reported_oi"] = reported.reindex(years)
    out["listed_total"] = out["announced"] + out["unannounced_listed"].fillna(0.0)
    out["coverage"] = out["listed_total"] / out["reported_oi"]
    out["gap"] = out["reported_oi"] - out["listed_total"]

    dups = duplicate_unannounced_years(unannounced)
    out["unann_copy_suspect"] = [y in dups for y in years]

    def status(row: pd.Series) -> str:
        if pd.isna(row["reported_oi"]):
            return "Not yet reported"
        if row["quarters_listed"] == 0:
            return "Announced only"
        if row["quarters_listed"] < 4:
            return "Partial year"
        if abs(row["coverage"] - 1.0) <= RECONCILED_TOLERANCE:
            return "Reconciled"
        return "Overshoots reported" if row["coverage"] > 1.0 else "Below reported"

    out["status"] = out.apply(status, axis=1)
    implied = out["reported_oi"] - out["announced"]
    out["implied_unannounced"] = implied.where(implied > 0)
    out["total_best"] = out["reported_oi"].where(out["reported_oi"].notna(), out["listed_total"])
    out["announced_share"] = out["announced"] / out["total_best"]
    return out


# --------------------------------------------------------------------------------------
# Momentum: YTD / LTM
# --------------------------------------------------------------------------------------
def ytd_table(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """For each year: MW booked up to the as-of month/day (YTD), with and without giga orders, plus the full year."""
    as_of = as_of or as_of_date(orders)
    rows = []
    for year in range(int(orders["order_year"].min()), as_of.year + 1):
        cut = same_day(year, as_of)
        y = orders[orders["order_year"] == year]
        ytd = y[y["order_date"] <= cut]
        rows.append(
            {
                "year": year,
                "ytd_mw": float(ytd["size_mw"].sum()),
                "ytd_orders": int(len(ytd)),
                "ytd_mw_ex_giga": float(ytd.loc[ytd["size_mw"] < GIGA_MW, "size_mw"].sum()),
                "ytd_giga_orders": int((ytd["size_mw"] >= GIGA_MW).sum()),
                "full_mw": float(y["size_mw"].sum()) if year < as_of.year else np.nan,
                "full_orders": int(len(y)) if year < as_of.year else np.nan,
            }
        )
    t = pd.DataFrame(rows).set_index("year")
    t["ytd_share_of_full"] = t["ytd_mw"] / t["full_mw"]
    t["ytd_rank"] = t["ytd_mw"].rank(ascending=False, method="min").astype(int)
    return t


def ltm_summary(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    """Last-twelve-months versus the twelve months before that."""
    as_of = as_of or as_of_date(orders)
    start = as_of - pd.DateOffset(years=1)
    prior_start = as_of - pd.DateOffset(years=2)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]
    prev = orders[(orders["order_date"] > prior_start) & (orders["order_date"] <= start)]

    def stats(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "mw": float(frame["size_mw"].sum()),
            "orders": int(len(frame)),
            "avg_mw": float(frame["size_mw"].mean()) if len(frame) else np.nan,
            "median_mw": float(frame["size_mw"].median()) if len(frame) else np.nan,
            "mw_ex_giga": float(frame.loc[frame["size_mw"] < GIGA_MW, "size_mw"].sum()),
            "giga_orders": int((frame["size_mw"] >= GIGA_MW).sum()),
            "countries": int(frame.loc[~frame["country_clean"].isin(["Unknown"]), "country_clean"].nunique())
            if "country_clean" in frame
            else int(frame["country"].nunique()),
        }

    c, p = stats(cur), stats(prev)
    return {
        "as_of": as_of,
        "window": (start, as_of),
        "current": c,
        "prior": p,
        "mw_growth": pct_change(c["mw"], p["mw"]),
        "mw_growth_ex_giga": pct_change(c["mw_ex_giga"], p["mw_ex_giga"]),
        "orders_growth": pct_change(c["orders"], p["orders"]),
    }


def cumulative_curves(orders: pd.DataFrame, as_of: pd.Timestamp | None = None, first_year: int = 2013) -> pd.DataFrame:
    """Daily cumulative announced MW per year, plotted on a common (leap-year) calendar axis."""
    as_of = as_of or as_of_date(orders)
    frames = []
    for year in range(first_year, as_of.year + 1):
        end = as_of if year == as_of.year else pd.Timestamp(year=year, month=12, day=31)
        days = pd.date_range(pd.Timestamp(year=year, month=1, day=1), end, freq="D")
        daily = (
            orders[orders["order_year"] == year]
            .groupby(orders["order_date"].dt.normalize())["size_mw"]
            .sum()
            .reindex(days, fill_value=0.0)
        )
        frame = pd.DataFrame({"year": year, "date": days, "cum_mw": daily.cumsum().to_numpy()})
        frame["plot_date"] = frame["date"].map(lambda d: d.replace(year=2024))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Full-year projection
# --------------------------------------------------------------------------------------
def trailing_unannounced_per_quarter(unannounced: pd.DataFrame, n_quarters: int = 8) -> float:
    if unannounced is None or unannounced.empty:
        return float("nan")
    un = unannounced.drop_duplicates(["sheet_year", "quarter"]).sort_values(["sheet_year", "quarter"])
    return float(un["unannounced_mw"].tail(n_quarters).mean())


def project_full_year(
    orders: pd.DataFrame,
    unannounced: pd.DataFrame,
    economy: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    lookback: int = 7,
) -> dict[str, Any]:
    """Analogue-based projection of the current year's announced and total order intake.

    Two transparent methods, each applied to every one of the last ``lookback`` complete years:
      * run-rate scaling:  YTD now / (YTD share of full year in that analogue year)
      * remainder add-on:  YTD now + MW booked after the same date in that analogue year
    The run-rate method amplifies lumpy YTD (a few mega orders inflate it); the add-on method does not.
    """
    as_of = as_of or as_of_date(orders)
    year = as_of.year
    ytd_now = float(orders[(orders["order_year"] == year) & (orders["order_date"] <= as_of)]["size_mw"].sum())

    rows = []
    for yy in range(year - lookback, year):
        yo = orders[orders["order_year"] == yy]
        if yo.empty:
            continue
        cut = same_day(yy, as_of)
        ytd = float(yo[yo["order_date"] <= cut]["size_mw"].sum())
        full = float(yo["size_mw"].sum())
        if ytd <= 0:
            continue
        rows.append(
            {
                "year": yy,
                "ytd_mw": ytd,
                "full_mw": full,
                "remainder_mw": full - ytd,
                "ratio": full / ytd,
                "est_runrate": ytd_now * full / ytd,
                "est_addon": ytd_now + (full - ytd),
            }
        )
    analog = pd.DataFrame(rows)

    # Unannounced MW: use what is listed for the year, fill missing quarters at the trailing average.
    un_year = (
        unannounced[unannounced["sheet_year"] == year].drop_duplicates("quarter")
        if unannounced is not None and not unannounced.empty
        else pd.DataFrame(columns=["quarter", "unannounced_mw"])
    )
    known_q = int(un_year["quarter"].nunique())
    per_q = trailing_unannounced_per_quarter(unannounced)
    un_known = float(un_year["unannounced_mw"].sum())
    un_est = un_known + max(4 - known_q, 0) * (per_q if not np.isnan(per_q) else 0.0)

    result: dict[str, Any] = {
        "year": year,
        "as_of": as_of,
        "ytd_announced": ytd_now,
        "analog": analog,
        "unannounced_known": un_known,
        "unannounced_known_quarters": known_q,
        "unannounced_per_quarter": per_q,
        "unannounced_est": un_est,
        "lookback": lookback,
    }

    if analog.empty:
        result.update({"announced": {}, "total": {}, "record_year": None, "record_mw": float("nan"), "analogs_above_record": float("nan")})
        return result

    announced = {
        "conservative": float(analog["est_addon"].min()),
        "base": float(np.mean([analog["est_runrate"].median(), analog["est_addon"].median()])),
        "strong": float(analog["est_runrate"].max()),
    }
    total = {k: v + un_est for k, v in announced.items()}

    recon = reconcile_intake(orders, unannounced, economy)
    complete = recon[(recon.index < year) & recon["reported_oi"].notna()]
    record_year = int(complete["total_best"].idxmax()) if not complete.empty else None
    record_mw = float(complete["total_best"].max()) if not complete.empty else float("nan")

    outcomes = pd.concat([analog["est_runrate"], analog["est_addon"]]) + un_est
    result.update(
        {
            "announced": announced,
            "total": total,
            "record_year": record_year,
            "record_mw": record_mw,
            "analogs_above_record": float((outcomes > record_mw).sum()),
            "n_analogs": int(len(outcomes)),
            "analog_totals": outcomes.reset_index(drop=True),
        }
    )
    return result


# --------------------------------------------------------------------------------------
# Timing / seasonality
# --------------------------------------------------------------------------------------
def quarter_share_table(orders: pd.DataFrame) -> pd.DataFrame:
    mw = orders.pivot_table(index="order_year", columns="order_quarter", values="size_mw", aggfunc="sum").fillna(0.0)
    mw = mw.reindex(columns=[1, 2, 3, 4], fill_value=0.0)
    share = mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)
    share.columns = [f"Q{c}" for c in share.columns]
    return share


def quarter_end_share_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    """Share of each year's announced MW (and orders) booked in the last N days of a quarter."""
    g = orders.groupby("order_year")
    mw_total = g["size_mw"].sum()
    mw_qe = orders[orders["in_quarter_end_window"]].groupby("order_year")["size_mw"].sum().reindex(mw_total.index).fillna(0.0)
    n_total = g.size()
    n_qe = orders[orders["in_quarter_end_window"]].groupby("order_year").size().reindex(n_total.index).fillna(0)
    out = pd.DataFrame({"mw_share": mw_qe / mw_total, "orders_share": n_qe / n_total})
    out.index.name = "year"
    return out


def quarter_end_overall(orders: pd.DataFrame, first_year: int = 2013) -> dict[str, float]:
    o = orders[orders["order_year"] >= first_year]
    expected = QUARTER_END_WINDOW_DAYS / 91.25
    return {
        "mw_share": float(o.loc[o["in_quarter_end_window"], "size_mw"].sum() / o["size_mw"].sum()),
        "orders_share": float(o["in_quarter_end_window"].mean()),
        "expected_if_uniform": float(expected),
        "window_days": QUARTER_END_WINDOW_DAYS,
        "first_year": first_year,
    }


def intra_quarter_profile(orders: pd.DataFrame, as_of: pd.Timestamp | None = None, first_year: int = 2013) -> pd.DataFrame:
    """Average cumulative share of a quarter's MW by day of quarter (complete quarters only)."""
    as_of = as_of or as_of_date(orders)
    grid = np.arange(0, 92)
    curves = []
    sub = orders[orders["order_year"] >= first_year]
    for (year, quarter), grp in sub.groupby(["order_year", "order_quarter"]):
        q_start = pd.Timestamp(year=int(year), month=3 * int(quarter) - 2, day=1)
        q_end = q_start + pd.offsets.QuarterEnd(0)
        if q_end >= as_of or grp["size_mw"].sum() <= 0:
            continue
        day_idx = (grp["order_date"].dt.normalize() - q_start).dt.days.clip(0, 91)
        daily = np.zeros(92)
        np.add.at(daily, day_idx.to_numpy(), grp["size_mw"].to_numpy())
        curves.append(np.cumsum(daily) / daily.sum())
    if not curves:
        return pd.DataFrame(columns=["day", "mean", "p25", "p75", "uniform"])
    arr = np.vstack(curves)
    return pd.DataFrame(
        {
            "day": grid + 1,
            "mean": arr.mean(axis=0),
            "p25": np.percentile(arr, 25, axis=0),
            "p75": np.percentile(arr, 75, axis=0),
            "uniform": (grid + 1) / 92.0,
            "quarters": len(curves),
        }
    )


def month_in_quarter_share(orders: pd.DataFrame, first_year: int = 2013) -> pd.Series:
    o = orders[orders["order_year"] >= first_year]
    pos = ((o["order_date"].dt.month - 1) % 3) + 1
    mw = o.groupby(pos)["size_mw"].sum()
    return (mw / mw.sum()).rename("share")
