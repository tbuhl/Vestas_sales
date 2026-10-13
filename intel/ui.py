"""Streamlit layout for the Executive Briefing and the Intelligence Lab."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import charts_outlook as co
from . import charts_structure as cs
from . import quality, structure
from .bundle import Intel, build_intel
from .common import UNKNOWN, prepare_orders, prepare_platforms
from .insights import generate_insights, gw, mwf, pc, watchlist
from .theme import finding_card_html, inject_css, show, table_view, tokens

LAB_SECTIONS = ["Momentum & Outlook", "Market Structure", "Accounts & Service", "Compare Lab", "Data Quality"]


def _data_signature(economy, orders, platforms, unannounced, stock, market) -> str:
    """Cheap fingerprint of the inputs, so the cached analytics refresh only when the data changes."""
    ib = Path("data") / "vestas_installed_base.json"
    ib_sig = f"{ib.stat().st_mtime_ns}" if ib.exists() else "no-ib"
    return "|".join(
        [
            str(len(orders)), str(orders["order_date"].max()), f"{orders['size_mw'].sum():.2f}", str(len(platforms)), f"{platforms['slot_mw'].sum():.2f}",
            str(len(economy)), f"{economy['value'].sum():.2f}", str(len(unannounced)), f"{unannounced['unannounced_mw'].sum():.2f}",
            str(None if stock is None else stock.shape), str(None if market is None else market.shape), ib_sig,
        ]
    )


@st.cache_data(show_spinner="Computing the intelligence layer...")
def _cached_intel(signature: str, _economy, _orders, _platforms, _unannounced, _stock, _market) -> Intel:  # noqa: ARG001 - signature drives the cache
    return build_intel(_economy, _orders, _platforms, _unannounced, _stock, _market)


def get_intel(economy, orders, platforms, unannounced, stock=None, market=None) -> Intel:
    return _cached_intel(_data_signature(economy, orders, platforms, unannounced, stock, market), economy, orders, platforms, unannounced, stock, market)


def _scope_note(text: str) -> None:
    st.markdown(f'<div class="intel-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def _csv_button(df: pd.DataFrame, label: str, name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key)


# ======================================================================================
# Executive Briefing
# ======================================================================================
def render_executive_briefing(intel: Intel) -> None:
    inject_css()
    t = tokens()
    year = intel.as_of.year
    pr = intel.projection
    ltm = intel.ltm

    st.markdown(
        f'<div class="intel-hero"><h2>Executive Briefing</h2>'
        f'<p class="intel-sub">Company-level view as of {intel.as_of:%d %B %Y} (latest announced order). Every number below is computed from the dataset - '
        "sidebar filters apply to the other tabs, not to this one.</p></div>",
        unsafe_allow_html=True,
    )

    # ---- KPI strip --------------------------------------------------------------------
    cols = st.columns(6)
    if year in intel.ytd.index and (year - 1) in intel.ytd.index:
        cur, prev = intel.ytd.loc[year, "ytd_mw"], intel.ytd.loc[year - 1, "ytd_mw"]
        cols[0].metric("Announced MW year-to-date", gw(cur), f"{(cur / prev - 1):+.0%} vs same date {year - 1}")
    cols[1].metric("Last 12 months", gw(ltm["current"]["mw"]), f"{ltm['mw_growth']:+.0%} vs prior 12 months")
    if pr.get("total"):
        tot = pr["total"]
        cols[2].metric(f"{year} outlook (base case)", gw(tot["base"]), f"range {tot['conservative'] / 1000:.1f}-{tot['strong'] / 1000:.1f} GW", delta_color="off", delta_arrow="off")
        need = max(pr["record_mw"] - (pr["ytd_announced"] + pr["unannounced_est"]), 0)
        cols[3].metric("Record year intake", gw(pr["record_mw"]), f"{pr['record_year']}; needs ~{need / 1000:.1f} GW more in {year}", delta_color="off", delta_arrow="off")
    if year in intel.mix.index and (year - 1) in intel.mix.index:
        r_cur, r_prev = intel.mix.loc[year, "rating_mw"], intel.mix.loc[year - 1, "rating_mw"]
        cols[4].metric("Avg ordered turbine", f"{r_cur:.1f} MW", f"{r_cur - r_prev:+.1f} MW vs {year - 1}")
    valid = intel.econ.dropna(subset=["backlog_wt_bneur", "backlog_service_bneur"]) if not intel.econ.empty else intel.econ
    if len(valid):
        row = valid.iloc[-1]
        cols[5].metric(f"Order backlog (end-{int(valid.index[-1])})", f"EUR {row['backlog_wt_bneur'] + row['backlog_service_bneur']:.0f}bn",
                       f"{row['cover_wt_years']:.1f} yrs turbine cover", delta_color="off", delta_arrow="off")

    # ---- hero charts --------------------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_cumulative_curves(intel), "brief_cum")
    with c2:
        show(co.fig_intake_outlook(intel), "brief_outlook")
    st.caption(
        "Outlook method: each of the last "
        f"{pr.get('lookback', 7)} complete years is used as an analogue - either scaling year-to-date by that year's full-year/YTD ratio, or adding that year's post-"
        f"{intel.as_of:%d %b} MW - then unannounced MW is added from the workbook (missing quarters at the trailing average). The range is the weakest and strongest analogue; "
        "this is a scenario range, not a forecast."
    )

    # ---- findings -----------------------------------------------------------------------
    st.subheader("Key findings")
    items = generate_insights(intel)
    if items:
        st.markdown('<div class="intel-grid">' + "".join(finding_card_html(i, t) for i in items) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate findings.")

    # ---- watchlist ----------------------------------------------------------------------
    st.subheader("What to watch next")
    wl = watchlist(intel)
    if wl:
        st.markdown('<ul class="intel-watch">' + "".join(f"<li>{html.escape(w)}</li>" for w in wl) + "</ul>", unsafe_allow_html=True)
    with st.expander("How these findings are generated"):
        st.markdown(
            "- Findings are rule-based: each template pulls its numbers from the same tables used on the Intelligence Lab tab, so every claim can be traced to a chart there.\n"
            "- **Year-to-date** compares announced MW up to the same calendar day in each year (as-of date = latest order in the data).\n"
            "- **Outlook** blends two analogue methods (run-rate scaling and Q4 add-on) over recent years; see the Momentum & Outlook section for the working.\n"
            "- **Announced vs unannounced**: unannounced MW comes from quarterly reports; since 2019 announced + unannounced matches reported order intake within about 1%.\n"
            "- Economics (backlog, price, margins) run to the last reported fiscal year and lag the order list."
        )


# ======================================================================================
# Intelligence Lab
# ======================================================================================
def render_intelligence_lab(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    inject_css()
    st.markdown(
        '<div class="intel-hero"><h2>Intelligence Lab</h2><p class="intel-sub">Deeper analytics derived from the order book, reported economics and installed-base data.</p></div>',
        unsafe_allow_html=True,
    )
    choice = None
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Section", LAB_SECTIONS, default=LAB_SECTIONS[0], key="intel_lab_section", label_visibility="collapsed")
    else:  # older Streamlit
        choice = st.radio("Section", LAB_SECTIONS, horizontal=True, key="intel_lab_section", label_visibility="collapsed")
    choice = choice or LAB_SECTIONS[0]

    if choice == "Momentum & Outlook":
        _section_momentum(intel)
    elif choice == "Market Structure":
        _section_structure(intel, orders_f, platforms_f)
    elif choice == "Accounts & Service":
        _section_accounts_service(intel, orders_f, platforms_f)
    elif choice == "Compare Lab":
        _section_compare(intel)
    else:
        _section_quality(intel)


# --------------------------------------------------------------------------------------
def _section_momentum(intel: Intel) -> None:
    _scope_note("Company-level section: uses the full dataset and ignores the sidebar filters.")
    pr = intel.projection
    year = intel.as_of.year

    st.subheader("Order momentum and outlook")
    show(co.fig_cumulative_curves(intel, height=500, width_hint=1150), "lab_cum")
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_intake_outlook(intel), "lab_outlook")
    with c2:
        show(co.fig_ytd_bars(intel), "lab_ytd")
    if pr.get("analog") is not None and len(pr["analog"]):
        with st.expander("Outlook working: the analogue years"):
            a = pr["analog"].copy()
            a.columns = ["Analogue year", "YTD MW at same date", "Full-year MW", "MW after that date", "Full-year / YTD", f"{year} estimate: run-rate scaling", f"{year} estimate: remainder add-on"]
            st.dataframe(a.round(2), width="stretch", hide_index=True)
            st.markdown(
                f"- {year} YTD announced: **{pr['ytd_announced']:,.0f} MW**; unannounced estimate **{pr['unannounced_est']:,.0f} MW** "
                f"({pr['unannounced_known']:,.0f} MW listed for {pr['unannounced_known_quarters']} quarter(s) + missing quarters at {pr['unannounced_per_quarter']:,.0f} MW).\n"
                "- *Conservative* = weakest remainder add-on; *Strong* = strongest run-rate scaling; *Base* = average of the two methods' medians.\n"
                "- Run-rate scaling overstates when year-to-date holds unusually large single orders; the add-on method ignores that. Showing both keeps the range honest."
            )

    st.subheader("Order timing")
    c3, c4 = st.columns(2)
    with c3:
        show(co.fig_intra_quarter(intel), "lab_intraq")
    with c4:
        show(co.fig_quarter_end_share(intel), "lab_qeshare")
    c5, c6 = st.columns(2)
    with c5:
        show(co.fig_quarter_heatmap(intel), "lab_qheat")
    with c6:
        ms = intel.month_share
        st.markdown("**Which month of the quarter?**")
        mt = pd.DataFrame({"Month of quarter": ["1st month", "2nd month", "3rd month"], "Share of announced MW (2013+)": [ms.get(1, np.nan), ms.get(2, np.nan), ms.get(3, np.nan)]})
        mt["Share of announced MW (2013+)"] = mt["Share of announced MW (2013+)"].map(lambda v: pc(v))
        st.dataframe(mt, width="stretch", hide_index=True)
        st.markdown(
            f"Announcements cluster at quarter end: {pc(intel.qe_overall['mw_share'])} of MW lands in the last {intel.qe_overall['window_days']} days of a quarter "
            f"(an even spread would give {pc(intel.qe_overall['expected_if_uniform'])}). Treat mid-quarter run-rates with caution and read quarters only after their last week."
        )
    table_view(intel.qe_by_year.assign(mw_share=lambda d: d["mw_share"].round(3), orders_share=lambda d: d["orders_share"].round(3)), "Data table: quarter-end share by year")

    st.subheader("From orders to earnings")
    if intel.econ.empty:
        st.info("Economy data not available.")
        return
    c7, c8 = st.columns(2)
    with c7:
        show(co.fig_backlog_cover(intel), "lab_cover")
    with c8:
        show(co.fig_service_cover(intel), "lab_scover")
    c9, c10 = st.columns(2)
    with c9:
        show(co.fig_asp(intel), "lab_asp")
        st.caption("The derived series is noisy where shipments and revenue recognition differ (2010) or where source values repeat (2016 and 2017 show the same shipped MW); the reported price is the cleaner line.")
    with c10:
        show(co.fig_book_to_bill(intel), "lab_b2b")
    c11, c12 = st.columns(2)
    with c11:
        show(co.fig_backlog_conversion(intel), "lab_conv")
    with c12:
        st.markdown("**Do orders lead shipments and revenue?** Correlation of annual growth rates (driver this year vs outcome `lag` years later).")
        if not intel.lag_corr.empty:
            lc = intel.lag_corr.copy()
            show_cols = ["pair"] + [c for c in lc.columns if c.startswith("lag")]
            lc = lc[show_cols].round(2)
            st.dataframe(lc, width="stretch", hide_index=True)
            n = int(intel.lag_corr[[c for c in intel.lag_corr.columns if c.startswith("n (")]].min().min())
            st.caption(f"Based on about {n}+ annual observations per cell - indicative only. Correlations near zero mean no reliable year-to-year link.")
        if intel.steel:
            st.caption(
                f"Steel price vs realised project price (annual changes): same year r={intel.steel.get('lag0', np.nan):+.2f} (n={intel.steel.get('n0', 0)}), "
                f"next year r={intel.steel.get('lag1', np.nan):+.2f} (n={intel.steel.get('n1', 0)})."
            )
    econ_view = intel.econ[["revenue_meur", "ebit_margin", "gross_margin", "project_asp_meur_mw", "book_to_bill_mw", "cover_wt_years", "cover_service_years", "backlog_conversion"]].dropna(how="all")
    table_view(econ_view.round(3), "Data table: derived economics by year")


# --------------------------------------------------------------------------------------
def _section_structure(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters (year range, geography, service, platform, minimum MW).")
    o = prepare_orders(orders_f)
    p = prepare_platforms(platforms_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    as_of = intel.as_of

    st.subheader("Order size and lumpiness")
    conc = structure.concentration_by_year(o, as_of)
    mode = st.radio("Size-class view", ["Absolute MW", "Share of MW"], horizontal=True, key="str_size_mode")
    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_top_share(conc), "str_top")
    with c2:
        show(cs.fig_size_class_mix(o, share=(mode == "Share of MW")), "str_size")
    big = structure.largest_orders(o, 15)
    st.markdown("**Largest orders in scope**")
    st.dataframe(big, width="stretch", hide_index=True)
    table_view(
        conc[["orders", "mw", "median_mw", "top1_share", "top5_share", "top10_share", "mega_orders", "mega_share", "giga_orders"]].round(3),
        "Data table: concentration by year",
    )

    st.subheader("Product mix")
    mix = structure.mix_by_year(p)
    rating_mix = structure.rating_bucket_mix(p)
    if mix.empty:
        st.info("No platform-level rows for the current filters.")
    else:
        c3, c4 = st.columns(2)
        with c3:
            show(cs.fig_rating_mix(rating_mix), "str_ratingmix")
        with c4:
            show(cs.fig_offshore_share(mix), "str_offshore")
        show(cs.fig_rating_rotor(mix), "str_ratingrotor")
        lead = structure.leading_platforms(p, 3)
        if not lead.empty:
            lt = lead.pivot(index="year", columns="rank", values="platform").rename(columns=lambda r: f"#{r} platform")
            sh = lead.pivot(index="year", columns="rank", values="share").rename(columns=lambda r: f"#{r} share")
            tbl = pd.concat([lt, sh.map(lambda v: pc(v))], axis=1)
            tbl = tbl[[c for r in (1, 2, 3) for c in (f"#{r} platform", f"#{r} share") if c in tbl.columns]]
            st.markdown("**Leading platforms by order year** (share of platform-mapped MW)")
            st.dataframe(tbl.sort_index(ascending=False), width="stretch")
        st.caption(
            "Offshore-class = V164, V174 and V236 platforms. Orders carry no onshore/offshore flag, so this is a proxy; "
            "earlier offshore models (e.g. V112-3.3) are not captured."
        )

    st.subheader("Geography")
    window = st.radio(
        "Compare", ["Last 12 months vs the 12 before", "Year-to-date vs same date last year", "Last full year vs the year before"], horizontal=True, key="str_geo_window"
    )
    if window.startswith("Last 12"):
        cur = (as_of - pd.DateOffset(years=1), as_of)
        prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
        title = f"Country movers: last 12 months vs prior 12 months (to {as_of:%d %b %Y})"
    elif window.startswith("Year-to-date"):
        cur = (pd.Timestamp(as_of.year - 1, 12, 31), as_of)
        prior = (pd.Timestamp(as_of.year - 2, 12, 31), as_of - pd.DateOffset(years=1))
        title = f"Country movers: {as_of.year} year-to-date vs {as_of.year - 1} at the same date"
    else:
        y = as_of.year - 1
        cur = (pd.Timestamp(y - 1, 12, 31), pd.Timestamp(y, 12, 31))
        prior = (pd.Timestamp(y - 2, 12, 31), pd.Timestamp(y - 1, 12, 31))
        title = f"Country movers: {y} vs {y - 1}"
    pcmp = structure.period_compare(o, cur, prior)
    c5, c6 = st.columns([3, 2])
    with c5:
        if pcmp["delta_mw"].abs().sum() == 0:
            st.info("No country-level orders in these windows for the current filters.")
        else:
            show(cs.fig_diverging(pcmp["delta_mw"], 10, title), "str_movers")
    with c6:
        show(cs.fig_continent_mix(o), "str_continent")
    table_view(pcmp.sort_values("delta_mw", ascending=False).round(3), "Data table: country movers")
    c7, c8 = st.columns(2)
    markets = structure.market_status(o, as_of)
    with c7:
        show(cs.fig_breadth(conc), "str_breadth")
    with c8:
        if not markets.empty:
            show(cs.fig_market_status(markets, 16), "str_markets")
    if not markets.empty:
        counts = markets["status"].value_counts()
        st.caption(
            "Market status by months since last order: Active <= 12, Cooling 13-36, Dormant > 36; Emerging = first order within the last 5 years. "
            + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ". Region labels (EU, North America ...) are excluded; UK-Offshore and Wales are folded into United Kingdom."
        )
        table_view(markets.round(1), "Data table: market lifecycle")


# --------------------------------------------------------------------------------------
def _section_accounts_service(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters. Installed-base figures come from the workbook's end-2023 track record.")
    o = prepare_orders(orders_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    as_of = intel.as_of

    st.subheader("Account radar")
    radar = structure.account_radar(o, as_of)
    if radar.empty:
        st.info("No disclosed customers in the filtered data.")
    else:
        min_mw = st.slider("Minimum lifetime MW for an account to appear", 100, 2000, 300, step=50, key="acc_min_mw")
        key_accounts = radar[(radar["lifetime_mw"] >= min_mw) & radar["repeat_buyer"]]
        summ = structure.account_status_summary(radar, float(min_mw), repeat_only=True)
        m = st.columns(4)
        for col, status in zip(m[:3], ["Active", "Cooling", "Dormant"]):
            n = int(summ.loc[status, "accounts"]) if status in summ.index else 0
            mw = float(summ.loc[status, "lifetime_mw"]) if status in summ.index else 0.0
            col.metric(f"{status} repeat accounts", f"{n}", f"{mw / 1000:.1f} GW lifetime", delta_color="off", delta_arrow="off")
        und = float(o.loc[(o["order_year"] >= as_of.year - 2) & (o["customer"] == UNKNOWN), "size_mw"].sum() / max(o.loc[o["order_year"] >= as_of.year - 2, "size_mw"].sum(), 1))
        m[3].metric(f"MW with undisclosed customer ({as_of.year - 2}+)", pc(und), "limits account analytics", delta_color="off", delta_arrow="off")
        show(cs.fig_account_radar(radar, float(min_mw)), "acc_radar")
        st.caption("Status by time since the last order: Active <= 12 months, Cooling 12-24, Dormant > 24. Bubble size reflects the number of orders; only disclosed customers are included.")

        c1, c2 = st.columns([2, 3])
        with c1:
            delta = radar.set_index("customer")["ltm_delta_mw"]
            show(cs.fig_diverging(delta, 8, "Account movers: last 12 months vs prior 12 months", height=520), "acc_movers")
        with c2:
            st.markdown("**Re-engagement watchlist** - repeat accounts that have gone quiet")
            status_pick = st.multiselect("Status", ["Cooling", "Dormant", "Active"], default=["Cooling", "Dormant"], key="acc_status_pick")
            wl = key_accounts[key_accounts["status"].isin(status_pick)].copy()
            wl = wl.sort_values("lifetime_mw", ascending=False)
            view = wl[["customer", "status", "orders", "lifetime_mw", "last_order", "days_since_last", "median_gap_days", "top_country"]].rename(
                columns={"customer": "Customer", "status": "Status", "orders": "Orders", "lifetime_mw": "Lifetime MW", "last_order": "Last order", "days_since_last": "Days since",
                         "median_gap_days": "Typical gap (days)", "top_country": "Main country"}
            )
            st.dataframe(view.round(0), width="stretch", hide_index=True, height=470)
            _csv_button(view, "Download watchlist (CSV)", "vestas_account_watchlist.csv", "acc_dl")

    st.subheader("Service contracts")
    service = structure.service_by_year(o)
    c3, c4 = st.columns(2)
    with c3:
        show(cs.fig_service_disclosure(service), "svc_disc")
    with c4:
        show(cs.fig_tenure(service), "svc_tenure")
    st.caption("Scheme and tenure are disclosed for only part of recent MW, so shifts in AOM mix mostly reflect reporting; compare disclosed shares, not totals.")
    table_view(service.round(3), "Data table: service by order year")

    st.subheader("Installed base and fleet")
    if intel.installed_base is None or intel.ib_table is None:
        st.info("Installed-base file `data/vestas_installed_base.json` not found. Run `python tools/build_installed_base.py <workbook>` to create it.")
        return
    ibt = structure.installed_base_table(o, intel.installed_base)
    ib_year = int(intel.installed_base["as_of_year"])
    c5, c6 = st.columns(2)
    with c5:
        show(cs.fig_installed_vs_orders(ibt), "ib_bars")
    with c6:
        show(cs.fig_ib_scatter(ibt), "ib_scatter")
    c7, c8 = st.columns([2, 3])
    with c7:
        if intel.fleet is not None and not intel.fleet.empty:
            show(cs.fig_fleet(intel.fleet), "ib_fleet")
    with c8:
        seg = ibt.groupby("segment").agg(Countries=("country", "count"), Installed_MW=("installed_mw", "sum"), Orders_since_MW=("orders_mw_since", "sum")).sort_values("Installed_MW", ascending=False)
        st.markdown("**Country segments**")
        st.dataframe(seg.round(0), width="stretch")
        front = ibt[ibt["segment"].str.startswith("Frontier")]
        if len(front):
            st.markdown("**Frontier markets** (orders but no installed base in the track record): " + ", ".join(f"{r.country} ({r.orders_mw_since:,.0f} MW)" for r in front.itertuples()))
        idle = ibt[ibt["segment"].str.startswith("Installed base, no new")]
        if len(idle):
            st.markdown("**Large installed base, no new orders:** " + ", ".join(f"{r.country} ({r.installed_mw / 1000:.1f} GW)" for r in idle.itertuples()))
    st.caption(
        f"Installed base = Vestas' track record at end-{ib_year} (all vintages). 'Orders since' counts announced orders from {ib_year + 1} onwards. "
        "Older platforms (2 MW class and earlier) make up the lifetime-extension and repowering pool."
    )
    table_view(ibt.round(3), "Data table: installed base vs orders by country")


# --------------------------------------------------------------------------------------
def _section_compare(intel: Intel) -> None:
    _scope_note("Head-to-head comparison uses the full dataset and ignores the sidebar filters.")
    o, p, as_of = intel.orders, intel.platforms, intel.as_of
    st.subheader("Compare Lab")
    dim = st.radio("Compare", structure.COMPARE_DIMENSIONS, horizontal=True, key="cmp_dim")
    options = structure.entity_options(o, p, dim)
    if len(options) < 2:
        st.info("Not enough entities to compare.")
        return
    ca, cb = st.columns(2)
    a = ca.selectbox("Side A", options, index=0, key=f"cmp_a_{dim}")
    b = cb.selectbox("Side B", options, index=1, key=f"cmp_b_{dim}")
    pa = structure.entity_profile(o, p, dim, a, as_of)
    pb = structure.entity_profile(o, p, dim, b, as_of)

    def fmt(profile: dict) -> dict[str, str]:
        return {
            "Announced MW (2008+)": f"{profile['mw']:,.0f}",
            "Share of total MW": pc(profile["share_of_total"], 1),
            "Orders": f"{profile['orders']:,}",
            "Average order (MW)": f"{profile['avg_order_mw']:,.0f}",
            "Largest order (MW)": f"{profile['largest_order_mw']:,.0f}",
            "Last 12 months (MW)": f"{profile['ltm_mw']:,.0f}",
            "Prior 12 months (MW)": f"{profile['prior_ltm_mw']:,.0f}",
            "12-month change": pc(profile["ltm_growth"], 0, signed=True),
            "Avg turbine rating (MW)": "n/a" if pd.isna(profile["avg_rating_mw"]) else f"{profile['avg_rating_mw']:.1f}",
            "Avg rotor (m)": "n/a" if pd.isna(profile["avg_rotor_m"]) else f"{profile['avg_rotor_m']:.0f}",
            "Service contract (years, MW-weighted)": "n/a" if pd.isna(profile["service_tenure_years"]) else f"{profile['service_tenure_years']:.1f}",
            "Service scheme disclosed": pc(profile["scheme_disclosed_share"]),
            "Median planned delivery (days)": "n/a" if pd.isna(profile["median_delivery_days"]) else f"{profile['median_delivery_days']:,.0f}",
            "First order": str(profile["first_order"]),
            "Last order": str(profile["last_order"]),
            "Active order years": f"{profile['active_years']}",
            "Countries": f"{profile['countries']}",
            "Named customers": f"{profile['customers']}",
        }

    table = pd.DataFrame({a: fmt(pa), b: fmt(pb)})
    table.index.name = "Metric"
    st.dataframe(table.reset_index(), width="stretch", height=670, hide_index=True, column_config={"Metric": st.column_config.TextColumn("Metric", width="large")})

    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_compare_yearly(structure.entity_yearly(o, p, dim, a), structure.entity_yearly(o, p, dim, b)), "cmp_yearly")
    with c2:
        by_options = [x for x in ["Country", "Customer", "Platform family"] if x != dim]
        by = st.selectbox("Break down by", by_options, index=0, key=f"cmp_by_{dim}")
        show(cs.fig_compare_mix(structure.entity_breakdown(o, p, dim, a, by), structure.entity_breakdown(o, p, dim, b, by), a, b, by), "cmp_mix")
    st.caption("Platform-family comparisons count platform-slot MW; other comparisons count order MW. 'Other' groups the tail beyond the top eight.")


# --------------------------------------------------------------------------------------
SEVERITY_LABEL = {"High": "▲ High", "Medium": "● Medium", "Low": "○ Low", "OK": "✓ OK", "Info": "i Info"}


def _section_quality(intel: Intel) -> None:
    _scope_note("Data-quality section: full dataset, ignoring the sidebar filters.")
    st.subheader("Data quality and reconciliation")
    summary = intel.anomalies.copy()
    n_high = int((summary["Severity"] == "High").sum())
    n_med = int((summary["Severity"] == "Medium").sum())
    n_ok = int((summary["Severity"] == "OK").sum())
    m = st.columns(4)
    m[0].metric("High-severity checks", n_high, delta_color="off", delta_arrow="off")
    m[1].metric("Medium", n_med, delta_color="off", delta_arrow="off")
    m[2].metric("Passed checks", n_ok, delta_color="off", delta_arrow="off")
    rec = intel.recon
    ok_years = int((rec["status"] == "Reconciled").sum())
    m[3].metric("Years reconciled to reported intake", f"{ok_years}", delta_color="off", delta_arrow="off")

    summary["Severity"] = summary["Severity"].map(lambda s: SEVERITY_LABEL.get(s, s))
    st.table(summary.set_index("Check"))

    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_reconciliation(intel.recon), "dq_recon")
    with c2:
        show(cs.fig_completeness_heatmap(intel.completeness), "dq_complete")
    view = intel.recon.loc[2010:, ["announced", "unannounced_listed", "listed_total", "reported_oi", "coverage", "gap", "implied_unannounced", "status"]].copy()
    view["coverage"] = view["coverage"].map(lambda v: "-" if pd.isna(v) else pc(v, 1))
    for col in ["announced", "unannounced_listed", "listed_total", "reported_oi", "gap", "implied_unannounced"]:
        view[col] = view[col].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
    st.markdown("**Reconciliation by year**")
    st.dataframe(view, width="stretch")
    st.caption(
        "Reported intake before 2017 has no unannounced MW in the workbook, so the 'implied unannounced' column (reported minus announced) reconstructs it. "
        "Since 2017 announced + unannounced MW reconciles to reported intake within about 1% in every year (the earlier 2017/2018 copy-paste error in the workbook has been corrected). "
        "Reconciliation runs automatically: any year outside +/-3%, or with unannounced values repeated from another year, is flagged in the checks above."
    )

    details = quality.anomaly_rows(intel.orders)
    if details:
        st.markdown("**Record-level detail**")
        for name, frame in details.items():
            with st.expander(f"{name} ({len(frame)})"):
                st.dataframe(frame, width="stretch", hide_index=True)

    with st.expander("Definitions and method notes"):
        st.markdown(
            "- **As-of date**: the latest order date in the data. Year-to-date comparisons use the same month and day in earlier years.\n"
            "- **Announced MW**: order sizes from the `OI YYYY` sheets. **Unannounced MW**: quarterly figures from the same sheets (2017+); earlier years are reconstructed as reported intake minus announced.\n"
            "- **Quarter-end window**: the last 14 days of a calendar quarter; the even-spread benchmark is 14 / 91.25 = 15%.\n"
            "- **Giga / mega orders**: 1,000 MW and above / 500 MW and above.\n"
            "- **Offshore-class**: V164, V174 and V236 platforms (a proxy - orders carry no onshore/offshore flag).\n"
            "- **Key account**: a disclosed customer with at least 2 order dates and the chosen minimum lifetime MW. Customers are grouped using the app's alias table.\n"
            "- **Countries**: region labels (EU, North America, Eastern Europe, United States and Canada) are excluded from country analytics; 'United Kingdom - Offshore' and 'Wales' are folded into United Kingdom.\n"
            "- **Project price**: project sales divided by MW shipped in the same year (recognition and shipment timing differ, so it is a proxy).\n"
            "- **Backlog cover**: backlog divided by the same year's sales; **backlog conversion**: project sales divided by the previous year-end turbine backlog.\n"
            "- **Installed base**: the workbook's end-2023 track record by country and turbine type."
        )
