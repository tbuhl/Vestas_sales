"""Charts for the Executive Briefing and the Momentum & Outlook section."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import outlook
from .bundle import Intel
from .common import GIGA_MW
from .theme import style_figure, tokens, year_axis


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------------------
def fig_cumulative_curves(intel: Intel, first_year: int = 2013, height: int = 470, width_hint: int = 540) -> go.Figure:
    """Cumulative announced MW by calendar day: current year against every earlier year, with the Q4 outlook range."""
    t = tokens()
    year = intel.as_of.year
    curves = outlook.cumulative_curves(intel.orders, intel.as_of, first_year)
    past_years = [int(y) for y in sorted(curves["year"].unique()) if y != year]
    finals = curves[curves["year"] != year].groupby("year")["cum_mw"].max()
    label_years = set(finals.nlargest(2).index.tolist()) | {year - 1}

    fig = go.Figure()
    first_gray = True
    for y in past_years:
        c = curves[curves["year"] == y]
        prev = bool(y == year - 1)
        color = t["orange"] if prev else t["neutral"]
        fig.add_trace(
            go.Scatter(
                x=c["plot_date"],
                y=c["cum_mw"],
                mode="lines",
                name=str(y) if prev else f"Earlier years ({first_year}-{year - 2})",
                legendgroup="prev" if prev else "earlier",
                showlegend=prev or first_gray,
                line=dict(color=color, width=2.4 if prev else 1.2),
                opacity=1.0 if prev else 0.55,
                hovertemplate=f"{y}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>",
            )
        )
        if not prev:
            first_gray = False
        if y in label_years:
            fig.add_annotation(
                x=c["plot_date"].iloc[-1], y=c["cum_mw"].iloc[-1], text=str(y), showarrow=False, xanchor="left", xshift=4,
                font=dict(size=10, color=t["muted"]),
            )

    cur = curves[curves["year"] == year]
    fig.add_trace(
        go.Scatter(
            x=cur["plot_date"], y=cur["cum_mw"], mode="lines", name=str(year), line=dict(color=t["blue"], width=3.6),
            hovertemplate=f"{year}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>",
        )
    )
    giga = intel.orders[(intel.orders["order_year"] == year) & (intel.orders["size_mw"] >= GIGA_MW)]
    if len(giga):
        pts = cur.set_index("date").reindex(giga["order_date"].dt.normalize())
        fig.add_trace(
            go.Scatter(
                x=pts["plot_date"], y=pts["cum_mw"], mode="markers", name=f"Orders of {GIGA_MW / 1000:.0f} GW+ in {year}",
                marker=dict(size=11, color=t["blue"], line=dict(color=t["surface"], width=2.5), symbol="diamond"),
                text=[f"{c} - {m:,.0f} MW" for c, m in zip(giga["customer"], giga["size_mw"])],
                hovertemplate="%{text}<br>%{x|%d %b}<extra></extra>",
            )
        )
    pr = intel.projection
    if pr.get("announced"):
        end = pd.Timestamp(year=2024, month=12, day=31)
        ann = pr["announced"]
        fig.add_trace(
            go.Scatter(
                x=[end, end, end], y=[ann["conservative"], ann["base"], ann["strong"]], mode="lines+markers",
                name=f"{year} full-year range (announced)", line=dict(color=t["blue"], width=2),
                marker=dict(size=[6, 11, 6], color=t["blue"], symbol=["line-ew", "diamond-open", "line-ew"], line=dict(color=t["blue"], width=2)),
                hovertemplate="%{y:,.0f} MW<extra>FY scenario</extra>",
            )
        )
        fig.add_annotation(x=end, y=ann["base"], text=f"Base {ann['base'] / 1000:.1f} GW", showarrow=False, xanchor="right", xshift=-12,
                           font=dict(size=11, color=t["ink"]))
    style_figure(fig, title=f"Cumulative announced MW: {year} against every year since {first_year}", height=height, ytitle="Cumulative MW", hovermode="closest", width_hint=width_hint)
    fig.update_xaxes(tickformat="%b", range=[pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-25")])
    return fig


# --------------------------------------------------------------------------------------
def fig_intake_outlook(intel: Intel, first_year: int = 2010, height: int = 470) -> go.Figure:
    """Order intake by year: announced vs unannounced (reported basis), with the current-year projection."""
    t = tokens()
    year = intel.as_of.year
    r = intel.recon.loc[first_year : year - 1]
    r = r[r["reported_oi"].notna()]
    ann = r["announced"]
    unann = (r["total_best"] - r["announced"]).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.index, y=ann, name="Announced orders", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>Announced</extra>"))
    fig.add_trace(go.Bar(x=r.index, y=unann, name="Unannounced (reported intake minus announced)", marker=dict(color=t["orange"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>Unannounced</extra>"))
    pr = intel.projection
    if pr.get("announced"):
        base_ann = pr["announced"]["base"]
        ytd = pr["ytd_announced"]
        fig.add_trace(go.Bar(x=[year], y=[ytd], name=f"{year} announced to date", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                             showlegend=False, hovertemplate=f"{year} announced to date: " + "%{y:,.0f} MW<extra></extra>"))
        fig.add_trace(go.Bar(x=[year], y=[max(base_ann - ytd, 0)], name=f"{year} projected Q4 announced (base)",
                             marker=dict(color=rgba(t["blue"], 0.38), line=dict(color=t["blue"], width=1.5), pattern=dict(shape="/", fgcolor=t["blue"], size=7)),
                             hovertemplate="Projected Q4 announced (base): %{y:,.0f} MW<extra></extra>"))
        fig.add_trace(go.Bar(x=[year], y=[pr["unannounced_est"]], name=f"{year} unannounced (listed + estimated)",
                             marker=dict(color=rgba(t["orange"], 0.45), line=dict(color=t["orange"], width=1.5), pattern=dict(shape="/", fgcolor=t["orange"], size=7)),
                             hovertemplate="Unannounced (listed + estimated): %{y:,.0f} MW<extra></extra>"))
        tot = pr["total"]
        fig.add_trace(go.Scatter(
            x=[year], y=[tot["base"]], mode="markers", name=f"{year} range (low / base / high)",
            marker=dict(symbol="diamond-open", size=11, color=t["ink"], line=dict(width=2, color=t["ink"])),
            error_y=dict(type="data", symmetric=False, array=[tot["strong"] - tot["base"]], arrayminus=[tot["base"] - tot["conservative"]],
                         color=t["ink"], thickness=1.6, width=7),
            hovertemplate=f"Base {tot['base']:,.0f} MW (range {tot['conservative']:,.0f}-{tot['strong']:,.0f})<extra></extra>",
        ))
        if pr.get("record_mw") == pr.get("record_mw"):
            fig.add_hline(y=pr["record_mw"], line=dict(color=t["muted"], width=1.2, dash="dot"),
                          annotation_text=f"Record {pr['record_mw'] / 1000:.1f} GW ({pr['record_year']})", annotation_position="top left",
                          annotation_font=dict(size=11, color=t["ink2"]))
    style_figure(fig, title="Order intake by year: announced vs unannounced, with the outlook", height=height, ytitle="MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig


def fig_ytd_bars(intel: Intel, first_year: int = 2013, height: int = 400) -> go.Figure:
    t = tokens()
    y = intel.ytd.loc[first_year:]
    ex = y["ytd_mw_ex_giga"]
    giga = y["ytd_mw"] - y["ytd_mw_ex_giga"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=y.index, y=ex, name=f"Orders below {GIGA_MW / 1000:.0f} GW", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>below 1 GW</extra>"))
    fig.add_trace(go.Bar(x=y.index, y=giga, name=f"Orders of {GIGA_MW / 1000:.0f} GW or more", marker=dict(color=t["orange"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>1 GW and above</extra>"))
    totals = y["ytd_mw"]
    fig.add_trace(go.Scatter(x=y.index, y=totals, mode="text", text=[f"{v / 1000:.1f}" for v in totals], textposition="top center",
                             textfont=dict(size=11, color=t["ink2"]), showlegend=False, hoverinfo="skip"))
    style_figure(fig, title=f"Announced MW to {intel.as_of:%d %b}, by year (GW labelled)", height=height, ytitle="MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig


# --------------------------------------------------------------------------------------
def fig_intra_quarter(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    prof = outlook.intra_quarter_profile(intel.orders, intel.as_of)
    fig = go.Figure()
    if prof.empty:
        return style_figure(fig, title="Order timing within a quarter", height=height)
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["p25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=rgba(t["blue"], 0.18),
                             name="Middle half of quarters (25th-75th pct)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["uniform"], mode="lines", name="Even spread", line=dict(color=t["neutral"], width=1.6),
                             hovertemplate="Even spread: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=prof["day"], y=prof["mean"], mode="lines", name="Average quarter", line=dict(color=t["blue"], width=3),
                             hovertemplate="Day %{x}: %{y:.0%} of the quarter's MW booked<extra></extra>"))
    fig.add_vrect(x0=92 - 13, x1=92, fillcolor=rgba(t["orange"], 0.12), line_width=0, annotation_text="last 14 days", annotation_position="top left",
                  annotation_font=dict(size=10, color=t["ink2"]))
    n = int(prof["quarters"].iloc[0])
    style_figure(fig, title=f"Within a quarter, MW arrives at the end ({n} quarters since 2013)", height=height, xtitle="Day of quarter", ytitle="Cumulative share of quarter MW",
                 hovermode="x unified", yformat=".0%")
    fig.update_yaxes(range=[0, 1.02])
    return fig


def fig_quarter_end_share(intel: Intel, first_year: int = 2013, height: int = 400) -> go.Figure:
    t = tokens()
    s = intel.qe_by_year.loc[first_year:, "mw_share"]
    exp = intel.qe_overall["expected_if_uniform"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index, y=s, name="Share of year's MW", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=s.index, y=[exp] * len(s), mode="lines", name="Even-spread expectation", line=dict(color=t["neutral"], width=1.6),
                             hovertemplate="Even spread: %{y:.0%}<extra></extra>"))
    style_figure(fig, title=f"Share of each year's MW booked in the last {intel.qe_overall['window_days']} days of a quarter", height=height, ytitle="Share of MW",
                 yformat=".0%", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(range=[0, 0.85])
    return fig


def fig_quarter_heatmap(intel: Intel, first_year: int = 2013, height: int = 400) -> go.Figure:
    t = tokens()
    q = intel.quarter_share.loc[first_year:]
    fig = go.Figure(
        go.Heatmap(
            z=q.to_numpy(), x=list(q.columns), y=[str(i) for i in q.index], colorscale=t["seq"], zmin=0, zmax=0.55,
            text=np.where(q.notna(), (q * 100).round(0).astype("Int64").astype(str) + "%", ""), texttemplate="%{text}",
            textfont=dict(size=11), hovertemplate="%{y} %{x}: %{z:.0%} of the year's announced MW<extra></extra>", colorbar=dict(title="", tickformat=".0%", thickness=10),
            xgap=2, ygap=2,
        )
    )
    style_figure(fig, title="Quarter share of each year's announced MW", height=height, hovermode="closest", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    return fig


# --------------------------------------------------------------------------------------
def fig_backlog_cover(intel: Intel, first_year: int = 2014, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=e["cover_wt_years"], mode="lines+markers", name="Turbine backlog / project sales (EUR basis)", line=dict(color=t["blue"], width=2.4),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra>EUR cover</extra>"))
    fig.add_trace(go.Scatter(x=e.index, y=e["cover_mw_years"], mode="lines+markers", name="Turbine backlog / MW shipped (volume basis)", line=dict(color=t["orange"], width=2.4),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra>MW cover</extra>"))
    style_figure(fig, title="Turbine backlog cover (years of sales)", height=height, ytitle="Years", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_service_cover(intel: Intel, first_year: int = 2014, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:]
    fig = go.Figure(go.Scatter(x=e.index, y=e["cover_service_years"], mode="lines+markers", name="Service backlog / service sales", line=dict(color=t["blue"], width=2.4),
                               marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra></extra>"))
    style_figure(fig, title="Service backlog cover (years of service sales)", height=height, ytitle="Years", hovermode="x unified", legend=False)
    year_axis(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_asp(intel: Intel, first_year: int = 2008, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=e["project_asp_meur_mw"], mode="lines+markers", name="Project sales / MW shipped (derived)", line=dict(color=t["blue"], width=2.4),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: EUR %{y:.2f}m/MW<extra>Derived</extra>"))
    fig.add_trace(go.Scatter(x=e.index, y=e["reported_asp_meur_mw"], mode="lines+markers", name="Reported average MW price", line=dict(color=t["orange"], width=2.4),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: EUR %{y:.2f}m/MW<extra>Reported</extra>"))
    style_figure(fig, title="Realised turbine price (EUR m per MW)", height=height, ytitle="EUR m / MW", hovermode="x unified")
    year_axis(fig)
    return fig


def fig_book_to_bill(intel: Intel, first_year: int = 2008, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.loc[first_year:]
    b = e["book_to_bill_mw"].dropna()
    fig = go.Figure(go.Bar(x=b.index, y=b, name="Order intake / MW shipped", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                           hovertemplate="%{x}: %{y:.2f}x<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=t["ink2"], width=1.4))
    style_figure(fig, title="Book-to-bill (MW): >1 means the backlog is growing", height=height, ytitle="x", legend=False, hovermode="x unified")
    year_axis(fig)
    return fig


def fig_backlog_conversion(intel: Intel, first_year: int = 2008, height: int = 400) -> go.Figure:
    t = tokens()
    c = intel.econ.loc[first_year:, "backlog_conversion"].dropna()
    fig = go.Figure(go.Scatter(x=c.index, y=c, mode="lines+markers", name="Project sales / opening turbine backlog", line=dict(color=t["blue"], width=2.4),
                               marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
    style_figure(fig, title="Share of opening turbine backlog recognised as project sales", height=height, ytitle="Share", yformat=".0%", legend=False, hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig
