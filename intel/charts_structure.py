"""Charts for Market Structure, Accounts & Service, Installed Base, Compare Lab and Data Quality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts_outlook import rgba
from .common import RATING_BUCKET_ORDER
from .theme import continent_colors, service_colors, status_colors, style_figure, tokens, year_axis

SIZE_CLASSES = [
    ("<50 MW", 0, 50),
    ("50-149 MW", 50, 150),
    ("150-499 MW", 150, 500),
    ("500-999 MW", 500, 1000),
    ("1 GW +", 1000, 1e9),
]


# ---- concentration / size ----------------------------------------------------------------
def fig_top_share(conc: pd.DataFrame, first_year: int = 2010, height: int = 400) -> go.Figure:
    t = tokens()
    c = conc.loc[first_year:]
    fig = go.Figure()
    for col, name, color in (("top1_share", "Largest order", t["aqua"]), ("top5_share", "Top 5 orders", t["orange"]), ("top10_share", "Top 10 orders", t["blue"])):
        fig.add_trace(go.Scatter(x=c.index, y=c[col], mode="lines+markers", name=name, line=dict(color=color, width=2.4),
                                 marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.0%}<extra>" + name + "</extra>"))
        fig.add_annotation(x=c.index[-1], y=c[col].iloc[-1], text=f"{c[col].iloc[-1]:.0%}", showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=t["ink"]))
    style_figure(fig, title="How much of a year's MW comes from its largest orders", height=height, ytitle="Share of annual announced MW", yformat=".0%", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_size_class_mix(orders: pd.DataFrame, share: bool = False, first_year: int = 2010, height: int = 420) -> go.Figure:
    t = tokens()
    o = orders[orders["order_year"] >= first_year]
    fig = go.Figure()
    for (label, lo, hi), color in zip(SIZE_CLASSES, t["ramp"]):
        sub = o[(o["size_mw"] >= lo) & (o["size_mw"] < hi)]
        mw = sub.groupby("order_year")["size_mw"].sum().reindex(sorted(o["order_year"].unique()), fill_value=0.0)
        fig.add_trace(go.Bar(x=mw.index, y=mw, name=label, marker=dict(color=color, line=dict(color=t["surface"], width=2)),
                             hovertemplate="%{x}: %{y:,.0f} MW<extra>" + label + "</extra>"))
    style_figure(fig, title="Announced MW by order-size class", height=height, ytitle="Share of MW" if share else "MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    if share:
        fig.update_layout(barnorm="percent")
        fig.update_yaxes(ticksuffix="%")
    year_axis(fig)
    return fig


# ---- product mix ---------------------------------------------------------------------------
def fig_rating_mix(rating_mix: pd.DataFrame, first_year: int = 2012, height: int = 420) -> go.Figure:
    t = tokens()
    r = rating_mix.loc[first_year:]
    fig = go.Figure()
    for bucket, color in zip(RATING_BUCKET_ORDER, t["ramp"]):
        fig.add_trace(go.Bar(x=r.index, y=r[bucket], name=bucket, marker=dict(color=color, line=dict(color=t["surface"], width=2)),
                             hovertemplate="%{x}: %{y:.0%}<extra>" + bucket + "</extra>"))
    style_figure(fig, title="Turbine size mix of ordered MW (share by rating class)", height=height, ytitle="Share of platform-mapped MW", yformat=".0%", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    fig.update_yaxes(range=[0, 1.0])
    return fig


def fig_rating_rotor(mix: pd.DataFrame, first_year: int = 2008, height: int = 380) -> go.Figure:
    t = tokens()
    m = mix.loc[first_year:]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MW-weighted turbine rating (MW)", "MW-weighted rotor diameter (m)"), horizontal_spacing=0.09)
    for col_idx, (col, unit) in enumerate((("rating_mw", "MW"), ("rotor_m", "m")), start=1):
        fig.add_trace(go.Scatter(x=m.index, y=m[col], mode="lines+markers", name=col, line=dict(color=t["blue"], width=2.4), marker=dict(size=7, line=dict(color=t["surface"], width=2)),
                                 hovertemplate="%{x}: %{y:.1f} " + unit + "<extra></extra>"), row=1, col=col_idx)
    style_figure(fig, height=height, legend=False, hovermode="x unified")
    fig.update_layout(margin=dict(l=10, r=14, t=48, b=10))
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=t["ink"])
        ann.xanchor = "left"
        ann.x = ann.x - 0.2
    fig.update_xaxes(dtick=2)
    return fig


def fig_offshore_share(mix: pd.DataFrame, first_year: int = 2015, height: int = 380) -> go.Figure:
    t = tokens()
    m = mix.loc[first_year:]
    fig = go.Figure(go.Bar(x=m.index, y=m["offshore_share"], name="Offshore-class share", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                           customdata=m["offshore_mw"], hovertemplate="%{x}: %{y:.0%} (%{customdata:,.0f} MW)<extra></extra>"))
    style_figure(fig, title="Offshore-class platforms (V164 / V174 / V236) share of platform-mapped MW", height=height, ytitle="Share", yformat=".0%", legend=False, hovermode="x unified")
    fig.update_layout(bargap=0.28)
    year_axis(fig)
    return fig


# ---- geography -------------------------------------------------------------------------------
def fig_diverging(delta: pd.Series, n: int, title: str, unit: str = "MW", height: int | None = None, gain_label: str = "Gained", loss_label: str = "Lost") -> go.Figure:
    t = tokens()
    s = delta[delta != 0].sort_values()
    losers, gainers = s[s < 0].head(n), s[s > 0].tail(n)
    s = pd.concat([losers, gainers])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=gainers.index, x=gainers, orientation="h", name=gain_label, marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         text=[f"{v:+,.0f}" for v in gainers], textposition="outside", textfont=dict(size=11, color=t["ink2"]), cliponaxis=False,
                         hovertemplate="%{y}: %{x:+,.0f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Bar(y=losers.index, x=losers, orientation="h", name=loss_label, marker=dict(color=t["red"], line=dict(color=t["surface"], width=2)),
                         text=[f"{v:+,.0f}" for v in losers], textposition="outside", textfont=dict(size=11, color=t["ink2"]), cliponaxis=False,
                         hovertemplate="%{y}: %{x:+,.0f} " + unit + "<extra></extra>"))
    style_figure(fig, title=title, height=height or max(320, 26 * len(s) + 110), xtitle=f"Change in {unit}", hovermode="closest")
    fig.update_layout(barmode="relative", bargap=0.3)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=list(s.index))
    lo, hi = min(float(s.min()), 0.0), max(float(s.max()), 0.0)
    pad = 0.24 * max(hi - lo, 1.0)
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=True, zerolinecolor=t["axis"], range=[lo - pad, hi + pad])
    return fig


def fig_breadth(conc: pd.DataFrame, first_year: int = 2010, height: int = 360) -> go.Figure:
    t = tokens()
    c = conc.loc[first_year:]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Effective number of countries (1 / HHI)", "Largest country's share of MW"), horizontal_spacing=0.09)
    fig.add_trace(go.Scatter(x=c.index, y=c["effective_countries"], mode="lines+markers", line=dict(color=t["blue"], width=2.4), marker=dict(size=7, line=dict(color=t["surface"], width=2)),
                             name="Effective countries", hovertemplate="%{x}: %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c.index, y=c["top_country_share"], mode="lines+markers", line=dict(color=t["blue"], width=2.4), marker=dict(size=7, line=dict(color=t["surface"], width=2)),
                             name="Top-country share", customdata=c["top_country"], hovertemplate="%{x}: %{y:.0%} (%{customdata})<extra></extra>"), row=1, col=2)
    style_figure(fig, height=height, legend=False, hovermode="x unified")
    fig.update_layout(margin=dict(l=10, r=14, t=48, b=10))
    fig.update_yaxes(tickformat=".0%", row=1, col=2, rangemode="tozero")
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color=t["ink"])
        ann.xanchor = "left"
        ann.x = ann.x - 0.2
    return fig


def fig_market_status(markets: pd.DataFrame, top: int = 20, height: int | None = None) -> go.Figure:
    t = tokens()
    m = markets.head(top).iloc[::-1]
    colors = status_colors()
    fig = go.Figure()
    for status in ["Active", "Cooling", "Dormant", "Emerging"]:
        sub = m[m["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(y=sub["country"], x=sub["lifetime_mw"], orientation="h", name=status, marker=dict(color=colors[status], line=dict(color=t["surface"], width=2)),
                             customdata=np.stack([sub["last_order"].astype(str), sub["orders"], sub["ltm_mw"]], axis=1),
                             hovertemplate="%{y}: %{x:,.0f} MW lifetime<br>last order %{customdata[0]}, %{customdata[1]} orders<br>last 12 months %{customdata[2]:,.0f} MW<extra>" + status + "</extra>"))
    style_figure(fig, title=f"Market lifecycle: top {top} markets by lifetime MW", height=height or max(380, 24 * len(m) + 120), xtitle="Lifetime announced MW (2008+)", hovermode="closest")
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=list(m["country"]))
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"])
    return fig


def fig_continent_mix(orders: pd.DataFrame, first_year: int = 2010, height: int = 400) -> go.Figure:
    t = tokens()
    mw = orders[orders["order_year"] >= first_year].pivot_table(index="order_year", columns="continent", values="size_mw", aggfunc="sum").fillna(0.0)
    share = mw.div(mw.sum(axis=1).replace(0, np.nan), axis=0)
    colors = continent_colors()
    fig = go.Figure()
    for cont in [c for c in ["Europe", "Americas", "Asia", "Oceania", "Africa", "Unknown"] if c in share.columns]:
        fig.add_trace(go.Scatter(x=share.index, y=share[cont], mode="lines", name=cont, stackgroup="one", line=dict(color=colors[cont], width=1.2),
                                 fillcolor=rgba(colors[cont], 0.78), hovertemplate="%{x}: %{y:.0%}<extra>" + cont + "</extra>"))
    style_figure(fig, title="Continent share of announced MW", height=height, ytitle="Share of MW", yformat=".0%", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(range=[0, 1])
    return fig


# ---- accounts ----------------------------------------------------------------------------------
def fig_account_radar(radar: pd.DataFrame, min_lifetime_mw: float = 300.0, height: int = 520) -> go.Figure:
    t = tokens()
    r = radar[(radar["lifetime_mw"] >= min_lifetime_mw)].copy()
    r["years_since"] = r["days_since_last"] / 365.25
    colors = status_colors()
    fig = go.Figure()
    for status in ["Active", "Cooling", "Dormant"]:
        sub = r[r["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["years_since"], y=sub["lifetime_mw"], mode="markers", name=f"{status} ({len(sub)})",
            marker=dict(color=colors[status], size=np.clip(np.sqrt(sub["orders"]) * 5 + 6, 8, 34), line=dict(color=t["surface"], width=2), opacity=0.9),
            customdata=np.stack([sub["customer"], sub["orders"], sub["last_order"].astype(str), sub["ltm_mw"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{y:,.0f} MW lifetime, %{customdata[1]} orders<br>last order %{customdata[2]} (%{x:.1f} yrs ago)<br>last 12 months %{customdata[3]:,.0f} MW<extra></extra>",
        ))
    top = r.nlargest(11, "lifetime_mw")
    for _, row in top.iterrows():
        fig.add_annotation(x=row["years_since"], y=np.log10(row["lifetime_mw"]), text=row["customer"], showarrow=False, yshift=14, font=dict(size=10, color=t["ink2"]))
    for x in (1, 2):
        fig.add_vline(x=x, line=dict(color=t["axis"], width=1))
    for x, label in ((1, "12 months"), (2, "24 months")):
        fig.add_annotation(x=x, y=0, yref="paper", text=label, showarrow=False, xanchor="left", xshift=4, yshift=-14, font=dict(size=10, color=t["ink2"]))
    style_figure(fig, title=f"Account radar: accounts with {min_lifetime_mw:,.0f}+ MW lifetime (bubble = number of orders)", height=height,
                 xtitle="Years since last order", ytitle="Lifetime MW (log scale)", hovermode="closest", legend_position="bottom")
    ticks = [v for v in (300, 500, 1000, 2000, 5000) if v >= r["lifetime_mw"].min() * 0.8 and v <= r["lifetime_mw"].max() * 1.3]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{v:,}" for v in ticks])
    fig.update_xaxes(rangemode="tozero")
    return fig


# ---- service -----------------------------------------------------------------------------------
def fig_service_disclosure(service: pd.DataFrame, first_year: int = 2010, height: int = 400) -> go.Figure:
    t = tokens()
    s = service.loc[first_year:]
    fig = go.Figure()
    for col, name, color in (
        ("scheme_disclosed_share", "Service scheme disclosed (share of MW)", t["blue"]),
        ("tenure_disclosed_share", "Contract length disclosed (share of MW)", t["aqua"]),
        ("aom5000_share_of_disclosed", "AOM5000 (share of disclosed MW)", t["orange"]),
    ):
        fig.add_trace(go.Scatter(x=s.index, y=s[col], mode="lines+markers", name=name, line=dict(color=color, width=2.4), marker=dict(size=7, line=dict(color=t["surface"], width=2)),
                                 hovertemplate="%{x}: %{y:.0%}<extra>" + name + "</extra>"))
    style_figure(fig, title="Service disclosure and AOM5000 uptake", height=height, ytitle="Share", yformat=".0%", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(range=[0, 1.02])
    return fig


def fig_tenure(service: pd.DataFrame, first_year: int = 2010, height: int = 400) -> go.Figure:
    t = tokens()
    s = service.loc[first_year:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s["tenure_median_years"], mode="lines+markers", name="Median contract length", line=dict(color=t["blue"], width=2.4),
                             marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.0f} years<extra>Median</extra>"))
    fig.add_trace(go.Scatter(x=s.index, y=s["tenure_wavg_years"], mode="lines+markers", name="MW-weighted average", line=dict(color=t["orange"], width=2.4),
                             marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra>MW-weighted</extra>"))
    style_figure(fig, title="Service contract length on new orders (years)", height=height, ytitle="Years", hovermode="x unified")
    year_axis(fig)
    fig.update_yaxes(rangemode="tozero")
    return fig


# ---- installed base ----------------------------------------------------------------------------
def fig_installed_vs_orders(ib: pd.DataFrame, top: int = 20, height: int | None = None) -> go.Figure:
    t = tokens()
    since = int(ib["since_year"].iloc[0])
    d = ib.nlargest(top, "installed_mw").iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["country"], x=d["installed_mw"], orientation="h", name="Installed base (end-2023)", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{y}: %{x:,.0f} MW installed<extra></extra>"))
    fig.add_trace(go.Bar(y=d["country"], x=d["orders_mw_since"], orientation="h", name=f"Announced orders since {since}", marker=dict(color=t["orange"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{y}: %{x:,.0f} MW ordered<extra></extra>"))
    style_figure(fig, title=f"Installed base vs new orders: top {top} countries", height=height or max(420, 34 * len(d) + 120), xtitle="MW", hovermode="closest")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=list(d["country"]))
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"])
    return fig


def fig_ib_scatter(ib: pd.DataFrame, height: int = 470) -> go.Figure:
    t = tokens()
    d = ib[(ib["installed_mw"] >= 150)].copy()

    def group(seg: str) -> str:
        if seg.startswith("Fast growth"):
            return "Fast growth (orders >= 25% of base)"
        if seg.startswith("Installed base, no new"):
            return "No orders since track record"
        return "Steady / other"

    d["grp"] = d["segment"].map(group)
    spec = {"Fast growth (orders >= 25% of base)": t["blue"], "Steady / other": t["neutral"], "No orders since track record": t["orange"]}
    fig = go.Figure()
    for grp, color in spec.items():
        sub = d[d["grp"] == grp]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["installed_mw"], y=sub["orders_vs_installed"], mode="markers", name=grp,
            marker=dict(color=color, size=np.clip(np.sqrt(sub["orders_mw_since"]) / 2.2 + 8, 9, 36), line=dict(color=t["surface"], width=2), opacity=0.9),
            customdata=np.stack([sub["country"], sub["orders_mw_since"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>installed %{x:,.0f} MW<br>orders since: %{customdata[1]:,.0f} MW (%{y:.0%} of base)<extra></extra>",
        ))
    labelled = d[(d["orders_mw_since"] >= 1500) | (d["installed_mw"] >= 9000)]
    for _, row in labelled.iterrows():
        fig.add_annotation(x=np.log10(row["installed_mw"]), y=row["orders_vs_installed"], text=row["country"], showarrow=False, yshift=18, font=dict(size=10, color=t["ink2"]))
    style_figure(fig, title="Where new orders outrun the installed base (bubble = MW ordered since)", height=height, xtitle="Installed base end-2023, MW (log scale)",
                 ytitle=f"Orders since {int(ib['since_year'].iloc[0])} as share of base", yformat=".0%", hovermode="closest")
    fig.update_xaxes(type="log")
    fig.update_yaxes(rangemode="tozero")
    return fig


def fig_fleet(fleet: pd.DataFrame, height: int = 360) -> go.Figure:
    t = tokens()
    fig = go.Figure()
    for (_, row), color in zip(fleet.iterrows(), t["ramp"]):
        fig.add_trace(go.Bar(y=[row["fleet_class"]], x=[row["installed_mw"]], orientation="h", name=row["fleet_class"], marker=dict(color=color, line=dict(color=t["surface"], width=2)),
                             text=[f"{row['installed_mw'] / 1000:,.1f} GW ({row['share']:.0%})"], textposition="auto", textfont=dict(size=11),
                             hovertemplate="%{y}: %{x:,.0f} MW<extra></extra>", showlegend=False))
    style_figure(fig, title="Installed fleet by platform class (end-2023)", height=height, xtitle="Installed MW", legend=False, hovermode="closest")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"])
    return fig


# ---- compare lab -------------------------------------------------------------------------------
def fig_compare_yearly(a: pd.Series, b: pd.Series, first_year: int = 2008, height: int = 400) -> go.Figure:
    t = tokens()
    fig = go.Figure()
    for s, color in ((a, t["blue"]), (b, t["orange"])):
        s = s.loc[first_year:]
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines+markers", name=str(s.name), line=dict(color=color, width=2.6), marker=dict(size=7, line=dict(color=t["surface"], width=2)),
                                 hovertemplate="%{x}: %{y:,.0f} MW<extra>" + str(s.name) + "</extra>"))
    style_figure(fig, title="Announced MW per order year", height=height, ytitle="MW", hovermode="x unified")
    year_axis(fig)
    return fig


def fig_compare_mix(a: pd.Series, b: pd.Series, name_a: str, name_b: str, by: str, height: int | None = None) -> go.Figure:
    t = tokens()
    sa, sb = a / a.sum() if a.sum() else a, b / b.sum() if b.sum() else b
    items = list(dict.fromkeys(list(sa.index) + list(sb.index)))
    items = [i for i in items if i != "Other"] + (["Other"] if "Other" in items else [])
    order = sorted([i for i in items if i != "Other"], key=lambda i: -(sa.get(i, 0) + sb.get(i, 0))) + (["Other"] if "Other" in items else [])
    fig = go.Figure()
    for s, name, color in ((sa, name_a, t["blue"]), (sb, name_b, t["orange"])):
        fig.add_trace(go.Bar(y=order, x=[s.get(i, 0.0) for i in order], orientation="h", name=name, marker=dict(color=color, line=dict(color=t["surface"], width=2)),
                             hovertemplate="%{y}: %{x:.0%}<extra>" + name + "</extra>"))
    style_figure(fig, title=f"Mix by {by.lower()} (share of each side's MW)", height=height or max(380, 28 * len(order) + 120), xtitle="Share of MW", hovermode="closest")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(showgrid=False, autorange="reversed", categoryorder="array", categoryarray=order)
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], tickformat=".0%")
    return fig


# ---- data quality ------------------------------------------------------------------------------
def fig_reconciliation(recon: pd.DataFrame, first_year: int = 2016, height: int = 440) -> go.Figure:
    t = tokens()
    r = recon.loc[first_year:]
    r = r[r["reported_oi"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.index, y=r["announced"], name="Announced orders", marker=dict(color=t["blue"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>Announced</extra>"))
    fig.add_trace(go.Bar(x=r.index, y=r["unannounced_listed"].fillna(0), name="Unannounced as listed in the workbook", marker=dict(color=t["orange"], line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: %{y:,.0f} MW<extra>Unannounced (listed)</extra>"))
    fig.add_trace(go.Scatter(x=r.index, y=r["reported_oi"], mode="markers", name="Reported order intake", marker=dict(symbol="diamond-open", size=12, color=t["ink"], line=dict(width=2, color=t["ink"])),
                             customdata=r["coverage"], hovertemplate="%{x}: reported %{y:,.0f} MW<br>listed total = %{customdata:.0%} of reported<extra></extra>"))
    for year, row in r.iterrows():
        if row["status"] == "Overshoots reported":
            fig.add_annotation(x=year, y=row["listed_total"], text=f"+{-row['gap'] / 1000:.1f} GW over", showarrow=False, yshift=14, font=dict(size=11, color=t["ink"]))
    style_figure(fig, title="Announced + unannounced MW against Vestas' reported order intake", height=height, ytitle="MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig


def fig_completeness_heatmap(comp: pd.DataFrame, first_year: int = 2010, height: int = 420) -> go.Figure:
    t = tokens()
    c = comp.loc[first_year:]
    fig = go.Figure(
        go.Heatmap(
            z=c.to_numpy().T, x=[str(i) for i in c.index], y=list(c.columns), colorscale=t["seq"], zmin=0, zmax=1,
            text=(c.to_numpy().T * 100).round(0).astype(int).astype(str), texttemplate="%{text}", textfont=dict(size=10),
            hovertemplate="%{y}, %{x}: %{z:.0%} of announced MW<extra></extra>", colorbar=dict(title="", tickformat=".0%", thickness=10), xgap=2, ygap=2,
        )
    )
    style_figure(fig, title="Share of each year's announced MW with the field disclosed", height=height, hovermode="closest", legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(type="category")
    return fig
