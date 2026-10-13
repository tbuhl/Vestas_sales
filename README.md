# Vestas Sales Intelligence Dashboard

Interactive Streamlit dashboard for Vestas commercial and economics analytics.

## What This App Includes
- **Executive Briefing**: a company-level front page with KPIs, a full-year outlook range, and twelve auto-generated, number-backed findings.
- **Intelligence Lab**: momentum and outlook, order timing, market structure, account radar, service, installed base, a head-to-head Compare Lab and a data-quality / reconciliation view.
- Overall economics trends from the `Vestas Economy` sheet.
- Year-by-year, quarterly, platform, country, delivery, and correlation analytics from `OI YYYY` sheets.
- Market overlays:
  - Vestas stock monthly OHLC (`VWS.CO`)
  - Steel monthly price (`HRC=F`)
  - Copper monthly price (`HG=F`)

## Data Files Used By The App
- `data_cache/vestas_parsed_data.pkl`
  - Parsed/cached core dataset used for app analytics.
- `data/vestas_stock_monthly.json`
  - Monthly stock history used in the Overall Economics market chart.
- `data/market_prices_monthly.json`
  - Monthly steel/copper series used for market overlays.
- `data/vestas_installed_base.json`
  - Installed base by country and turbine type (the workbook's end-2023 track record). Rebuild with
    `python tools/build_installed_base.py <workbook.xlsx>`. The Installed Base section is hidden if the file is missing.

Optional source workbook for data refresh:
- `Vestas_economical_data_start_2026.xlsx`

## How Data Loading Works
- If the Excel workbook is present, the app parses it and refreshes `data_cache/vestas_parsed_data.pkl`.
- If the workbook is not present, the app runs from `data_cache/vestas_parsed_data.pkl`.

## Run Locally
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## User Guide
### Sidebar
- `Dark mode`: toggle theme.
- `Order year range`: global filter for order-based analytics.
- `Continents`, `Regions`, `Countries`, `Service schemes`, `Platforms`: filter data scope.
- `Minimum order MW`: exclude smaller orders.

### Tabs
- `Executive Briefing` (new, second tab; ignores the sidebar filters)
  - KPI strip: announced MW year-to-date vs the same date last year, last 12 months, full-year outlook (base case and range),
    record year, average ordered turbine, order backlog.
  - Cumulative-MW curves (current year against every year since 2013) and order intake by year with the outlook range.
  - Key findings: momentum, outlook, concentration, quarter-end timing, product, geography, pricing, backlog cover,
    accounts, service disclosure, installed fleet, data integrity. Every figure is computed from the data at runtime.
  - "What to watch next" list (e.g. how much Q4 must deliver to beat the record).
- `Intelligence Lab` (new; choose a section with the selector at the top)
  - `Momentum & Outlook` (full data): YTD vs prior years, outlook working (analogue years), quarter-end clustering,
    within-quarter order timing, backlog cover, realised turbine price, book-to-bill, backlog conversion, lead/lag correlations.
  - `Market Structure` (respects sidebar filters): order-size lumpiness and mega-order dependence, turbine size mix,
    rating/rotor growth, offshore-class share, leading platforms, country movers (LTM, YTD or full-year), market breadth, market lifecycle.
  - `Accounts & Service` (respects sidebar filters): account radar (Active / Cooling / Dormant), re-engagement watchlist with CSV download,
    account movers, service-scheme disclosure and contract length, installed base vs new orders, fleet by platform class.
  - `Compare Lab` (full data): head-to-head profile of two countries, continents, customers or platform families.
  - `Data Quality` (full data): reconciliation of announced + unannounced MW against reported order intake, field completeness by year,
    anomaly checks with severity, record-level detail, definitions.
- `Economy Summary` (formerly Overall Economics; now the front page)
  - Economics KPI cards and time series.
  - Derived ratios: gross/EBIT margin, service share of revenue, book-to-bill (MW), revenue per employee.
  - Stock candlestick chart with Y2/Y3 overlays.
  - Overlay options include economics metrics plus steel/copper series.
  - Economy year range slider also controls stock/market chart range.
- `Year-by-Year Overview`
  - Announced vs unannounced MW, order counts, average size.
  - Year-over-year growth and order-size distribution views.
  - Continent accumulation and market share views.
- `Quarterly Analytics`
  - Quarterly announced/unannounced mix with trailing 4-quarter trend, plus correlations.
- `Across Years`
  - Country, platform, service/time, rotor/MW, and customer trends.
- `Platform Analytics`
  - Timeline, service mix/time, customer and delivery views by platform.
- `Turbine Explorer`
  - Full catalog of every turbine variant in the order book: rotor, rating, swept area, specific power.
  - Technology map (rotor vs rating with specific-power isolines), specific-power trend.
  - Sales lifecycle Gantt per model and per-model drill-down with indicative family profiles.
  - CSV download of the catalog.
- `Country Analytics`
  - Single full-width map (bubble default; switchable to choropleth).
  - Country-level platform/service/delivery summaries.
- `Customer Intelligence`
  - Customer concentration (top-1/5/10 share), new vs returning customer MW.
  - Key account table, cumulative account growth, and per-customer drill-down.
- `Delivery and Capacity`
  - Installed capacity and delivery-time trends.
  - Implied delivery pipeline and lead-time distribution.
- `Correlations`
  - Numeric correlation matrix and delivery-vs-order scatter analysis.
- `Information`
  - Source/disclaimer statements and short author section.

## Project Layout (intelligence layer)
```
app.py                  existing dashboard; adds the two new tabs and passes data to the intel package
intel/common.py         constants, cleaning helpers, derived order columns
intel/outlook.py        reconciliation, YTD/LTM, cumulative curves, full-year projection, order timing
intel/structure.py      concentration, product mix, geography, accounts, service, installed base, compare lab
intel/economics.py      price, backlog cover, conversion, lead/lag correlations
intel/quality.py        completeness and anomaly checks
intel/insights.py       rule-based findings and watchlist
intel/bundle.py         computes everything once (cached by the app)
intel/charts_*.py       Plotly figures;  intel/theme.py  palette, chart chrome, card CSS;  intel/ui.py  page layouts
tools/build_installed_base.py   extracts the installed-base JSON from the workbook
tests/                  pytest suite (python -m pytest tests)
```

## Method Notes
- **As-of date** is the latest order date in the data. Year-to-date comparisons use the same month and day in earlier years.
- **Outlook** applies the last 7 complete years as analogues, two ways (scale year-to-date by that year's full-year/YTD ratio; add that year's
  post-as-of MW), then adds unannounced MW (listed quarters plus missing quarters at the trailing 8-quarter average).
  It is a scenario range, not a forecast.
- **Reconciliation**: since 2019 announced + unannounced MW matches reported order intake within about 1%. The 2017 and 2018 unannounced values
  are identical to 2020's and overshoot reported intake, so the dashboard uses reported intake for those years.
- **Quarter-end clustering**: share of MW in the last 14 days of a quarter; the even-spread benchmark is 15%.
- **Offshore-class** means V164, V174 and V236 platforms (orders carry no onshore/offshore flag, so this is a proxy).
- **Countries**: region labels (EU, North America, ...) are excluded from country analytics; "United Kingdom - Offshore" and "Wales" are folded into United Kingdom.

## Tests
```powershell
python -m pip install pytest
python -m pytest tests
```
The tests check each analytic against an independent brute-force calculation on the cached dataset.

## Updating Data
1. Put the latest Excel workbook in the project root.
2. Run the app once to regenerate `data_cache/vestas_parsed_data.pkl`.
3. Update market JSON files if needed:
   - `data/vestas_stock_monthly.json`
   - `data/market_prices_monthly.json`
   - `data/vestas_installed_base.json` (run `python tools/build_installed_base.py <workbook.xlsx>` when the track-record sheet changes)
4. Commit updated cache/data files and deploy.

## Deploy (Streamlit Community Cloud)
1. Push repository to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Set `app.py` as the main file.
4. Deploy.

## Troubleshooting
- If visuals look stale after deployment, do a hard refresh in the browser.
- If no workbook is present, ensure `data_cache/vestas_parsed_data.pkl` exists.
- If market overlays are missing, check that JSON files exist in `data/`.
