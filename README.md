# Vestas Sales Intelligence Dashboard

Interactive Streamlit dashboard for Vestas commercial and economics analytics.

## What This App Includes
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
- `Overall Economics`
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

## Updating Data
1. Put the latest Excel workbook in the project root.
2. Run the app once to regenerate `data_cache/vestas_parsed_data.pkl`.
3. Update market JSON files if needed:
   - `data/vestas_stock_monthly.json`
   - `data/market_prices_monthly.json`
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
