# Vestas Sales Dashboard

Interactive Streamlit dashboard for the workbook `Vestas_economical_data_start_2026.xlsx`.

## Run

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app auto-detects the Excel file in the current folder and loads:
- `Vestas Economy` (overall economics)
- all `OI YYYY` sheets (order intake analytics across years/platforms/countries/customers/services/delivery)
