"""Shared fixtures: the cached Vestas dataset, prepared once per test session."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intel import common  # noqa: E402
from intel.bundle import build_intel  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    with open(ROOT / "data_cache" / "vestas_parsed_data.pkl", "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="session")
def orders(raw):
    return common.prepare_orders(raw["orders"])


@pytest.fixture(scope="session")
def platforms(raw):
    return common.prepare_platforms(raw["platforms"])


@pytest.fixture(scope="session")
def intel(raw):
    market = pd.read_json(ROOT / "data" / "market_prices_monthly.json")
    stock = pd.read_json(ROOT / "data" / "vestas_stock_monthly.json")
    stock = stock.rename(columns={c: "date" for c in stock.columns if "date" in str(c).lower()})
    # the installed-base loader uses a path relative to the working directory
    import os

    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        yield build_intel(raw["economy"], raw["orders"], raw["platforms"], raw["unannounced"], stock, market)
    finally:
        os.chdir(previous)
