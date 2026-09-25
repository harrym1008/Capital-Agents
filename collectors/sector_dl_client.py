import os
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import SECTOR_DIRECTORY, START_DATE_STR, END_DATE_STR, UTC


@dataclass
class SectorInfo:
    ticker: str
    name: str
    description: str
    category: str


GICS_SECTORS: Dict[str, SectorInfo] = {
    "XLK": SectorInfo(
        ticker="XLK",
        name="Information Technology",
        description="Technology hardware, software, semiconductors, and IT services",
        category="Growth / Cyclical"
    ),
    "XLF": SectorInfo(
        ticker="XLF",
        name="Financials",
        description="Banks, financial services, insurance, and capital markets",
        category="Value / Cyclical"
    ),
    "XLV": SectorInfo(
        ticker="XLV",
        name="Health Care",
        description="Pharmaceuticals, healthcare equipment, biotechnology, and managed care",
        category="Defensive"
    ),
    "XLY": SectorInfo(
        ticker="XLY",
        name="Consumer Discretionary",
        description="Automobiles, retail, luxury goods, travel, and leisure",
        category="Cyclical / Growth"
    ),
    "XLI": SectorInfo(
        ticker="XLI",
        name="Industrials",
        description="Aerospace & defense, industrial machinery, transportation, and construction",
        category="Cyclical"
    ),
    "XLC": SectorInfo(
        ticker="XLC",
        name="Communication Services",
        description="Interactive media, telecom, media networks, and entertainment",
        category="Growth / Sensitive"
    ),
    "XLP": SectorInfo(
        ticker="XLP",
        name="Consumer Staples",
        description="Food & beverage, consumer household goods, tobacco, and personal products",
        category="Defensive"
    ),
    "XLE": SectorInfo(
        ticker="XLE",
        name="Energy",
        description="Oil & gas exploration, refining, equipment, and consumable fuels",
        category="Cyclical / Value"
    ),
    "XLU": SectorInfo(
        ticker="XLU",
        name="Utilities",
        description="Electric utilities, multi-utilities, water, and renewable energy",
        category="Defensive"
    ),
    "XLRE": SectorInfo(
        ticker="XLRE",
        name="Real Estate",
        description="Equity real estate investment trusts (REITs) and real estate development",
        category="Rate-Sensitive / Defensive"
    ),
    "XLB": SectorInfo(
        ticker="XLB",
        name="Materials",
        description="Chemicals, construction materials, containers & packaging, metals & mining",
        category="Cyclical"
    )
}

# Direct mapping for tickers.parquet underscored lowercase sector strings
DB_SECTOR_TO_TICKER = {
    "financials": "XLF",
    "financial_services": "XLF",    # Flexibility for LLMs that arent following the rules!
    "industrials": "XLI",
    "health_care": "XLV",
    "consumer_staples": "XLP",
    "consumer_discretionary": "XLY",
    "real_estate": "XLRE",
    "communication_services": "XLC",
    "materials": "XLB",
    "utilities": "XLU",
    "energy": "XLE",
    "information_technology": "XLK",
}

# Lookup map by sector name in lowercase for flexible agent querying
SECTOR_NAME_TO_TICKER = {info.name.lower(): ticker for ticker, info in GICS_SECTORS.items()}
# Merge DB underscored keys into lookup map
for dbKey, tick in DB_SECTOR_TO_TICKER.items():
    SECTOR_NAME_TO_TICKER[dbKey] = tick
    SECTOR_NAME_TO_TICKER[dbKey.replace("_", " ")] = tick


class SectorDataClient:
    def __init__(self, startDateStr: str = START_DATE_STR, endDateStr: str = END_DATE_STR, rateLimiterDatabase: Optional[GlobalRateLimiters] = None):
        self.startDateStr = startDateStr
        self.endDateStr = endDateStr
        self.rateLimiters = rateLimiterDatabase or GlobalRateLimiters()
        self.yfRateLimiter = self.rateLimiters.yFinanceLimiter


    def massDownload(self, pbar=None):
        if not os.path.exists(SECTOR_DIRECTORY):
            os.makedirs(SECTOR_DIRECTORY, exist_ok=True)

        if pbar is None:
            pbar = tqdm(
                total=len(GICS_SECTORS),
                desc="Downloading GICS sector ETFs",
                smoothing=0.1,
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=len(GICS_SECTORS))
            pbar.set_description("Sectors: Downloading")

        # yfinance end date is exclusive, so add 1 day
        yfEndDate = (pd.Timestamp(self.endDateStr) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        for ticker, sectorInfo in GICS_SECTORS.items():
            try:
                self.yfRateLimiter.wait()
                rawDf = yf.download(
                    ticker,
                    start=self.startDateStr,
                    end=yfEndDate,
                    auto_adjust=False,
                    progress=False
                )

                if rawDf.empty:
                    pbar.update(1)
                    continue

                if isinstance(rawDf.columns, pd.MultiIndex):
                    rawDf.columns = rawDf.columns.get_level_values(0)

                rawDf.reset_index(inplace=True)
                rawDf.columns = rawDf.columns.str.lower()

                dateCol = pd.to_datetime(rawDf["date"])
                if dateCol.dt.tz is not None:
                    dateCol = dateCol.dt.tz_convert(UTC).dt.tz_localize(None)
                rawDf["date"] = dateCol

                targetCols = ["date", "open", "high", "low", "close", "volume"]
                validCols = [col for col in targetCols if col in rawDf.columns]
                cleanedDf = rawDf[validCols].copy()

                for col in ["open", "high", "low", "close"]:
                    if col in cleanedDf.columns:
                        cleanedDf[col] = cleanedDf[col].round(4).ffill()

                parquetPath = os.path.join(SECTOR_DIRECTORY, f"{ticker}.parquet")
                cleanedDf.to_parquet(parquetPath, index=False)
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
            finally:
                pbar.update(1)

        if pbar is not None and not hasattr(pbar, "_external"):
            pbar.close()


if __name__ == "__main__":
    client = SectorDataClient()
    client.massDownload()
