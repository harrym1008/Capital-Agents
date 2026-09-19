import os
from pathlib import Path
import yfinance as yf
import pandas as pd
from fredapi import Fred
from tqdm import tqdm

from dotenv import load_dotenv

from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import MACRO_DIRECTORY



YFINANCE_MACRO_TICKERS = {
    "SP500": {
        "desc": "S&P 500 Index",
        "yfticker": "^GSPC"
    },
    "NDQ100": {
        "desc": "NASDAQ 100 Index",
        "yfticker": "^NDX"
    },
    "DJIA": {
        "desc": "Dow Jones Industrial Average Index",
        "yfticker": "^DJI"
    },
    "RUS2000": {
        "desc": "Russell 2000 Index",
        "yfticker": "^RUT"
    },
    "VIX": {
        "desc": "CBOE Volatility Index",
        "yfticker": "^VIX"
    },
    "OIL_WTI": {
        "desc": "WTI Crude Oil",
        "yfticker": "CL=F"
    },
    "OIL_BRENT": {
        "desc": "Brent Crude Oil",
        "yfticker": "BZ=F"
    },
    "GOLD": {
        "desc": "Gold",
        "yfticker": "GC=F"
    },
    "SILVER": {
        "desc": "Silver",
        "yfticker": "SI=F"
    },
    "NATGAS": {
        "desc": "Natural Gas",
        "yfticker": "NG=F"
    },
    "BTCUSD": {
        "desc": "1 Bitcoin to US Dollar exchange rate",
        "yfticker": "BTC-USD"
    },
    "GBPUSD": {
        "desc": "1 British Pound to US Dollar exchange rate",
        "yfticker": "GBPUSD=X"
    },
    "EURUSD": {
        "desc": "1 Euro to US Dollar exchange rate",
        "yfticker": "EURUSD=X"
    },
    "USDJPY": {
        "desc": "1 US Dollar to Japanese Yen exchange rate",
        "yfticker": "USDJPY=X"
    },
}


FRED_MACRO_SERIES = {
    "CPI": {
        "desc": "Consumer Price Index",
        "unit": "Index (1982-1984=100)",
        "id": "CPIAUCSL"
    },
    "CORECPI": {
        "desc": "Core Consumer Price Index (excludes food and energy)",
        "unit": "Index (1982-1984=100)",
        "id": "CPILFESL"
    },
    "UNEMPLOYMENT": {
        "desc": "Unemployment Rate",
        "unit": "Percent",
        "id": "UNRATE"
    },
    "FEDFUNDS": {
        "desc": "Federal Funds Rate",
        "unit": "Percent",
        "id": "FEDFUNDS"
    },
    "GDP": {
        "desc": "Gross Domestic Product",
        "unit": "Billions of Dollars",
        "id": "GDP"
    },
    "TREAS_30Y": {
        "desc": "US 30-Year Treasury Yield",
        "unit": "Percent",
        "id": "DGS30"
    },
    "TREAS_10Y": {
        "desc": "US 10-Year Treasury Yield",
        "unit": "Percent",
        "id": "DGS10"
    },
    "TREAS_2Y": {
        "desc": "US 2-Year Treasury Yield",
        "unit": "Percent",
        "id": "DGS2"
    },
    "TREAS_3MO": {
        "desc": "US 3-Month Treasury Yield",
        "unit": "Percent",
        "id": "DGS3MO"
    },
}



class MacroDataClient:
    def __init__(self, startDateStr, endDateStr, rateLimiterDatabase: GlobalRateLimiters):
        load_dotenv()
        self.fredClient = Fred(api_key=os.getenv("FRED_API_KEY"))
        self.limiters = rateLimiterDatabase

        self.startDateStr = startDateStr
        self.endDateStr = endDateStr


    def massDownload(self, pbar=None):
        if not os.path.exists(MACRO_DIRECTORY):
            os.mkdir(MACRO_DIRECTORY)
        else:
            for filename in os.listdir(MACRO_DIRECTORY):
                if filename.endswith(".parquet"):
                    os.remove(os.path.join(MACRO_DIRECTORY, filename))

        # First download the yfinance macro data
        if pbar is None:
            pbar = tqdm(
                total=len(YFINANCE_MACRO_TICKERS) + len(FRED_MACRO_SERIES),
                desc="Downloading macro data",
                smoothing=0.1,
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=len(YFINANCE_MACRO_TICKERS) + len(FRED_MACRO_SERIES))
            pbar.set_description("Macro: Downloading")

        # yfinance 'end' is exclusive - add 1 day so END_DATE's data is included
        yfEndDate = (pd.Timestamp(self.endDateStr) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        for name, info in YFINANCE_MACRO_TICKERS.items():
            yfTicker = info.get("yfticker", name)
            df = yf.download(
                yfTicker,
                start=self.startDateStr,
                end=yfEndDate,
                auto_adjust=False,
                progress=False
            )

            df.columns = df.columns.get_level_values(0)
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()

            df["date"] = pd.to_datetime(df["date"])
            df["date"] = df["date"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")

            targetCols = ["date", "open", "high", "low", "close", "volume"]

            df = df[[col for col in targetCols if col in df.columns]]
            df[targetCols[1:5]] = df[targetCols[1:5]].round(4)

            # Backfill empty values
            df[targetCols[1:5]] = df[targetCols[1:5]].ffill()

            parquetPath = os.path.join(MACRO_DIRECTORY, f"{name}.parquet")
            df.to_parquet(parquetPath, index=False)
            pbar.update(1)
            self.limiters.yFinanceLimiter.wait()


        # Then download the FRED macro data
        for name, info in FRED_MACRO_SERIES.items():
            seriesId = info["id"]
            df = self.fredClient.get_series(seriesId, observation_start=self.startDateStr, observation_end=self.endDateStr)
            df = df.reset_index()
            df.columns = ["date", "value"]
            df.dropna(subset=["value"], inplace=True)

            df["date"] = pd.to_datetime(df["date"])
            df["date"] = df["date"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")

            parquetPath = os.path.join(MACRO_DIRECTORY, f"{name}.parquet")
            df.to_parquet(parquetPath, index=False)
            pbar.update(1)
            self.limiters.fredLimiter.wait()

        if pbar is not None and not hasattr(pbar, '_external'):
            pbar.close()
