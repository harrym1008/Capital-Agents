import os
from dataclasses import dataclass
from datetime import date, timedelta

from tqdm import tqdm
import yfinance as yf
import pandas as pd

from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import FOREX_DIRECTORY, NEW_YORK


@dataclass
class Currency:
    code: str
    name: str
    symbol: str
    yfinanceTicker: str = None


CURRENCIES = [
    Currency("USD", "United States Dollar", "$", ""),
    Currency("TWD", "New Taiwan Dollar", "NT$", "TWDUSD=X"),
    Currency("EUR", "Euro", "€", "EURUSD=X"),
    Currency("GBP", "British Pound", "£", "GBPUSD=X"),
    Currency("JPY", "Japanese Yen", "JP¥", "JPYUSD=X"),
    Currency("CNY", "Chinese Yuan", "CN¥", "CNYUSD=X"),
    Currency("HKD", "Hong Kong Dollar", "HK$", "HKDUSD=X"),
    Currency("KRW", "South Korean Won", "₩", "KRWUSD=X"),
    Currency("CAD", "Canadian Dollar", "CA$", "CADUSD=X"),
    Currency("CHF", "Swiss Franc", "Fr.", "CHFUSD=X"),
    Currency("AUD", "Australian Dollar", "AU$", "AUDUSD=X"),
    Currency("BRL", "Brazilian Real", "R$", "BRLUSD=X"),
    Currency("INR", "Indian Rupee", "₹", "INRUSD=X"),
    Currency("MXN", "Mexican Peso", "MX$", "MXNUSD=X"),
    Currency("ZAR", "South African Rand", "R", "ZARUSD=X"),
    Currency("SEK", "Swedish Krona", "kr", "SEKUSD=X"),
    Currency("DKK", "Danish Krone", "kr", "DKKUSD=X"),
    Currency("NOK", "Norwegian Krone", "kr", "NOKUSD=X"),
    Currency("ILS", "Israeli Shekel", "₪", "ILSUSD=X"),
    Currency("ARS", "Argentine Peso", "AR$", "ARSUSD=X"),
    Currency("CLP", "Chilean Peso", "CL$", "CLPUSD=X"),
]

CURRENCY_MAP = {currency.code: currency for currency in CURRENCIES}


class CurrencyDataClient:
    def __init__(self, startDate: str, endDate: str, rateLimiterDatabase: GlobalRateLimiters):
        self.startDate = startDate
        self.endDate = endDate
        self.yfRateLimiter = rateLimiterDatabase.yFinanceLimiter

    def massDownload(self, pbar=None):
        if not os.path.exists(FOREX_DIRECTORY):
            os.mkdir(FOREX_DIRECTORY)
        else:
            for filename in os.listdir(FOREX_DIRECTORY):
                if filename.endswith(".parquet"):
                    os.remove(os.path.join(FOREX_DIRECTORY, filename))


        # First download the yfinance macro data
        if pbar is None:
            pbar = tqdm(
                total=len(CURRENCIES),
                desc="Downloading forex data",
                smoothing=0.1,
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=len(CURRENCIES))
            pbar.set_description("Forex: Downloading")


        # yfinance 'end' is exclusive — add 1 day so END_DATE's data is included
        yfEndDate = (pd.Timestamp(self.endDate) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        for currency in CURRENCIES:
            if currency.code == "USD":
                # Skip USD
                pbar.update(1)
                continue

            try:
                self.yfRateLimiter.wait()

                df = yf.download(
                    currency.yfinanceTicker,
                    start=self.startDate,
                    end=yfEndDate,
                    interval="1d",
                    auto_adjust=False,
                    progress=False
                )

                if df.empty:
                    print(f"Warning: No data found for {currency.code} ({currency.name})")
                    pbar.update(1)
                    continue

                # Flatten yfinance columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                df.columns = [str(column).lower().replace(" ", "_") for column in df.columns]

                df = df.drop(columns=["adj_close", "volume"], errors="ignore")
                df.columns = ["date", "open", "high", "low", "close"]

                # To NY timezone
                df["date"] = pd.to_datetime(df["date"])
                if df["date"].dt.tz is None:
                    df["date"] = df["date"].dt.tz_localize(NEW_YORK)
                else:
                    df["date"] = df["date"].dt.tz_convert(NEW_YORK)

                filename = f"{currency.code}_USD.parquet"
                filepath = os.path.join(FOREX_DIRECTORY, filename)
                df.to_parquet(filepath, index=False)

            except Exception as e:
                print(f"Error downloading data for {currency.code} ({currency.name}): {e}")

            pbar.update(1)
