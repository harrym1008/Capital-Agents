import pandas as pd
from dateutil import tz


# General constants for the data collectors

UTC = tz.gettz("UTC")
NEW_YORK = tz.gettz("America/New_York")

START_DATE_STR = "2016-01-01"
START_DATE = pd.Timestamp(START_DATE_STR, tz=NEW_YORK)

FIRST_TRAD_DAY_AFTER_START_STR = "2016-01-04"
FIRST_TRAD_DAY_AFTER_START = pd.Timestamp(FIRST_TRAD_DAY_AFTER_START_STR, tz=NEW_YORK)

END_DATE_STR = "2026-04-30"
END_DATE = pd.Timestamp(END_DATE_STR, tz=NEW_YORK)

IPO_BEFORE_START_DATE = pd.Timestamp("2015-12-31", tz=NEW_YORK)


# For tickerclient.py

BAD_SECURITY_TERMS = [
    "warrant",
    "right",
    "unit",
    "depositary",
    "notes",
    "note",
    "senior",
    "bond",
    "etf",
    "fund",
    "trust preferred",
    "income",
    "series"
]

GOOD_SECURITY_TERMS = [
    "common stock",
    "common shares",
    "ordinary shares",
    "preferred units",
]

ALL_TICKERS_FILE = "data/tickers.parquet"
TEMP_TICKERS_FILE = "data/all_tickers.parquet"



# For newsclient.py

NEWS_BATCHES_DIR = "data/newsbatches"
NEWS_PARQUET_PATH = "data/news.parquet"
NEWS_INDEX_PARQUET_PATH = "data/newsindex.parquet"
NEWS_BATCH_SIZE = 2000


# For ohlcvclient.py

NYSE_DIRECTORY = "data/ohlcv_nyse/"
NASDAQ_DIRECTORY = "data/ohlcv_nasdaq/"
OHLC_FILE_OUTPUT = "{}.parquet"

CORP_ACTIONS_OUTPUT = "data/corpactions.parquet"
TICKER_CHANGES_OUTPUT = "data/tickerchanges.parquet"