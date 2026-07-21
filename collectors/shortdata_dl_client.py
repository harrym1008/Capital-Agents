import time
from io import StringIO
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm

from collectors.constants import SHORT_PARQUET_PATH
from collectors.rate_limiter import GlobalRateLimiters
from collectors.market_calendar import MarketCalendar

CSV_URL_FEED = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{dateStr}.csv"
CSV_CUTOFF = date(2021, 5, 31)


class ShortDataClient:
    def __init__(self, startDate: date, endDate: date, rateLimiterDatabase: GlobalRateLimiters):
        self.startDate = max(startDate, CSV_CUTOFF)
        self.endDate = endDate

        # Data is not released in the first 8 business days after the settlement date
        if self.endDate.day >= 1 and self.endDate.day < 11:
            if self.endDate.month == 1:
                self.endDate = date(self.endDate.year - 1, 12, 31)
            else:
                self.endDate = date(self.endDate.year, self.endDate.month - 1, 28) 
        if self.endDate.day >= 15 and self.endDate.day < 25:
            self.endDate = date(self.endDate.year, self.endDate.month, 14)

        self.calendar = MarketCalendar(
            (startDate - timedelta(days=365)).strftime("%Y-%m-%d"), 
            (endDate + timedelta(days=365)).strftime("%Y-%m-%d")
        )

        self.shortDataPath = Path(SHORT_PARQUET_PATH)
        self.shortDataPath.parent.mkdir(parents=True, exist_ok=True)

        self.rateLimiter = rateLimiterDatabase.finraLimiter


    def getSettlementDates(self) -> list[date]:
        dates = []    

        def getLastMarketOpenDay(d: date) -> date:
            while not self.calendar.isOpenDay(d.strftime("%Y-%m-%d")):
                d -= timedelta(days=1)
                if d < (self.startDate - timedelta(days=365)):
                    break    #Failsafe
            return d

        d = date(self.startDate.year, self.startDate.month, 1)

        while d <= self.endDate:
            midMonth = getLastMarketOpenDay(date(d.year, d.month, 15))
            if d.month == 12:
                endMonth = getLastMarketOpenDay(date(d.year, 12, 31))
            else:
                endMonth = getLastMarketOpenDay(date(d.year, d.month + 1, 1) - timedelta(days=1))

            if midMonth >= self.startDate and midMonth <= self.endDate:
                dates.append(midMonth)
            if endMonth >= self.startDate and endMonth <= self.endDate:
                dates.append(endMonth)

            if d.month == 12:
                d = date(d.year + 1, 1, 1)
            else:
                d = date(d.year, d.month + 1, 1)

        return sorted(dates)


    def fetchReport(self, session: requests.Session, url: str) -> pd.DataFrame:
        for attempt in range(1, 5):
            try:
                # self.rateLimiter.wait()
                resp = session.get(url, timeout=20)
            except requests.RequestException as e:
                print(f"Attempt {attempt}: Error fetching {url}: {e}")
                time.sleep(10 * attempt)
                continue

            if resp.status_code == 404:
                return pd.DataFrame() 
            
            if resp.status_code != 200:
                print(f"Attempt {attempt}: Error fetching {url}: {resp.status_code}")
                time.sleep(10 * attempt)
                continue

            try:
                text = resp.text.strip()
                if not text:
                    return pd.DataFrame()
                df = pd.read_csv(StringIO(text), sep="|", engine="python")
            except Exception as e:
                print(f"Failed to parse CSV from {url}: {e}")
                return pd.DataFrame()
            
            df = df.dropna(how="all")
            return df
        
        return pd.DataFrame() 


    def cleanDataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        columnsToKeep = [
            "settlementDate", 
            "symbolCode", 
            "marketClassCode", 
            "currentShortPositionQuantity", 
            "previousShortPositionQuantity",
            "changePercent",
            "averageDailyVolumeQuantity", 
            "daysToCoverQuantity"
        ]

        df = df[columnsToKeep]
        df = df[df["marketClassCode"].isin(["NYSE", "NNM"])]
        df["marketClassCode"] = df["marketClassCode"].replace({
            "NYSE": "NYS",
            "NNM": "NAS"
        })

        df = df.rename(columns={
            "settlementDate": "date",
            "symbolCode": "ticker",
            "marketClassCode": "exchange",
            "currentShortPositionQuantity": "currentShortPositions",
            "previousShortPositionQuantity": "previousShortPositions",
            "averageDailyVolumeQuantity": "avgDailyVolume",
            "daysToCoverQuantity": "daysToCover"
        })
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
        return df


    def processSingleDate(self, d: date, session: requests.Session) -> pd.DataFrame:
        dStr = d.strftime("%Y%m%d")
        url = CSV_URL_FEED.format(dateStr=dStr)

        df = self.fetchReport(session, url)
        df = self.cleanDataframe(df)
        return df


    def massDownload(self, pbar=None):
        if not self.shortDataPath.parent.exists():
            self.shortDataPath.parent.mkdir(parents=True, exist_ok=True)
        else:
            if self.shortDataPath.exists():
                self.shortDataPath.unlink()

        settlementDates = self.getSettlementDates()
        if pbar is None:
            pbar = tqdm(
                total=len(settlementDates),
                desc="Downloading short data",
                smoothing=0.1,
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=len(settlementDates))
            pbar.set_description("Downloading short data...")

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (CapitalAgents)"})

        dataframes = [None] * len(settlementDates)


        with ThreadPoolExecutor(max_workers=8) as executor:
            futureToIndex = {
                executor.submit(self.processSingleDate, d, session): i 
                for i, d in enumerate(settlementDates)
            }

            for future in as_completed(futureToIndex):
                index = futureToIndex[future]
                try:
                    dataframes[index] = future.result()
                except Exception as e:
                    print(f"Error processing date {settlementDates[index]}: {e}")
                    dataframes[index] = pd.DataFrame()
                pbar.update(1)

        combined = pd.concat(dataframes, ignore_index=True, sort=False)
        numericCols = ["currentShortPositions", "previousShortPositions", "avgDailyVolume", "daysToCover", "changePercent"]
        for col in numericCols:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

        combined["ticker"] = combined["ticker"].str.strip().str.upper()
        combined = combined.sort_values(by=["date", "ticker"]).reset_index(drop=True)
        combined.to_parquet(self.shortDataPath, index=False)