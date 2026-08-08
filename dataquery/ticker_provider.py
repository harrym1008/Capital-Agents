from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from dataclasses import dataclass
import requests

from collectors.constants import ALL_TICKERS_FILE, NEW_YORK


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    exchange: str
    isAdrc: bool

    ipoDate: pd.Timestamp
    delistDate: pd.Timestamp

    sector: str
    industry: str

    website: str
    summary: str

    cik: str
    isin: str



class TickerDataProvider:
    def __init__(self):
        self.tickerIndex = self.buildTickerIndex()
        self.logoIndex = {}


    def buildTickerIndex(self):
        allTickersDf = pd.read_parquet(ALL_TICKERS_FILE)

        tickerIndex = {}
        for row in allTickersDf.itertuples():
            tickerIndex[row.ticker] = CompanyProfile(
                ticker=row.ticker,
                name=row.name,
                exchange=row.exchange,
                isAdrc=row.isAdrc,
                ipoDate=pd.Timestamp(row.ipoDate, tz=NEW_YORK) if pd.notna(row.ipoDate) else None,
                delistDate=pd.Timestamp(row.delistDate, tz=NEW_YORK) if pd.notna(row.delistDate) else None,
                sector=row.sector,
                industry=row.industry,
                website=row.website,
                summary=row.summary,
                cik=row.cik,
                isin=row.isin
            )
        
        return tickerIndex
    

    def getTickerProfile(self, ticker) -> CompanyProfile:
        return self.tickerIndex.get(ticker)
    

    def getTickerProfileData(self, ticker, *fields):
        profile = self.getTickerProfile(ticker)
        if profile is None:
            return None
        
        return {field: getattr(profile, field, None) for field in fields}
    

    def isTickerListed(self, ticker, date):
        profile = self.getTickerProfile(ticker)
        if profile is None:
            return False

        ts = pd.Timestamp(date)
        tsNorm = ts.tz_localize(None) if ts.tzinfo is not None else ts

        if profile.ipoDate is not None:
            ipoNorm = profile.ipoDate.tz_localize(None) if profile.ipoDate.tzinfo is not None else profile.ipoDate
            if tsNorm < ipoNorm:
                return False

        if profile.delistDate is not None:
            delistNorm = profile.delistDate.tz_localize(None) if profile.delistDate.tzinfo is not None else profile.delistDate
            if tsNorm > delistNorm:
                return False

        return True


    def getCompanyLogoFromFinnhub(self, ticker):
        if ticker in self.logoIndex:
            return self.logoIndex[ticker]

        finnhubApiKey = os.getenv("FINNHUB_API_KEY")
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker.upper()}&token={finnhubApiKey}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            logoUrl = data.get("logo")
            if logoUrl:
                self.logoIndex[ticker] = logoUrl
                return logoUrl

        return f"https://placehold.co/256x256?text={ticker.upper()}"
