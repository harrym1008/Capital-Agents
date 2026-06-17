import os
import pandas as pd
from dataclasses import dataclass

from collectors.constants import NYSE_DIRECTORY, NASDAQ_DIRECTORY, ALL_TICKERS_FILE, CORP_ACTIONS_OUTPUT, NEW_YORK, UTC
from dataquery.lrucache import LRUCache


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
        
        if profile.ipoDate is not None and date < profile.ipoDate:
            return False
        
        if profile.delistDate is not None and date > profile.delistDate:
            return False
        
        return True