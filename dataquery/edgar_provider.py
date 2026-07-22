import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import pandas as pd

from edgar import Company, set_identity, Filing
from collectors.constants import UTC, SEC_EDGAR_IDENTITY
from dataquery.lru_cache import LRUCache


class FormType(Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"

    def __init__(self, formCode: str):
        self.formCode = formCode



class EdgarDataProvider:
    def __init__(self, cache: LRUCache):
        self.cache = cache
        set_identity(SEC_EDGAR_IDENTITY) 

    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)
    
    def loadFilingsForTicker(self, ticker: str, formType: FormType = None):
        ticker = ticker.upper().strip()
        formStr = formType.formCode if isinstance(formType, FormType) else (formType or "ALL")
        
        key = f"edgar|rawFilings_{ticker}_{formStr}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            company = Company(ticker)
            if formType is not None:
                filings = company.get_filings(form=formStr)
            else:
                filings = company.get_filings()

            if filings is None or len(filings) == 0:
                self.cache.put(key, None)
                return None

            self.cache.put(key, filings)
            return filings
        except Exception:
            self.cache.put(key, None)
            return None


    def getLatestFiling(self, ticker: str, formType: FormType = None, before: pd.Timestamp = None) -> Optional[Filing]:
        if before is None:
            before = pd.Timestamp.now(tz=UTC)

        formStr = formType.formCode if isinstance(formType, FormType) else (formType or "ALL")
        beforeNorm = self.normaliseTimestamp(before)

        key = f"edgar|latestFiling_{ticker}_{formStr}_{beforeNorm.isoformat()}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        filings = self.loadFilingsForTicker(ticker, formType=formType)
        if filings is None or len(filings) == 0:
            self.cache.put(key, None)
            return None
        
        validFilings = []
        for filing in filings:
            try:
                fDate = pd.to_datetime(filing.filing_date)
                if fDate.tzinfo is None:
                    fDate = fDate.tz_localize(UTC)
                else:
                    fDate = fDate.tz_convert(UTC)
                fDate = fDate.tz_localize(None)

                if fDate <= beforeNorm:
                    validFilings.append((fDate, filing))
            except Exception:
                continue

        if not validFilings:
            self.cache.put(key, None)
            return None
        
        # Sort chronologically ascending and return the last one
        validFilings.sort(key=lambda x: x[0])
        latestFiling: Filing = validFilings[-1][1]

        self.cache.put(key, latestFiling)
        return latestFiling
    

    def getFilingBeforeAnother(self, ticker: str, formType: FormType, beforeFiling: Filing) -> Optional[Filing]:
        if beforeFiling is None:
            return None

        try:
            beforeDate = pd.to_datetime(beforeFiling.filing_date)
            if beforeDate.tzinfo is None:
                beforeDate = beforeDate.tz_localize(UTC)
            else:
                beforeDate = beforeDate.tz_convert(UTC)
            beforeDate = beforeDate.tz_localize(None) - pd.Timedelta(days=1)
        except Exception:
            return None

        return self.getLatestFiling(ticker, formType=formType, before=beforeDate)
    

    def downloadFiling(self, filing: Filing):
        if filing is None:
            return None

        accessionNumber = filing.accession_number
        cacheKey = f"edgar|filingDownload_{accessionNumber}"

        cached = self.cache.get(cacheKey)
        if cached is not None:
            return cached

        try:
            parsedObj = filing.obj()
            self.cache.put(cacheKey, parsedObj)
            return parsedObj
        except Exception:
            return None