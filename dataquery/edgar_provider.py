import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Any
import pandas as pd

from edgar import Company, set_identity, Filing
from edgar.company_reports import CompanyReport
from edgar.xbrl import XBRL

from collectors.constants import UTC, SEC_EDGAR_IDENTITY
from dataquery.lru_cache import LRUCache
from dataquery.ticker_provider import TickerDataProvider


class FormType(Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_20F = "20-F"
    FORM_40F = "40-F"
    FORM_6K = "6-K"

    def __init__(self, formCode: str):
        self.formCode = formCode


class CompanyRef:
    def __init__(self, ticker: Optional[str], cik: Optional[str] = None):
        if ticker is None and cik is None:
            raise ValueError("Either ticker or cik must be provided.")
        self.ticker = ticker.strip().upper() if ticker else None
        self.cik = str(cik).strip().zfill(10) if cik else None

    def loadCik(self, data: TickerDataProvider):
        if self.cik is not None:
            return self.cik
        if self.ticker is None:
            return None
        
        profile = data.getTickerProfile(self.ticker)
        if profile is None:
            return None
        
        self.cik = profile.cik

    def __str__(self):
        return self.cik or self.ticker or None



class EdgarDataProvider:
    def __init__(self, tickerProvider: TickerDataProvider, cache: LRUCache):
        self.tickerProvider = tickerProvider
        self.cache = cache
        set_identity(SEC_EDGAR_IDENTITY) 

    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)
    
    def loadFilingRefsForCompany(self, companyRef: CompanyRef, formType: FormType|List[FormType] = None) -> List[Filing]:
        companyRef.loadCik(self.tickerProvider)
        formCodes = [formType.formCode] if isinstance(formType, FormType) else [f.formCode for f in formType]
        
        key = f"edgar|rawFilings_{companyRef}_{"&".join(formCodes)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            company = Company(str(companyRef))
            if formType is not None:
                filings = company.get_filings(form=formCodes)
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


    def getLatestFilingRef(self, companyRef: CompanyRef, formType: FormType = None, before: pd.Timestamp = None) -> Filing:
        companyRef.loadCik(self.tickerProvider)
        if before is None:
            before = pd.Timestamp.now(tz=UTC)

        formStr = formType.formCode if isinstance(formType, FormType) else (formType or "ALL")
        beforeNorm = self.normaliseTimestamp(before)

        key = f"edgar|latestFiling_{companyRef}_{formStr}_{beforeNorm.strftime('%Y-%m-%d')}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        filings = self.loadFilingRefsForCompany(companyRef, formType=formType)
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
    

    def getFilingRefBeforeAnother(self, companyRef: CompanyRef, formType: FormType, beforeFiling: Filing) -> Filing:
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

        return self.getLatestFilingRef(companyRef, formType=formType, before=beforeDate)
    

    def downloadFilingObjects(self, filing: Filing) -> Tuple[CompanyReport, XBRL]:
        if filing is None:
            return None, None

        accessionNumber = filing.accession_number
        cacheKey = f"edgar|filingDownload_{accessionNumber}"

        cached = self.cache.get(cacheKey)
        if cached is not None:
            return cached

        try:
            parsedObj = filing.obj()
            parsedXbrl = filing.xbrl()

            self.cache.put(cacheKey, (parsedObj, parsedXbrl))
            return parsedObj, parsedXbrl
        except Exception:
            return None, None


    def findRefAndDownloadFilingObjects(self, companyRef: CompanyRef, formType: FormType, before: pd.Timestamp = None):
        latestFiling = self.getLatestFilingRef(companyRef, formType=formType, before=before)
        if latestFiling is None:
            return None
        downloaded = self.downloadFilingObjects(latestFiling)
        return downloaded