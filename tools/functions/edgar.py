import datetime
import re
import pandas as pd
import numpy as np
from typing import List
from io import StringIO

from edgar import Filing
from edgar.xbrl import XBRL
from edgar.financials import Statement

from tools.tool_registry import DataProviders, Tool
from tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, isLocalDataAvailable, NumberType 

from dataquery.price_provider import DailyPriceProvider
from dataquery.edgar_provider import EdgarDataProvider, FormType, CompanyRef
from dataquery.macro_provider import MacroDataProvider, MacroSeries
from dataquery.forex_provider import ForexDataProvider, CURRENCY_MAP

from collectors.constants import NEW_YORK


def tsToNy(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NEW_YORK)
    else:
        ts = ts.tz_convert(NEW_YORK)
    return ts


def safeDivide(numerator: float, denominator: float):
    if denominator == 0 or denominator is None:
        return None
    try:
        return numerator / denominator
    except Exception:
        return None
    

def extractConceptFromStatement(statement: Statement, conceptNames: list, preferYtd: bool = False):
    if not statement or not conceptNames:
        return None
    try:
        df = statement.to_dataframe()
        if df.empty:
            return None

        matchingRows = pd.DataFrame()
        if "concept" in df.columns:
            matchingRows = df[df["concept"].isin(conceptNames)]
        if matchingRows.empty and "standard_concept" in df.columns:
            matchingRows = df[df["standard_concept"].isin(conceptNames)]

        if matchingRows.empty:
            return None

        valueCols = [c for c in df.columns if any(char.isdigit() for char in str(c)) and "level" not in str(c).lower()]
        if not valueCols:
            return None

        if preferYtd:
            def extractMonths(colName):
                match = re.search(r'(\d+)\s*Months?', str(colName), re.IGNORECASE)
                return int(match.group(1)) if match else 0
            valueCols = sorted(valueCols, key=extractMonths, reverse=True)

        val = matchingRows.iloc[0][valueCols[0]]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None

        # Convert bracketed numbers to negative, remove other punctuation
        strVal = str(val).strip().replace("(", "-")
        for punct in [")", ",", "$"]:
            strVal = strVal.replace(punct, "")
        return float(strVal)

    except Exception as e:
        return None


def extractMetric(filing: Filing, metricType: str, provider: EdgarDataProvider):
    preferYtd = (filing.form in ["10-Q", "6-K"])

    try:
        obj, xbrl = provider.downloadFilingObjects(filing)
        if not obj:
            return None

        financials = obj.financials
        metricsDict = financials.get_financial_metrics() or {}
        incomeStatement = financials.income_statement()
        balanceSheet = financials.balance_sheet()
        cashFlowStatement = financials.cash_flow_statement()

        match metricType:
            case "revenue":
                concepts = ["us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap_SalesRevenueNet", "us-gaap_Revenues", "us-gaap_InterestAndFeeIncomeLoansAndLeases"]
                val = extractConceptFromStatement(incomeStatement, concepts, preferYtd)
                return val if val is not None else metricsDict.get("revenue")
            case "netIncome":
                concepts = ["us-gaap_NetIncomeLoss", "us-gaap_ProfitLoss", "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic"]
                val = extractConceptFromStatement(incomeStatement, concepts, preferYtd)
                return val if val is not None else metricsDict.get("net_income")
            case "operating_income":
                concepts = ["us-gaap_OperatingIncomeLoss", "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "us-gaap_IncomeBeforeProvisionForIncomeTaxes", "us-gaap_IncomeLossFromContinuingOperationsBeforeIncomeTaxes"]
                val = extractConceptFromStatement(incomeStatement, concepts, preferYtd)
                return val if val is not None else metricsDict.get("operating_income")
            case "da":
                concepts = ["us-gaap_DepreciationDepletionAndAmortization", "us-gaap_DepreciationAndAmortization", "us-gaap_Depreciation"]
                # D&A only exists in cash flow statements, isn't inside metrics dict
                return extractConceptFromStatement(cashFlowStatement, concepts, preferYtd)
    except Exception:
        pass
    return None


def calculateTrueTtm(filings: List[Filing], metricType: str, provider: EdgarDataProvider, startIndex=0):
    if not filings or startIndex >= len(filings):
        return None

    targetFiling = filings[startIndex]
    val = extractMetric(targetFiling, metricType, provider)

    if val is None:
        return None

    annualForms = ["10-K", "20-F", "40-F"]
    quarterlyForms = ["10-Q", "6-K"]

    if targetFiling.form in annualForms:
        return val

    targetPeriod = datetime.datetime.strptime(targetFiling.period_of_report, "%Y-%m-%d").date()
    prior10kVal = None
    priorSame10qVal = None

    for f in filings[startIndex + 1:]:
        fPeriod = datetime.datetime.strptime(f.period_of_report, "%Y-%m-%d").date()
        daysDiff = (targetPeriod - fPeriod).days

        if daysDiff > 450:
            break

        if f.form in annualForms and prior10kVal is None:
            prior10kVal = extractMetric(f, metricType, provider)
        elif f.form in quarterlyForms and priorSame10qVal is None:
            if 330 <= daysDiff <= 390:
                priorSame10qVal = extractMetric(f, metricType, provider)

    if prior10kVal is not None and priorSame10qVal is not None:
        return val + prior10kVal - priorSame10qVal

    return val


def calculateHistoricalBeta(companyRef: CompanyRef, timestamp: pd.Timestamp, 
                            ohlcvProvider: DailyPriceProvider, macroProvider: MacroDataProvider):
    try:
        startDate = timestamp - pd.DateOffset(years=3)

        stockPrices = ohlcvProvider.getPeriodDailyTickerData(companyRef.ticker, startDate, timestamp)

        # The stock has IPOed after the start date, adjust 
        firstDate = stockPrices["date"].min()
        if firstDate > startDate:
            startDate = firstDate
        if len(stockPrices) < 60:
            return None     # Not enough data to calculate beta

        # Adjust for splits
        finalSplitFactor = stockPrices["splitFactor"].iloc[-1]
        for col in ["open", "high", "low", "close", "vwap"]:
            stockPrices[col] = stockPrices[col] * (stockPrices["splitFactor"] / finalSplitFactor)
        stockPrices["splitFactor"] = finalSplitFactor

        sp500Prices = macroProvider.getSeries(MacroSeries.SP500, startDate, timestamp)

        stockPrices["date"] = pd.to_datetime(stockPrices["date"]).dt.tz_localize(None)
        sp500Prices["date"] = pd.to_datetime(sp500Prices["date"]).dt.tz_localize(None)

        df = pd.merge(
            stockPrices[["date", "close"]],
            sp500Prices[["date", "close"]],
            on="date",
            suffixes=("_stock", "_sp500")
        ).dropna()
        
        dfReturns = df[["close_stock", "close_sp500"]].pct_change().dropna()
        dfReturns.columns = ["stock", "sp500"]
        covMatrix = np.cov(dfReturns["stock"], dfReturns["sp500"])
        varMarket = covMatrix[1, 1]
        covStockMarket = covMatrix[0, 1]

        if varMarket == 0:
            return None
        return float(covStockMarket / varMarket)
    except Exception:
        return None


def findPriorYearComparableFiling(filings: List[Filing]):
    if not filings:
        return None, None
    targetFiling = filings[0]
    targetForm = targetFiling.form
    targetPeriod = datetime.datetime.strptime(targetFiling.period_of_report, "%Y-%m-%d").date()

    for idx, f in enumerate(filings[1:], start=1):
        fPeriod = datetime.datetime.strptime(f.period_of_report, "%Y-%m-%d").date()
        daysDiff = (targetPeriod - fPeriod).days
        if daysDiff > 450:
            break
        if f.form == targetForm and 330 <= daysDiff <= 390:
            return idx, f
    return None, None


def calculateDividendYield(companyRef: CompanyRef, timestamp: pd.Timestamp, priceProvider: DailyPriceProvider):
    try:    
        ticker = companyRef.ticker

        thisYear = timestamp.year
        lastYear = thisYear - 1

        startDate = pd.Timestamp(timestamp - pd.DateOffset(years=1)).tz_convert(NEW_YORK)
        endDate = pd.Timestamp(timestamp).tz_convert(NEW_YORK)

        corpActions = pd.concat(
            [priceProvider.getYearCorporateActions(lastYear), priceProvider.getYearCorporateActions(thisYear)]
        )
        corpActions = corpActions[(corpActions["ticker"] == ticker) & (corpActions["actionType"] == "cash_dividend")]
        corpActions = corpActions[(corpActions["date"] >= startDate) & (corpActions["date"] <= endDate)]

        totalRate = corpActions["rate"].sum()
        currentSharePrice = priceProvider.getSingleDayTickerData(ticker, endDate)["close"]
        return totalRate / currentSharePrice if currentSharePrice else None
    
    except Exception as e:
        return None


def extractCurrency(filing: Filing, filingXbrl: XBRL):
    if filing.form in ["10-K", "10-Q"]:
        return "USD"
    try:
        if not filingXbrl:
            return None
        df = filingXbrl.facts.to_dataframe()
        if df.empty or "currency" not in df.columns:
            return None
        currencies = df["currency"].dropna()
        if currencies.empty:
            return None
        return currencies.mode().iloc[0]
    except Exception:
        return None



def fetchValidFilings(companyRef: CompanyRef, data: DataProviders, timestamp: pd.Timestamp, formsToFetch: List[FormType]):
    validFormsStr = [f.formCode for f in formsToFetch]

    allFilingsRaw = data.edgar.loadFilingRefsForCompany(companyRef, formsToFetch) or []
    validFilings: List[Filing] = []

    for filing in allFilingsRaw:
        if filing.form in validFormsStr and not filing.form.endswith("/A"):     # No amended filings
            try:
                filingDate = pd.to_datetime(filing.filing_date, utc=True)
                if filingDate <= timestamp:
                    validFilings.append(filing)
            except Exception:
                continue

    validFilings.sort(key=lambda f: pd.to_datetime(f.filing_date, utc=True), reverse=True)
    if len(validFilings) > 8:
        validFilings = validFilings[:8]     # Limit to the first (most recent) 8 filings for TTM calculations   
    return validFilings



def fetchCompanyValuationMetrics(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    cacheKey = f"valuation|{ticker}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached
    
    companyRef = CompanyRef(ticker)

    formsToFetch = [FormType.FORM_10K, FormType.FORM_10Q, FormType.FORM_20F, FormType.FORM_40F]
    validFilings = fetchValidFilings(companyRef, data, timestamp, formsToFetch)
    latestFiling = validFilings[0]
    print(f"Latest filing for {ticker} before {timestamp}: {latestFiling.filing_url} ")

    # Calculate recent price, shares outstanding, mkt cap
    todayRow = data.ohlcv.getSingleDayTickerData(ticker, timestamp)
    if todayRow is None or todayRow.empty:
        recentPrice = None
        sharesOutstanding = None
        marketCap = None
    else:
        recentPrice = todayRow.get("close", None)
        sharesOutstanding = todayRow.get("outstandingShares", None)
        if recentPrice is not None and sharesOutstanding is not None:
            marketCap = recentPrice * sharesOutstanding
        else:
            marketCap = None

    filingObj, filingXbrl = data.edgar.downloadFilingObjects(latestFiling)
    financials = filingObj.financials if filingObj else None
    metrics = financials.get_financial_metrics() if financials else {}

    ttmRevenue = calculateTrueTtm(validFilings, "revenue", data.edgar)
    ttmOperatingIncome = calculateTrueTtm(validFilings, "operating_income", data.edgar)
    ttmNetIncome = calculateTrueTtm(validFilings, "netIncome", data.edgar)
    ttmDA = calculateTrueTtm(validFilings, "da", data.edgar) or 0.0     # Depreciation and amortisation

    if ttmOperatingIncome is not None and ttmNetIncome is not None:
        rev_val = ttmRevenue if ttmRevenue is not None else 0.0
        if ttmOperatingIncome > 0 and ttmOperatingIncome > rev_val and ttmNetIncome < 0:
            ttmOperatingIncome = -ttmOperatingIncome

    if ttmOperatingIncome is None and ttmNetIncome is None:
        ttmEbitda = None
    else:
        ttmEbitda = (ttmOperatingIncome or ttmNetIncome) + ttmDA

    totalAssets = metrics.get("total_assets")
    totalLiabilities = metrics.get("total_liabilities")
    stockholdersEquity = metrics.get("stockholders_equity")

    if stockholdersEquity is None and totalAssets is not None and totalLiabilities is not None:
        stockholdersEquity = totalAssets - totalLiabilities

    totalDebt = None
    cashAndEquivalents = None
    if financials:
        try:
            balanceSheet = financials.balance_sheet()
            if balanceSheet:
                shortDebt = extractConceptFromStatement(balanceSheet, ["us-gaap_ShortTermBorrowings", "us-gaap_CommercialPaper", "us-gaap_LongTermDebtCurrent", "us-gaap_DebtCurrent"]) or 0.0
                longDebt = extractConceptFromStatement(balanceSheet, ["us-gaap_LongTermDebtNoncurrent", "us-gaap_LongTermDebtAndCapitalLeaseObligations", "us-gaap_LongTermDebt"]) or 0.0
                if shortDebt > 0 or longDebt > 0:
                    totalDebt = shortDebt + longDebt

                cashVal = extractConceptFromStatement(balanceSheet, ["us-gaap_CashAndCashEquivalentsAtCarryingValue", "us-gaap_CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "us-gaap_Cash"]) or 0.0
                mktVal = extractConceptFromStatement(balanceSheet, ["us-gaap_MarketableSecuritiesCurrent", "us-gaap_ShortTermInvestments"]) or 0.0
                if cashVal > 0 or mktVal > 0:
                    cashAndEquivalents = cashVal + mktVal
        except Exception:
            pass
            
    if totalDebt is None and metrics.get("debt_to_assets") is not None and totalAssets is not None:
        totalDebt = totalAssets * metrics.get("debt_to_assets")
        
    if cashAndEquivalents is None and metrics.get("current_assets") is not None:
        cashAndEquivalents = metrics.get("current_assets")


    # Get the currency and convert if necessary
    reportCurrency = extractCurrency(latestFiling, filingXbrl)
    if reportCurrency not in CURRENCY_MAP:
        reportCurrency = "USD"    # Default to USD if unknown
    fxRate = data.forex.getLatestCurrToUsd(reportCurrency, before=timestamp) 

    if fxRate != 1.0:
        if ttmRevenue is not None: ttmRevenue *= fxRate
        if ttmOperatingIncome is not None: ttmOperatingIncome *= fxRate
        if ttmNetIncome is not None: ttmNetIncome *= fxRate
        if ttmEbitda is not None: ttmEbitda *= fxRate
        if ttmDA is not None: ttmDA *= fxRate
        if totalAssets is not None: totalAssets *= fxRate
        if totalLiabilities is not None: totalLiabilities *= fxRate
        if stockholdersEquity is not None: stockholdersEquity *= fxRate
        if totalDebt is not None: totalDebt *= fxRate
        if cashAndEquivalents is not None: cashAndEquivalents *= fxRate


    # Calculate enterprise value
    if marketCap is not None and totalDebt is not None and cashAndEquivalents is not None:
        enterpriseValue = marketCap + totalDebt - cashAndEquivalents
    else:
        enterpriseValue = None

    # Calculate trailing P/E ratio
    if marketCap is not None and ttmNetIncome is not None:
        trailingPE = safeDivide(marketCap, ttmNetIncome)
    else:
        trailingPE = None

    # Calculate a proxy for forward P/E ratio (trailingPE/(1.0 + yoyGrowthRate)) (analyst estimates not available)
    # Also calculate proxy for PEG ratio
    priorIdx, _ = findPriorYearComparableFiling(validFilings)
    yoyGrowthRate = None
    if priorIdx is not None:
        priorTtmNetIncome = calculateTrueTtm(validFilings, "netIncome", data.edgar, startIndex=priorIdx)
        if ttmNetIncome is not None and priorTtmNetIncome is not None:
            yoyGrowthRate = safeDivide(ttmNetIncome - priorTtmNetIncome, priorTtmNetIncome)

    if trailingPE is not None and yoyGrowthRate is not None:
        forwardPE = safeDivide(trailingPE, (1.0 + yoyGrowthRate))
        pegRatio = safeDivide(forwardPE, yoyGrowthRate*100.0)
    else:
        forwardPE = None
        pegRatio = None

    # Calculate price to book ratio
    if marketCap is not None and stockholdersEquity is not None:
        priceToBook = safeDivide(marketCap, stockholdersEquity)
    else:
        priceToBook = None

    # Calculate beta and dividend yield using helper functions
    beta = calculateHistoricalBeta(companyRef, timestamp, data.ohlcv, data.macro)
    dividendYield = calculateDividendYield(companyRef, timestamp, data.ohlcv)

    # Calculate margins
    profitMargins = safeDivide(ttmNetIncome, ttmRevenue)
    ebitdaMargins = safeDivide(ttmEbitda, ttmRevenue)
    operatingMargins = safeDivide(ttmOperatingIncome, ttmRevenue)
    returnOnEquity = safeDivide(ttmNetIncome, stockholdersEquity)

    # Access short interest data
    shortPositionData = data.short.getLatestShortInterestForTicker(ticker, before=timestamp)
    if shortPositionData is not None:
        shortInterest = safeDivide(shortPositionData.currentShortPositions, sharesOutstanding)
        daysToCover = shortPositionData.daysToCover
    else:
        shortInterest = "unknown"
        daysToCover = "unknown"


    # All data collated, return as a dictionary

    dataHeader = {
        "ticker": companyRef.ticker,
        "cik": companyRef.cik,
        "company": latestFiling.company,
        "timestamp": timestamp.isoformat(),
        "latestPrice": cleanNumber(recentPrice, NumberType.STOCK_PRICE),
    }
    if reportCurrency != "USD":
        dataHeader |= {
            "currencyNote": f"Financials are reported in {reportCurrency}. The data provided below has been converted to USD "
                            f"using the latest exchange rate ({cleanNumber(fxRate, NumberType.DECIMAL)} {reportCurrency} --> 1 USD)"
        }
    else:
        dataHeader |= {
            "currencyNote": "Financials are reported in USD."
        }

    filingForm = latestFiling.form
    filingPeriod = latestFiling.period_of_report
    filingDate = latestFiling.filing_date
    if filingForm in ["10-K", "20-F", "40-F"]:
        periodDesc = f"fiscal year ended {filingPeriod}"
    else:
        periodDesc = f"fiscal quarter ended {filingPeriod}"
        
    dataHeader |= {
        "filingNote": f"Values are calculated using the {filingForm} filed on {filingDate}, "
                      f"covering the {periodDesc}. TTM metrics combine this filing with prior-year "
                      f"comparable filings (prior 10-K and same-quarter 10-Q) to represent a "
                      f"trailing twelve-month window.\n"
                      f"Consider the age of the filings used to calculate metrics when interpreting the results "
                      f"and analysing the company."
    }

    fundamentalData = cleanData({
        "sharesOutstanding": cleanNumber(sharesOutstanding, NumberType.LARGE_NUMBER),
        "marketCap": cleanNumber(marketCap, NumberType.LARGE_DOLLARS),
        "enterpriseValue": cleanNumber(enterpriseValue, NumberType.LARGE_DOLLARS),

        "trailingPE": cleanNumber(trailingPE, NumberType.DECIMAL),
        "growthAdjustedPE": cleanNumber(forwardPE, NumberType.DECIMAL),
        # "pegRatio": cleanNumber(pegRatio, NumberType.DECIMAL),
        "growthAdjustedPENote": "Calculated as trailing P/E divided by (1 + YoY TTM growth rate). "
                                "This is a proxy heuristic, as analyst estimates for forward earnings are not available. "
                                "It is not equivalent to analyst-based forward P/E, and should be interpreted with caution.",

        "priceToBook": cleanNumber(priceToBook, NumberType.DECIMAL),
        "beta": cleanNumber(beta, NumberType.DECIMAL),
        "dividendYield": cleanNumber(dividendYield, NumberType.UNSCALED_PERCENTAGE),
        "shortInterest": cleanNumber(shortInterest, NumberType.UNSCALED_PERCENTAGE),
        "daysToCover": cleanNumber(daysToCover, NumberType.DECIMAL),

        "profitMargins": cleanNumber(profitMargins, NumberType.UNSCALED_PERCENTAGE),
        "ebitdaMargins": cleanNumber(ebitdaMargins, NumberType.UNSCALED_PERCENTAGE),
        "operatingMargins": cleanNumber(operatingMargins, NumberType.UNSCALED_PERCENTAGE),
        "returnOnEquity": cleanNumber(returnOnEquity, NumberType.UNSCALED_PERCENTAGE)
    })

    hiddenData = cleanData({
        "ttmRevenue": cleanNumber(ttmRevenue, NumberType.LARGE_DOLLARS),
        "ttmOperatingIncome": cleanNumber(ttmOperatingIncome, NumberType.LARGE_DOLLARS),
        "ttmNetIncome": cleanNumber(ttmNetIncome, NumberType.LARGE_DOLLARS),
        "ttmEbitda": cleanNumber(ttmEbitda, NumberType.LARGE_DOLLARS),
        "ttmDA": cleanNumber(ttmDA, NumberType.LARGE_DOLLARS),
        "totalAssets": cleanNumber(totalAssets, NumberType.LARGE_DOLLARS),
        "totalLiabilities": cleanNumber(totalLiabilities, NumberType.LARGE_DOLLARS),
        "stockholdersEquity": cleanNumber(stockholdersEquity, NumberType.LARGE_DOLLARS),
        "totalDebt": cleanNumber(totalDebt, NumberType.LARGE_DOLLARS),
        "cashAndEquivalents": cleanNumber(cashAndEquivalents, NumberType.LARGE_DOLLARS),
        "yoyGrowthRate": cleanNumber(yoyGrowthRate, NumberType.UNSCALED_PERCENTAGE),
    })

    jsonOutput = dataHeader | fundamentalData
    jsonOutput["more"] = hiddenData
    data.cache.put(cacheKey, jsonOutput)
    return jsonOutput



def formatStatementFromDataframe(df: pd.DataFrame, filing: Filing, data: DataProviders, statementName: str, ticker: str, currency: str):
    if df.empty:
        return f"No {statementName} data available for filing {filing.form} filed on {filing.filing_date}."

    try:
        out = StringIO()

        companyName = filing.company
        filingForm = filing.form
        filingDate = pd.to_datetime(filing.filing_date).strftime("%Y-%m-%d")

        out.write(f" ======== {companyName.upper()} ({ticker.upper()}) - {filingForm} filed {filingDate} ======== \n")
        out.write(f"Extracted statement: {statementName.upper()}\n")
        out.write(f"Figures are in {currency} unless stated otherwise")

        if currency != "USD":
            fxRate = data.forex.getLatestCurrToUsd(currency, before=filingDate)
            out.write(f" (where 1 USD = {cleanNumber(fxRate, NumberType.EXCHANGE_RATE)} {currency})")
        out.write("\n\n")

        rawDf = df.copy()

        # Use abstract mask to find headers, and fix the broken "level" values in the dataframe
        abstractMask = df["concept"].str.endswith("Abstract", na=False)
        sections = []
        currentSection = []

        for _, row in df.iterrows():
            if row["concept"].endswith("Abstract"):
                if currentSection:
                    sections.append(currentSection)
                currentSection = [row]
            else:
                currentSection.append(row)

        if currentSection:
            sections.append(currentSection)

        formattedSections = []

        for section in sections:
            sectionDf = pd.DataFrame(section)
            # Make the abstract rows have level 0, so they are treated as headers
            sectionDf.loc[sectionDf["concept"].str.endswith("Abstract", na=False), "level"] = 0
            # Then apply a mapping to the other rows to ensure they have increasing levels, max gap of 1
            levelMapping = {level: i for i, level in enumerate(sorted(sectionDf["level"].unique()))}

            sectionDf["level"] = sectionDf["level"].map(levelMapping)
            formattedSections.append(sectionDf)

        df = pd.concat(formattedSections, ignore_index=True)

        periods = [col for col in df.columns if str(col).startswith("20")]
        periodCount = len(periods)
        if periodCount == 0:
            return f"No valid periods found in {statementName} data for filing {filing.form} filed on {filing.filing_date}."
        out.write(f"{periodCount} periods laid out in columns:  { ' | '.join(periods) }\n\n")

        indent = " --> "
        maxLabelLength = (
            df["label"].str.len()         # Length of the label
            + df["level"] * len(indent)   # Indentation based on level
            + 1                           # Colon
        ).max()

        for rowDict in df.to_dict(orient="records"):
            label = rowDict["label"].strip(": ") + ":"
            level = rowDict["level"]
            indentStr = indent * level
            linePrefix = f"{indentStr}{label.upper() if level == 0 else label}".ljust(maxLabelLength)

            out.write(linePrefix)
            for i, period in enumerate(periods):
                out.write(" | ")
                val = rowDict.get(period, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    out.write("    -    ")
                else:
                    out.write(cleanNumber(val, NumberType.LARGE_NUMBER).rjust(9))
            out.write("\n")

        # rawDf.to_parquet(f"working_statement.parquet", index=False)
        return out.getvalue()


    except Exception as e:
        return f"Error processing {statementName} data for filing {filing.form} filed on {filing.filing_date}: {str(e)}"


def fetchFormattedStatementInJson(data: DataProviders, timestamp: pd.Timestamp, 
                            ticker: str, statementName: str, statementGetter: callable, periodType: str = "annual"):
    companyRef = CompanyRef(ticker)
    formsToFetch = [FormType.FORM_10K, FormType.FORM_20F, FormType.FORM_40F] if periodType == "annual" else [FormType.FORM_10Q]
    validFilings = fetchValidFilings(companyRef, data, timestamp, formsToFetch)

    allFilingsAccessionNumbers = [f.accession_number for f in validFilings]
    allFilingsNumberCacheStr = ",".join(allFilingsAccessionNumbers)
    cacheKey = f"stmt|{ticker}_{statementName.replace(' ', '')}_{allFilingsNumberCacheStr}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    if not validFilings:
        return f"No valid filings found for {ticker} before {timestamp}."

    latestFiling = validFilings[0]
    filingObj, filingXbrl = data.edgar.downloadFilingObjects(latestFiling)
    financials = filingObj.financials if filingObj else None
    statement: Statement = statementGetter(financials) if financials else None

    if statement is None:
        return f"No {statementName} data available for filing {latestFiling.form} filed on {latestFiling.filing_date}."

    df = statement.to_dataframe()
    reportCurrency = extractCurrency(latestFiling, filingXbrl) or "USD"

    formattedStatement = formatStatementFromDataframe(df, latestFiling, data, statementName, ticker, reportCurrency)
    jsonOutput = {"date": timestamp.strftime("%Y-%m-%d"), "statement": formattedStatement}

    data.cache.put(cacheKey, jsonOutput)
    return jsonOutput



def fetchIncomeStatement(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, periodType: str = "annual"):
    return fetchFormattedStatementInJson(data, timestamp, ticker, "Income Statement", lambda f: f.income_statement(), periodType)


def fetchBalanceSheet(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, periodType: str = "annual"):
    return fetchFormattedStatementInJson(data, timestamp, ticker, "Balance Sheet", lambda f: f.balance_sheet(), periodType)


def fetchCashFlowStatement(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, periodType: str = "annual"):
    return fetchFormattedStatementInJson(data, timestamp, ticker, "Cash Flow Statement", lambda f: f.cash_flow_statement(), periodType)


def fetchStatementOfEquity(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, periodType: str = "annual"):
    return fetchFormattedStatementInJson(data, timestamp, ticker, "Statement of Equity", lambda f: f.statement_of_equity(), periodType)


def fetchComprehensiveIncomeStatement(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, periodType: str = "annual"):
    return fetchFormattedStatementInJson(data, timestamp, ticker, "Comprehensive Income Statement", lambda f: f.comprehensive_income(), periodType)

