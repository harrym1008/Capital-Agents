import os
import pandas as pd
import numpy as np

from collectors.macro_dl_client import YFINANCE_MACRO_TICKERS
from tools.tool_registry import DataProviders, Tool
from tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, formatArticleAge, NumberType


FRED_SERIES_MAP = {
    "CPI": {
        "id": "CPIAUCSL",  
        "desc": "Consumer Price Index", 
        "unit": "Index (1982-1984=100)",
        "numType": NumberType.DECIMAL
    },
    "CORECPI": {
        "id": "CPILFESL",
        "desc": "Core CPI (excl. food & energy)",
        "unit": "Index (1982-1984=100)",
        "numType": NumberType.DECIMAL
    },
    "UNEMPLOYMENT": {
        "id": "UNRATE",
        "desc": "Unemployment Rate",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
    "FEDFUNDS": {
        "id": "FEDFUNDS",
        "desc": "Federal Funds Rate",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
    "GDP": {
        "id": "GDP",
        "desc": "Gross Domestic Product",
        "unit": "Billions of Dollars",
        "numType": NumberType.LARGE_DOLLARS
    },
    "TREAS_30Y": {
        "id": "DGS30",
        "desc": "30-Year Treasury Yield",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
    "TREAS_10Y": {
        "id": "DGS10",
        "desc": "10-Year Treasury Yield",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
    "TREAS_2Y": {
        "id": "DGS2",
        "desc": "2-Year Treasury Yield",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
    "TREAS_3MO": {
        "id": "DGS3MO",
        "desc": "3-Month Treasury Yield",
        "unit": "Percent",
        "numType": NumberType.PERCENTAGE
    },
}


def fetchMacroContext(tool: Tool, data: DataProviders, timestamp: pd.Timestamp):
    cacheKey = f"macro|context_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    jsonResult = {}
    allSeries = data.macro.getAllSeries()
    todaySnapshot = data.macro.getSnapshot(allSeries, timestamp)

    pastDates = {
        "5d": timestamp - pd.DateOffset(weeks=1),
        "1mo": timestamp - pd.DateOffset(months=1),
        "3mo": timestamp - pd.DateOffset(months=3),
        "6mo": timestamp - pd.DateOffset(months=6),
        "12mo": timestamp - pd.DateOffset(years=1),
        "3y": timestamp - pd.DateOffset(years=3),
        "5y": timestamp - pd.DateOffset(years=5)
    }
    pastSnapshots = {name: data.macro.getSnapshot(allSeries, date) for name, date in pastDates.items()}

    for series in allSeries:
        seriesIdentifier = series.parquetName
        seriesOrigin = series.source

        if seriesOrigin == "yfinance":
            seriesDesc = YFINANCE_MACRO_TICKERS.get(seriesIdentifier, {}).get("desc", seriesIdentifier)
            numberType = NumberType.STOCK_PRICE if "USD" not in seriesIdentifier \
                                else (NumberType.EXCHANGE_RATE if "BTC" not in seriesIdentifier else NumberType.STOCK_PRICE)
        else:
            seriesDesc = FRED_SERIES_MAP.get(seriesIdentifier, {}).get("desc", seriesIdentifier)
            numberType = FRED_SERIES_MAP.get(seriesIdentifier, {}).get("numType", NumberType.DECIMAL)

        seriesRows = todaySnapshot.loc[todaySnapshot["series"] == series]
        if seriesRows.empty:
            continue

        latestPrice = seriesRows["value"].iloc[0]
        latestValueStr = cleanNumber(latestPrice, numberType)

        outputsPerPeriod = {}
        fiftyTwoWeekMin = None
        fiftyTwoWeekMax = None

        for period, snapshot in pastSnapshots.items():
            if seriesOrigin == "yfinance" and period == "5y":
                continue
            elif seriesOrigin == "fred":
                if period == "5d":
                    continue
                if period == "1mo" and not seriesIdentifier.startswith("TREAS"):
                    continue
                elif period == "3mo" and seriesIdentifier in ["CPI", "CORECPI", "GDP"]:
                    continue

            snapRows = snapshot.loc[snapshot["series"] == series]
            if snapRows.empty:
                continue

            pastPrice = snapRows["value"].iloc[0]
            priceChangePct = ((latestPrice - pastPrice) / pastPrice) * 100 if pastPrice != 0 else 0

            if seriesOrigin == "yfinance" and period == "12mo":
                lowDate, lowVal = data.macro.getLowest(series, pastDates[period], timestamp)
                highDate, highVal = data.macro.getHighest(series, pastDates[period], timestamp)
                fiftyTwoWeekMin = cleanNumber(lowVal, numberType) if lowVal is not None else None
                fiftyTwoWeekMax = cleanNumber(highVal, numberType) if highVal is not None else None

            startPrice = cleanNumber(pastPrice, numberType)

            if seriesOrigin == "fred" and seriesIdentifier.startswith("TREAS"):
                bpsChange = (latestPrice - pastPrice) * 100
                outputsPerPeriod[period] = {
                    f"price_{period}_ago": startPrice,
                    f"change_{period}_bps": cleanNumber(bpsChange, NumberType.CHANGE_BP)
                }
            else:
                outputsPerPeriod[period] = {
                    f"price_{period}_ago": startPrice,
                    f"change_{period}_pct": cleanNumber(priceChangePct, NumberType.PERCENTAGE_CHANGE),
                }

        if seriesOrigin == "yfinance":
            jsonResult[seriesIdentifier] = {
                "description": seriesDesc,
                "latestValue": latestValueStr,
                "history": outputsPerPeriod,
                "52w": {
                    "52wLow": fiftyTwoWeekMin,
                    "52wHigh": fiftyTwoWeekMax
                }
            }
        else:
            jsonResult[seriesIdentifier] = {
                "description": seriesDesc,
                "unit": FRED_SERIES_MAP.get(seriesIdentifier, {}).get("unit", ""),
                "latestValue": latestValueStr,
                "history": outputsPerPeriod
            }

    jsonOutput = cleanData({"date": timestamp.strftime("%Y-%m-%d"), "macroContext": cleanData(jsonResult)})
    data.cache.put(cacheKey, jsonOutput)
    return jsonOutput


def fetchMacroNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, limit: int = 12):
    limit = min(max(limit, 1), 18)
    jsonResult = []

    newsWithContent = data.news.getRecentNewsForTickers(
        tickers=["SPY", "QQQ", "DIA", "GLD", "SLV", "VIX", "USO", "TLT"],
        before=timestamp,
        limit=limit,
        mustHaveContent=True,
        maxReferencedTickers=15
    )

    if newsWithContent is None or newsWithContent.empty:
        newsWithContent = data.news.getRecentNewsForTickers(
            tickers=["SPY", "QQQ", "DIA", "GLD", "SLV", "VIX", "USO", "TLT"],
            before=timestamp,
            limit=limit,
            mustHaveContent=False,
            maxReferencedTickers=15
        )

    idx = 0
    if newsWithContent is not None and not newsWithContent.empty:
        for _, row in newsWithContent.head(limit).iterrows():
            headline = row.get("headline", "").strip()
            if not headline:
                continue
            headline = cleanHtmlContent(headline)

            content = row.get("content", "").strip()
            if not content:
                content = "Article has no content available."
            else:
                content = cleanHtmlContent(content)
                if len(content) > 2500:
                    content = content[:2500] + "... [content truncated]"

            author = row.get("author", "").strip()
            rawDate = row.get("date")
            ageStr = formatArticleAge(rawDate, timestamp)
            articleId = str(row.get("id", ""))
            rawTs = pd.to_datetime(rawDate)
            if not pd.isna(rawTs):
                year = rawTs.year % 100
                month = rawTs.month
                url = f"https://www.benzinga.com/news/{year:02d}/{month:02d}/{articleId}"
            else:
                url = f"https://www.benzinga.com/news/{articleId}"

            jsonResult.append({
                "index": idx,
                "headline": headline,
                "content": content,
                "author": author,
                "articleAge": ageStr,
                "url": url
            })
            idx += 1
    else:
        jsonResult.append({"error": "No macro news data available."})

    validArticlesCount = len([item for item in jsonResult if "headline" in item])
    if validArticlesCount > 0 and validArticlesCount < limit:
        jsonResult.append({"info": f"Only {validArticlesCount} articles could be retrieved."})

    return {"date": timestamp.strftime("%Y-%m-%d"), "news": cleanData(jsonResult)}