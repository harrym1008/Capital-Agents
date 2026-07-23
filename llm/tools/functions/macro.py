import os

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from fredapi import Fred

from dotenv import load_dotenv
load_dotenv()

from collectors.macro_dl_client import YFINANCE_MACRO_TICKERS

from llm.tools.tool_registry import DataProviders, Tool
from llm.tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, isLocalDataAvailable, NumberType 


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
    }
}


def fetchMacroContext(tool: Tool, data: DataProviders, timestamp: pd.Timestamp):
    useLocal = isLocalDataAvailable(timestamp)
    jsonResult = {}

    if useLocal:
        allSeries = data.macro.getAllSeries()

        todaySnapshot = data.macro.getSnapshot(allSeries, timestamp)
        pastDates = {
            "1w": timestamp - pd.DateOffset(weeks=1),
            "1mo": timestamp - pd.DateOffset(months=1),
            "3mo": timestamp - pd.DateOffset(months=3),
            "6mo": timestamp - pd.DateOffset(months=6),
            "1y": timestamp - pd.DateOffset(years=1),
            "3y": timestamp - pd.DateOffset(years=3)
        }
        pastSnapshots = {name: data.macro.getSnapshot(allSeries, date) for name, date in pastDates.items()}

        for series in allSeries:
            seriesIdentifier = series.parquetName
            seriesOrigin = series.source

            if seriesOrigin == "yfinance":
                seriesDesc = YFINANCE_MACRO_TICKERS[seriesIdentifier]["desc"]
            elif seriesOrigin == "fred":
                seriesDesc = FRED_SERIES_MAP[seriesIdentifier]["desc"] + f" (unit: {FRED_SERIES_MAP[seriesIdentifier]['unit']})"
            
            latestValue = todaySnapshot.loc[todaySnapshot["series"] == series, "value"].iloc[0]
            latestValueStr = cleanNumber(latestValue, NumberType.STOCK_PRICE if seriesOrigin == "yfinance" else FRED_SERIES_MAP[seriesIdentifier]["numType"])

            outputsPerPeriod = {}
            
            for period, snapshot in pastSnapshots.items():
                if seriesOrigin == "fred" and period in ["1w", "1mo"]:      # Skip 1w and 1mo for FRED series due to low frequency
                    continue

                pastPrice = snapshot.loc[snapshot["series"] == series, "value"].iloc[0]
                if pd.isna(latestValue) or pd.isna(pastPrice) or pastPrice == 0:
                    pctChangeStr = "N/A"
                else:
                    pctChange = ((latestValue - pastPrice) / pastPrice) * 100
                    if pctChange == 0:
                        pctChangeStr = "0% (no change)"
                    else:
                        pctChangeStr = cleanNumber(pctChange, NumberType.PERCENTAGE_CHANGE)
                
                highest = cleanNumber(data.macro.getHighest(series, pastDates[period], timestamp)[1], 
                                      NumberType.STOCK_PRICE if seriesOrigin == "yfinance" else FRED_SERIES_MAP[seriesIdentifier]["numType"])
                lowest = cleanNumber(data.macro.getLowest(series, pastDates[period], timestamp)[1],
                                      NumberType.STOCK_PRICE if seriesOrigin == "yfinance" else FRED_SERIES_MAP[seriesIdentifier]["numType"])
                
                startPrice = cleanNumber(pastPrice, NumberType.STOCK_PRICE if seriesOrigin == "yfinance" else FRED_SERIES_MAP[seriesIdentifier]["numType"])
                
                outputsPerPeriod[period] = {
                    f"pctChange_{period}": pctChangeStr,
                    f"valueAtPeriodStart_{period}": startPrice,
                    f"highest_{period}": highest,
                    f"lowest_{period}": lowest
                }

            jsonResult[seriesIdentifier] = {
                "description": seriesDesc,
                "latestValue": latestValueStr,
                "history": outputsPerPeriod
            }   

    else:  # Get from online sources
        # First deal with yfinance
        allSeries = data.macro.getAllSeries()
        yfinanceSeries = [s for s in allSeries if s.source == "yfinance"]
        for series in yfinanceSeries:
            seriesIdentifier = series.parquetName
            seriesOrigin = series.source
            seriesDesc = YFINANCE_MACRO_TICKERS[seriesIdentifier]["desc"]

            yfTicker = YFINANCE_MACRO_TICKERS[seriesIdentifier]["yfticker"]
            tickerObj = yf.Ticker(yfTicker)

            fiveDaysHistory = tickerObj.history(period="5d", interval="1d")["Close"]
            if fiveDaysHistory.empty:
                jsonResult[seriesIdentifier] = {
                    "description": seriesDesc,
                    "error": "Could not download data from yfinance for this series."
                }
                continue

            latestPrice = fiveDaysHistory.iloc[-1]
            latestPriceStr = cleanNumber(latestPrice, NumberType.STOCK_PRICE)

            outputsPerPeriod = {}

            for timeFrame in ["5d", "1mo", "3mo", "6mo", "1y", "3y"]:
                timeFrameHistory = tickerObj.history(period=timeFrame, interval="1d" if timeFrame == "5d" else "1wk")  \
                                               .dropna(subset=["Open", "High", "Low", "Close"])
                if timeFrameHistory.empty:
                    outputsPerPeriod[timeFrame] = {"error": f"Could not download data from yfinance for this series for the '{timeFrame}' timeframe."}
                    continue

                pastPrice = timeFrameHistory["Close"].iloc[0]
                priceChangePct = ((latestPrice - pastPrice) / pastPrice) * 100 if pastPrice != 0 else 0
                highPrice = timeFrameHistory["High"].max()
                lowPrice = timeFrameHistory["Low"].min()

                pastPriceStr = cleanNumber(pastPrice, NumberType.STOCK_PRICE)
                priceChangePctStr = cleanNumber(priceChangePct, NumberType.PERCENTAGE_CHANGE)
                lowestPriceStr = cleanNumber(lowPrice, NumberType.STOCK_PRICE)
                highestPriceStr = cleanNumber(highPrice, NumberType.STOCK_PRICE)

                outputsPerPeriod[timeFrame] = {                    
                    f"valueAtPeriodStart_{timeFrame}": pastPriceStr,
                    f"pctChange_{timeFrame}": priceChangePctStr,
                    f"lowest_{timeFrame}": lowestPriceStr,
                    f"highest_{timeFrame}": highestPriceStr
                }

            jsonResult[seriesIdentifier] = {
                "description": seriesDesc,
                "latestValue": latestPriceStr,
                "history": outputsPerPeriod
            } 

        # Then deal with FRED
        fredClient = Fred(api_key=os.getenv("FRED_API_KEY"))
        periods = {
            "1mo": pd.DateOffset(months=1),
            "3mo": pd.DateOffset(months=3),
            "6mo": pd.DateOffset(months=6),
            "1y": pd.DateOffset(years=1),
            "3y": pd.DateOffset(years=3)
        }

        # Ensure base timestamp is naive for pandas compatibility
        naiveTimestamp = timestamp.tz_localize(None) if getattr(timestamp, "tz", None) is not None else timestamp

        for seriesName, seriesData in FRED_SERIES_MAP.items():
            seriesId = seriesData["id"]
            seriesDesc = seriesData["desc"]
            seriesUnit = seriesData["unit"]

            observationStart = (naiveTimestamp - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
            observationEnd = naiveTimestamp.strftime("%Y-%m-%d")

            rawSeries = fredClient.get_series(
                seriesId, 
                observation_start=observationStart, 
                observation_end=observationEnd
            ).dropna()

            if rawSeries.empty:
                jsonResult[seriesName] = {
                    "description": seriesDesc,
                    "unit": seriesUnit,
                    "error": "Could not download data from FRED for this series."
                }
                continue

            latestValue = rawSeries.iloc[-1]
            latestValueStr = cleanNumber(latestValue, seriesData["numType"])

            outputsPerPeriod = {}
            hasIndexError = False

            for label, offset in periods.items():
                targetDate = naiveTimestamp - offset
                closestDateIdx = rawSeries.index.get_indexer([targetDate], method="pad")[0]

                if closestDateIdx == -1:
                    jsonResult[seriesName] = {
                        "description": seriesDesc,
                        "unit": seriesUnit,
                        "error": f"No data available for the '{label}' period."
                    }
                    hasIndexError = True
                    break

                pastValue = rawSeries.iloc[closestDateIdx]
                pctChange = ((latestValue - pastValue) / pastValue) * 100 if pastValue != 0 else 0
                lowestValue = rawSeries.iloc[closestDateIdx:].min()
                highestValue = rawSeries.iloc[closestDateIdx:].max()

                pastPriceStr = cleanNumber(pastValue, seriesData["numType"])
                priceChangePctStr = cleanNumber(pctChange, NumberType.PERCENTAGE_CHANGE)
                lowestPriceStr = cleanNumber(lowestValue, seriesData["numType"])
                highestPriceStr = cleanNumber(highestValue, seriesData["numType"])

                outputsPerPeriod[label] = {                    
                    f"valueAtPeriodStart_{label}": pastPriceStr,
                    f"pctChange_{label}": priceChangePctStr,
                    f"lowest_{label}": lowestPriceStr,
                    f"highest_{label}": highestPriceStr
                }

            if not hasIndexError:
                jsonResult[seriesName] = {
                    "description": seriesDesc,
                    "unit": seriesUnit,
                    "latestValue": latestValueStr,
                    "history": outputsPerPeriod
                }

    return cleanData(jsonResult)

        

def fetchMacroNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, limit: int = 15):
    useLocal = isLocalDataAvailable(timestamp)
    limit = min(max(limit, 1), 50)

    jsonResult = []  

    if useLocal:
        newsWithContent = data.news.getRecentNewsForTickers(
            tickers=["SPY", "QQQ", "DIA", "GLD", "SLV", "VIX", "USO", "TLT"],
            before=timestamp,
            limit=limit,
            mustHaveContent=True
        )

        idx = 0
        if not newsWithContent.empty:
            for _, row in newsWithContent.iterrows():
                headline = row.get("headline", "").strip()
                if not headline:
                    continue
                else:
                    headline = cleanHtmlContent(headline)

                content = row.get("content", "").strip()
                if not content:
                    content = "Article has no content available."
                else:
                    content = cleanHtmlContent(content)

                author = row.get("author", "").strip()

                articleTimestamp = row["updated_at"].replace(tzinfo=timestamp.tzinfo)
                age = timestamp - articleTimestamp
                if age < pd.Timedelta(hours=1):
                    ageStr = f"{age.components.minutes}m old"
                if age < pd.Timedelta(days=1):
                    ageStr = f"{age.components.hours}h {age.components.minutes}m old"
                else:
                    ageStr = f"{age.components.days}d old"

                articleId = str(row.get("id", ""))
                year = articleTimestamp.year % 100
                month = articleTimestamp.month
                url = f"https://www.benzinga.com/news/{year:02d}/{month:02d}/{articleId}"

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
            jsonResult.append({"error": "No local news data available."})


    else:
        # Retrieve via Alpaca API
        apiKeyId = os.getenv("ALPACA_API_KEY_ID")
        apiKeySecret = os.getenv("ALPACA_API_SECRET_KEY")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        macroSymbols = "SPY,QQQ,DIA,GLD,SLV,VIX,USO,TLT"
        url = (
            f"https://data.alpaca.markets/v1beta1/news"
            f"?sort=desc&symbols={macroSymbols}&limit={limit}"
            f"&include_content=true&exclude_contentless=true&sort=desc"
            f"&start={(timestamp - pd.Timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')}-0400"
            f"&end={timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}-0400"
        )

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            rawNews = response.json().get("news", [])
        except requests.exceptions.RequestException as e:
            jsonResult.append({"error": f"Error fetching news from Alpaca API: {str(e)}"})
            return cleanData(jsonResult)
        
        for idx, article in enumerate(rawNews):
            headline = article.get("headline", "").strip()
            if not headline:
                continue
            else:
                headline = cleanHtmlContent(headline)

            content = article.get("content", "").strip()
            if not content:
                content = "Article has no content available."       # Should not occur due to exclude_contentless=true
            else:
                content = cleanHtmlContent(content)

            author = article.get("author", "").strip()

            articleTimestamp = pd.Timestamp(article.get("updated_at")).tz_convert(timestamp.tzinfo)
            age = timestamp - articleTimestamp
            if age < pd.Timedelta(hours=1):
                ageStr = f"{age.components.minutes}m old"
            elif age < pd.Timedelta(days=1):
                ageStr = f"{age.components.hours}h {age.components.minutes}m old"
            else:
                ageStr = f"{age.components.days}d old"

            url = article.get("url", "")

            jsonResult.append({
                "index": idx,
                "headline": headline,
                "content": content,
                "author": author,
                "articleAge": ageStr,
                "url": url
            }) 


    if len(jsonResult) < limit:
        jsonResult.append({"info": f"Only {len(jsonResult)} articles could be retrieved."})

    return cleanData(jsonResult)