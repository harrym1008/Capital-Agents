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
from llm.tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, isLocalDataAvailable, formatArticleAge, NumberType 


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
    cacheKey = f"macro|context_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached
    
    useLocal = isLocalDataAvailable(timestamp)
    jsonResult = {}

    if useLocal:
        allSeries = data.macro.getAllSeries()

        todaySnapshot = data.macro.getSnapshot(allSeries, timestamp)
        pastDates = {
            "5d": timestamp - pd.DateOffset(weeks=1),
            "1mo": timestamp - pd.DateOffset(months=1),
            "3mo": timestamp - pd.DateOffset(months=3),
            "6mo": timestamp - pd.DateOffset(months=6),
            "1y": timestamp - pd.DateOffset(years=1),
            "3y": timestamp - pd.DateOffset(years=3),
            "5y": timestamp - pd.DateOffset(years=5)
        }
        pastSnapshots = {name: data.macro.getSnapshot(allSeries, date) for name, date in pastDates.items()}

        for series in allSeries:
            seriesIdentifier = series.parquetName
            seriesOrigin = series.source

            if seriesOrigin == "yfinance":
                seriesDesc = YFINANCE_MACRO_TICKERS[seriesIdentifier]["desc"]
                numberType = NumberType.STOCK_PRICE if "USD" not in seriesIdentifier \
                                    else (NumberType.EXCHANGE_RATE if "BTC" not in seriesIdentifier else NumberType.STOCK_PRICE)
            else:   # FRED series
                seriesDesc = FRED_SERIES_MAP[seriesIdentifier]["desc"]
                numberType = FRED_SERIES_MAP[seriesIdentifier]["numType"]

            latestPrice = todaySnapshot.loc[todaySnapshot["series"] == series, "value"].iloc[0]
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

                pastPrice = snapshot.loc[snapshot["series"] == series, "value"].iloc[0]
                priceChangePct = ((latestPrice - pastPrice) / pastPrice) * 100 if pastPrice != 0 else 0

                if seriesOrigin == "yfinance" and period == "1y":
                    fiftyTwoWeekMin = cleanNumber(data.macro.getLowest(series, pastDates[period], timestamp)[1], numberType)
                    fiftyTwoWeekMax = cleanNumber(data.macro.getHighest(series, pastDates[period], timestamp)[1], numberType)
                
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
                    "unit": FRED_SERIES_MAP[seriesIdentifier]["unit"],
                    "latestValue": latestValueStr,
                    "history": outputsPerPeriod
                }

    else:  # Get from online sources
        # First deal with yfinance
        allSeries = data.macro.getAllSeries()
        yfinanceSeries = [s for s in allSeries if s.source == "yfinance"]

        todayStr = (timestamp + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        threeYearAgoStr = (timestamp - pd.DateOffset(years=3, days=7)).strftime("%Y-%m-%d")
                    
        yfTickerMap ={s.parquetName: YFINANCE_MACRO_TICKERS[s.parquetName]["yfticker"] for s in yfinanceSeries}
        yfTickerTickers = list(set(yfTickerMap.values()))

        data.rateLimiters.yFinanceLimiter.wait()
        batchData = yf.download(
            tickers=yfTickerTickers,
            start=threeYearAgoStr,
            end=todayStr,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False
        )

        for series in yfinanceSeries:
            seriesIdentifier = series.parquetName
            seriesDesc = YFINANCE_MACRO_TICKERS[seriesIdentifier]["desc"]
            yfTicker = YFINANCE_MACRO_TICKERS[seriesIdentifier]["yfticker"]

            tickerData = batchData[yfTicker] if yfTicker in batchData.columns.levels[0] else None
            if tickerData is None or tickerData.empty:
                jsonResult[seriesIdentifier] = {
                    "description": seriesDesc,
                    "error": "Could not download data from yfinance for this series."
                }
                continue

            fullHistory = tickerData.dropna(subset=["Open", "High", "Low", "Close"])
     
            numberType = NumberType.STOCK_PRICE if "USD" not in seriesIdentifier \
                            else (NumberType.EXCHANGE_RATE if "BTC" not in seriesIdentifier else NumberType.STOCK_PRICE)

            latestPrice = fullHistory["Close"].iloc[-1]
            latestPriceStr = cleanNumber(latestPrice, numberType)
            latestDate = fullHistory.index[-1]
            earliestDate = fullHistory.index[0]

            oneYearAgoDate = latestDate - pd.DateOffset(years=1)
            oneYearSlice = fullHistory.loc[fullHistory.index >= oneYearAgoDate]
            
            fiftyTwoWeekMin = oneYearSlice["Low"].min() if not oneYearSlice.empty else None
            fiftyTwoWeekMax = oneYearSlice["High"].max() if not oneYearSlice.empty else None

            timeFramesMap = {
                "5d": pd.DateOffset(days=7),
                "1mo": pd.DateOffset(months=1),
                "3mo": pd.DateOffset(months=3),
                "6mo": pd.DateOffset(months=6),
                "1y": pd.DateOffset(years=1),
                "3y": pd.DateOffset(years=3),
            }

            outputsPerPeriod = {}      

            for timeFrame, offset in timeFramesMap.items():
                targetDate = latestDate - offset
                historicalSlice = fullHistory.loc[fullHistory.index <= targetDate]

                if historicalSlice.empty:
                    outputsPerPeriod[timeFrame] = {"error": f"No data available prior to {targetDate.strftime('%Y-%m-%d')}."}
                    continue

                pastPrice = historicalSlice["Close"].iloc[-1]
                priceChangePct = ((latestPrice - pastPrice) / pastPrice) * 100 if pastPrice != 0 else 0
                
                outputsPerPeriod[timeFrame] = {                    
                    f"price_{timeFrame}_ago": cleanNumber(pastPrice, numberType),
                    f"change_{timeFrame}_pct": cleanNumber(priceChangePct, NumberType.PERCENTAGE_CHANGE)
                }

            jsonResult[seriesIdentifier] = {
                "description": seriesDesc,
                "latestValue": latestPriceStr,
                "history": outputsPerPeriod,
                "52w": {
                    "52wLow": cleanNumber(fiftyTwoWeekMin, numberType),
                    "52wHigh": cleanNumber(fiftyTwoWeekMax, numberType)
                }
            } 

        # Then deal with FRED
        fredClient = Fred(api_key=os.getenv("FRED_API_KEY"))
        periods = {
            "1mo": pd.DateOffset(months=1),
            "3mo": pd.DateOffset(months=3),
            "6mo": pd.DateOffset(months=6),
            "1y": pd.DateOffset(years=1),
            "3y": pd.DateOffset(years=3),
            "5y": pd.DateOffset(years=5)
        }

        # Ensure base timestamp is naive for pandas compatibility
        naiveTimestamp = timestamp.tz_localize(None) if getattr(timestamp, "tz", None) is not None else timestamp

        for seriesName, seriesData in FRED_SERIES_MAP.items():
            seriesId = seriesData["id"]
            seriesDesc = seriesData["desc"]
            seriesUnit = seriesData["unit"]

            observationStart = (naiveTimestamp - pd.DateOffset(years=5, months=6)).strftime("%Y-%m-%d")
            observationEnd = naiveTimestamp.strftime("%Y-%m-%d")

            data.rateLimiters.fredLimiter.wait()
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

            latestPrice = rawSeries.iloc[-1]
            latestValueStr = cleanNumber(latestPrice, seriesData["numType"])

            outputsPerPeriod = {}
            hasIndexError = False

            for label, offset in periods.items():
                if label == "1mo" and not seriesName.startswith("TREAS"):
                    continue
                elif label == "3mo" and seriesName in ["CPI", "CORECPI", "GDP"]:
                    continue

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

                if "TREAS" in seriesName:
                    bpsChange = (latestPrice - pastValue) * 100
                    outputsPerPeriod[label] = {
                        f"price_{label}_ago": cleanNumber(pastValue, seriesData["numType"]),
                        f"change_{label}_bps": cleanNumber(bpsChange, NumberType.CHANGE_BP)
                    }
                else:
                    pctChange = ((latestPrice - pastValue) / pastValue) * 100 if pastValue != 0 else 0
                    outputsPerPeriod[label] = {                    
                        f"price_{label}_ago": cleanNumber(pastValue, seriesData["numType"]),
                        f"change_{label}_pct": cleanNumber(pctChange, NumberType.PERCENTAGE_CHANGE)
                    }

            if not hasIndexError:
                jsonResult[seriesName] = {
                    "description": seriesDesc,
                    "unit": seriesUnit,
                    "latestValue": latestValueStr,
                    "history": outputsPerPeriod
                }

    jsonOutput = cleanData({"date": timestamp.strftime("%Y-%m-%d"), "macroContext": cleanData(jsonResult)})
    data.cache.put(cacheKey, jsonOutput)
    return jsonOutput

def fetchMacroNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, limit: int = 12):
    useLocal = isLocalDataAvailable(timestamp)
    limit = min(max(limit, 1), 18)

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
            jsonResult.append({"error": "No local news data available."})


    else:
        # Retrieve via Alpaca API
        apiKeyId = os.getenv("ALPACA_API_KEY")
        apiKeySecret = os.getenv("ALPACA_API_SECRET")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        url = "https://data.alpaca.markets/v1beta1/news"
 
        params = {
            "start": (timestamp - pd.Timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S-04:00"),
            "end": timestamp.strftime("%Y-%m-%dT%H:%M:%S-04:00"),
            "sort": "desc",
            "symbols": "SPY,QQQ,DIA,GLD,SLV,VIX,USO,TLT",
            "limit": limit,
            "include_content": "true",
            "exclude_contentless": "true",
        }

        try:
            # Check if it is in the cache
            cacheKey = f"alpaca|news_macro_{timestamp.strftime('%Y-%m-%dH%H')}_{limit}"
            cached = data.cache.get(cacheKey)
            if cached is not None:
                rawNews = cached
            else:
                data.rateLimiters.alpacaLimiter.wait()
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                rawNews = response.json().get("news", [])
                data.cache.put(cacheKey, rawNews)

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

            rawDate = article.get("date") or article.get("created_at") or article.get("updated_at")
            ageStr = formatArticleAge(rawDate, timestamp)

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

    return {"date": timestamp.strftime("%Y-%m-%d"), "news": cleanData(jsonResult)}