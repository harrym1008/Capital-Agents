import os
import threading
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from tools.functions.company import fetchStockPricePerformance

from tools.tool_registry import DataProviders, Tool
from tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, NumberType


# Singleton model pipeline and re-entrant lock
sentimentPipeline = None
cudaAvailable = None
sentimentLock = threading.RLock()

def getModernFinbertPipeline():
    global sentimentPipeline, cudaAvailable

    with sentimentLock:
        if sentimentPipeline is None:
            import torch
            from transformers import pipeline

            if cudaAvailable is None:
                cudaAvailable = torch.cuda.is_available()

            deviceIndex = 0 if cudaAvailable else -1

            modelDir = "data/models"
            sentimentPipeline = pipeline(
                "text-classification",
                model="tabularisai/ModernFinBERT",
                device=deviceIndex,
                dtype=torch.float16 if cudaAvailable else torch.float32,
                cache_dir=modelDir,
                truncation=True,
                max_length=3172
            )

            # Force PyTorch/CUDA kernel compilation and memory buffer allocation during preloading
            try:
                _ = sentimentPipeline(["Financial market sentiment analysis initialisation warmup."])
            except Exception:
                pass

        return sentimentPipeline


def preloadSentimentModelAsync():
    def loadWorker():
        try:
            getModernFinbertPipeline()
        except Exception as e:
            print(f"[Sentiment Preloader] Preload encountered an issue: {e}")

    thread = threading.Thread(target=loadWorker, daemon=True, name="SentimentModelPreloader")
    thread.start()
    return thread

# Preloading can be explicitly invoked after VRAM clearing (e.g. in rudimentaryVramClear or ServerManager)
# preloadSentimentModelAsync()


def scoreHeadlinesBatch(headlines: list[str]) -> list[dict]:
    classifier = getModernFinbertPipeline()
    batchSize = 64 if cudaAvailable else 8
    predictions = classifier(headlines, batch_size=batchSize)
    return predictions


def normaliseTs(ts: pd.Timestamp) -> pd.Timestamp:
    tsNorm = pd.Timestamp(ts)
    if tsNorm.tzinfo is not None:
        tsNorm = tsNorm.tz_convert("UTC").tz_localize(None)
    return tsNorm


def getHeadlineWeight(articleRow: pd.Series, bestMinTickers=2) -> float:
    # Weigh articles by the number of tickers mentioned and the length of the article content
    numTickers = len(articleRow["tickers"])
    contentLength = len(articleRow["content"].strip())

    if contentLength == 0:
        score = 1.2     # Lots of analyst ratings have no content... analyst ratings are important (also weigh more because they will not get additional content rating)
    elif contentLength < 300:     
        score = 0.70   
    elif contentLength < 1000:
        score = 0.85  
    elif contentLength > 6000:
        score = 0.75     # Too long, likely to be too complex for the model to accurately classify 
    else:
        score = 1.0

    if numTickers <= bestMinTickers:
        return score
    return score * 0.89 ** (numTickers - bestMinTickers) 


def getContentWeight(articleRow: pd.Series, bestMinTickers=1) -> float:
    numTickers = len(articleRow["tickers"])
    contentLength = len(articleRow["content"].strip())

    # Base weights are 
    if contentLength < 300:
        score = 0.16
    elif contentLength < 1000:
        score = 0.28
    elif contentLength > 6000:
        score = 0.22     # Too long, likely to be too complex for the model to accurately classify  
    else:
        score = 0.33    # Between 1200 and 6000 chars, good length for the model

    if numTickers <= bestMinTickers:
        return score
    return score * 0.82 ** (numTickers - bestMinTickers)


def getClassificationDf(newsDf: pd.DataFrame, bestMinTickers: int = 2, contentTruncate=600) -> list[str]:
    headlines = pd.DataFrame({
        "articleIndex": newsDf.index,
        "date": newsDf["date"],
        "text": newsDf["headline"].str.strip(),
        "type": "headline",
        "weight": newsDf.apply(getHeadlineWeight, axis=1, bestMinTickers=bestMinTickers)
    })

    if contentTruncate is False:
        return headlines

    contents = newsDf.loc[newsDf["content"].str.strip().str.len() > 5].copy()
    contents = pd.DataFrame({
        "articleIndex": contents.index,
        "date": contents["date"],
        "text": contents["content"].str.strip().str.slice(0, contentTruncate),
        "type": "content",
        "weight": contents.apply(getContentWeight, axis=1, bestMinTickers=bestMinTickers)
    })

    return pd.concat([headlines, contents], ignore_index=True)


def fetchTickerSentimentHistory(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    ticker = ticker.upper()
    baseTs = normaliseTs(timestamp)

    windows = {
        "1mo": baseTs - pd.DateOffset(months=1),
        "3mo": baseTs - pd.DateOffset(months=3),
        "6mo": baseTs - pd.DateOffset(months=6),
        "12mo": baseTs - pd.DateOffset(months=12)
    }

    with sentimentLock:
        cacheKey = f"sentiment|history_{ticker}_{baseTs.strftime('%Y-%m-%dH%H')}"
        cachedResult = data.cache.get(cacheKey)
        if cachedResult is not None:
            return cachedResult

        newsDf = data.news.getNewsForTickerBetweenTimes(
            ticker, 
            start=windows["12mo"] - pd.DateOffset(days=1), 
            end=timestamp, 
            mustHaveContent=False, 
            maxReferencedTickers=5
        )

        if newsDf is None or newsDf.empty:
            return {"error": f"No news data available for ticker {ticker} in the last 12 months."}

        # Ensure newsDf["date"] is tz-naive for consistent comparisons
        if newsDf["date"].dt.tz is not None:
            newsDf["date"] = newsDf["date"].dt.tz_convert("UTC").dt.tz_localize(None)

        classificationDf = getClassificationDf(newsDf, bestMinTickers=2, contentTruncate=4096)
        rawPredictions = scoreHeadlinesBatch(classificationDf["text"].tolist())

        classificationDf["sentimentLabel"] = [pred["label"].lower() for pred in rawPredictions]
        classificationDf["sentimentScore"] = [pred["score"] for pred in rawPredictions]

        classificationDf["netScore"] = classificationDf["sentimentScore"] * \
                                    classificationDf["sentimentLabel"].map({"bullish": 1, "bearish": -1, "neutral": 0})

        netSentimentScores = {}
        breakdowns = {}

        for windowName, startDate in windows.items():
            windowDf = classificationDf[classificationDf["date"] >= startDate]        

            posCount = (windowDf["sentimentLabel"] == "bullish").sum()
            negCount = (windowDf["sentimentLabel"] == "bearish").sum()
            neuCount = (windowDf["sentimentLabel"] == "neutral").sum()

            totalWeight = windowDf["weight"].sum()
            if windowDf.empty or totalWeight == 0:
                weightedMean = 0.0
            else:
                weightedMean = (windowDf["netScore"] * windowDf["weight"]).sum() / totalWeight

            netSentimentScores[windowName] = round(float(weightedMean), 4)
            breakdowns[windowName] = {"bullish": int(posCount), "bearish": int(negCount), "neutral": int(neuCount)}


        momentum = round(netSentimentScores["1mo"] - netSentimentScores["6mo"], 3)
        if momentum > 0.30:
            momentumLabel = "Heavily optimistic"
        elif momentum > 0.20:
            momentumLabel = "Moderately optimistic"
        elif momentum > 0.10:
            momentumLabel = "Slightly optimistic"
        elif momentum < -0.30:
            momentumLabel = "Heavily pessimistic"
        elif momentum < -0.20:
            momentumLabel = "Moderately pessimistic"
        elif momentum < -0.10:
            momentumLabel = "Slightly pessimistic"
        else:
            momentumLabel = "Stable neutral sentiment"

        result = {
            "ticker": ticker,
            "timestamp": baseTs.strftime("%Y-%m-%d %H:%M:%S"),
            "netSentimentScores": netSentimentScores,
            "sentimentMomentum": {
                "value": momentum,
                "label": momentumLabel
            },
            "articleCountsPerWindow": breakdowns,
        }

        data.cache.put(cacheKey, result)
        return cleanData(result)


def fetchSentimentDivergence(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    ticker = ticker.upper()
    baseTs = normaliseTs(timestamp)

    with sentimentLock:
        cacheKey = f"sentiment|divergence_{ticker}_{baseTs.strftime('%Y-%m-%dH%H')}"
        cachedResult = data.cache.get(cacheKey)
        if cachedResult is not None:
            return cachedResult

        perfData = fetchStockPricePerformance(tool, data, timestamp, ticker)
        try:
            returnsDict = perfData.get("returns", {})
            return3moVal = returnsDict.get("3m", returnsDict.get("3mo", "0"))
            return3mo = float(str(return3moVal).replace("%", ""))

            sentimentHistory = fetchTickerSentimentHistory(tool, data, timestamp, ticker)
            if isinstance(sentimentHistory, dict) and "error" in sentimentHistory:
                return sentimentHistory
            sentiment3mo = float(sentimentHistory["netSentimentScores"]["3mo"])

            divergenceGap = round(sentiment3mo - return3mo, 4)
            signal = "Neutral"
            description = "Stock price movement is aligned with news sentiment."

            if return3mo < 10 and sentiment3mo > 0.12:
                signal = "Bullish Divergence (Oversold)"
                description = "Stock price is underperforming relative to bullish news sentiment."
            elif return3mo > 10 and sentiment3mo < -0.12:
                signal = "Bearish Divergence (Overbought)"
                description = "Stock price is outperforming relative to bearish news sentiment."

            result = {
                "ticker": ticker,
                "timestamp": baseTs.strftime("%Y-%m-%d %H:%M:%S"),
                "stockReturns3mo": return3mo,
                "finbertSentiment3mo": sentiment3mo,
                "divergenceGap": divergenceGap,
                "divergenceSignal": signal,
                "signalDescription": description
            }

            data.cache.put(cacheKey, result)
            return cleanData(result)

        except Exception as e:
            return {"error": f"Failed to get data: {str(e)}"}



def fetchMacroSentimentHistory(tool: Tool, data: DataProviders, timestamp: pd.Timestamp):
    baseTs = normaliseTs(timestamp)

    windows = {
        "1w": baseTs - pd.DateOffset(weeks=1),
        "1mo": baseTs - pd.DateOffset(months=1),
        "3mo": baseTs - pd.DateOffset(months=3),
        "6mo": baseTs - pd.DateOffset(months=6),
        "12mo": baseTs - pd.DateOffset(months=12),
        # "2y": baseTs - pd.DateOffset(years=2)
    }

    with sentimentLock:
        cacheKey = f"sentiment|history_macro_{baseTs.strftime('%Y-%m-%dH%H')}"
        cachedResult = data.cache.get(cacheKey)
        if cachedResult is not None:
            return cachedResult

        newsDf = data.news.getNewsForTickersBetweenTimes(
            ["SPY", "QQQ", "DIA", "GLD", "SLV", "VIX", "USO", "TLT"], 
            start=windows["12mo"] - pd.DateOffset(days=1), 
            end=timestamp, 
            mustHaveContent=True, 
            maxReferencedTickers=12
        )

        if newsDf is None or newsDf.empty:
            return {"error": f"No macro news data available for the last 2 years."}

        # Ensure newsDf["date"] is tz-naive for consistent comparisons
        if newsDf["date"].dt.tz is not None:
            newsDf["date"] = newsDf["date"].dt.tz_convert("UTC").dt.tz_localize(None)

        classificationDf = getClassificationDf(newsDf, bestMinTickers=8, contentTruncate=512)   # Allow lots more tickers for macro sentiment given more are requested
        classificationDf["weight"] = 1.0
        rawPredictions = scoreHeadlinesBatch(classificationDf["text"].tolist())                 # Truncate article content to 512 chars (otherwise, much too slow)

        classificationDf["sentimentLabel"] = [pred["label"].lower() for pred in rawPredictions]
        classificationDf["sentimentScore"] = [pred["score"] for pred in rawPredictions]

        classificationDf["netScore"] = classificationDf["sentimentScore"] * \
                                    classificationDf["sentimentLabel"].map({"bullish": 1, "bearish": -1, "neutral": 0})

        netSentimentScores = {}
        breakdowns = {}

        endDate = baseTs
        for windowName, startDate in windows.items():
            windowDf = classificationDf[(classificationDf["date"] >= startDate) & (classificationDf["date"] <= endDate)]
            # endDate = startDate

            posCount = (windowDf["sentimentLabel"] == "bullish").sum()
            negCount = (windowDf["sentimentLabel"] == "bearish").sum()
            neuCount = (windowDf["sentimentLabel"] == "neutral").sum()

            totalWeight = windowDf["weight"].sum()
            if windowDf.empty or totalWeight == 0:
                weightedMean = 0.0
            else:
                neutralTotalWeight = windowDf.loc[windowDf["sentimentLabel"] == "neutral", "weight"].sum()
                nonNeutralWeight = totalWeight - neutralTotalWeight
                if nonNeutralWeight == 0:
                    weightedMean = 0.0
                else:
                    weightedMean = (windowDf["netScore"] * windowDf["weight"]).sum() / nonNeutralWeight

            netSentimentScores[windowName] = round(float(weightedMean), 4)
            breakdowns[windowName] = {"bullish": int(posCount), "bearish": int(negCount), "neutral": int(neuCount)}


        momentum = round(netSentimentScores["1mo"] - netSentimentScores["6mo"], 3)
        # Classify momentum more finely than ticker, given single tickers are more volatile than macro sentiment which is generally more stable
        if momentum > 0.24:
            momentumLabel = "Heavily optimistic"
        elif momentum > 0.16:
            momentumLabel = "Moderately optimistic"
        elif momentum > 0.08:
            momentumLabel = "Slightly optimistic"
        elif momentum < -0.24:
            momentumLabel = "Heavily pessimistic"
        elif momentum < -0.16:
            momentumLabel = "Moderately pessimistic"
        elif momentum < -0.08:
            momentumLabel = "Slightly pessimistic"
        else:
            momentumLabel = "Stable neutral sentiment"

        result = {
            "timestamp": baseTs.strftime("%Y-%m-%d %H:%M:%S"),
            "netSentimentScores": netSentimentScores,
            "sentimentMomentum": {
                "value": momentum,
                "label": momentumLabel
            },
            "articleCountsPerWindow": breakdowns,
        }

        data.cache.put(cacheKey, result)
        return cleanData(result)