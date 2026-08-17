import os
import hashlib
import threading
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from llmtools.functions.company import fetchStockPricePerformance

from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, NumberType

from finbert.finbert_engines import getBestInferenceEngine, logitsToPredictions, \
    TrtCudaInferenceEngine, OnnxCudaInferenceEngine, PytorchCudaInferenceEngine


# Singleton model engine and re-entrant lock
sentimentEngine = None
sentimentTokenizer = None
engineType = None
engineLoadAttempted = False
sentimentLock = threading.RLock()


def getSentimentEngine():
    global sentimentEngine, engineType, engineLoadAttempted

    try:
        with sentimentLock:
            if not engineLoadAttempted:
                engineLoadAttempted = True
                engine = getBestInferenceEngine()
                if engine is not None:
                    sentimentEngine = engine
                    engineType = getattr(engine, "engineType", engine.__class__.__name__)
                    try:
                        warmupText = ["Financial market sentiment analysis initialisation warmup."]
                        _ = engine.infer(warmupText)
                    except Exception as warmupError:
                        print(f"[Sentiment Engine] Prewarm encountered an issue: {warmupError}")

            return sentimentEngine, engineType
    except Exception as e:
        print(f"[Sentiment Engine] Error loading engine: {e}")
        return None, None


def getModernFinbertPipeline():
    engine, _ = getSentimentEngine()
    return engine


def preloadSentimentModelAsync():
    thread = threading.Thread(target=getSentimentEngine, daemon=True, name="SentimentModelPreloader")
    thread.start()
    return thread


def getTextHash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clearTorchCache():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def scoreHeadlinesBatch(headlines: list[str]) -> list[dict] | None:
    if not headlines:
        return []

    engine, _ = getSentimentEngine()
    if engine is None:
        return None

    try:
        logits = engine.infer(headlines)
        return logitsToPredictions(logits)
    except Exception as e:
        print(f"[Sentiment Engine] Inference error: {e}")
        return None
    finally:
        clearTorchCache()


def scoreTextsWithCache(texts: list[str], data: DataProviders = None) -> list[dict] | None:
    if not texts:
        return []

    sentimentCache = getattr(data, "sentimentCache", None) if data is not None else None

    results = [None] * len(texts)
    uncachedIndices = []
    uncachedTexts = []

    for i, text in enumerate(texts):
        key = f"sent|{getTextHash(text)}"
        if sentimentCache is not None:
            cachedVal = sentimentCache.get(key)
            if cachedVal is not None:
                results[i] = cachedVal
                continue
        uncachedIndices.append(i)
        uncachedTexts.append(text)

    if uncachedTexts:
        newPredictions = scoreHeadlinesBatch(uncachedTexts)
        if newPredictions is None:
            return None
        for idx, text, pred in zip(uncachedIndices, uncachedTexts, newPredictions):
            results[idx] = pred
            if sentimentCache is not None:
                key = f"sent|{getTextHash(text)}"
                sentimentCache.put(key, pred)

    return results


def getMaxArticlesPerMonth() -> int:
    engine, _ = getSentimentEngine()
    engineType = type(engine)

    if engineType is TrtCudaInferenceEngine:
        return 100
    elif engineType is OnnxCudaInferenceEngine:
        return 50
    elif engineType is PytorchCudaInferenceEngine:
        return 30
    return 12


def sampleMonthlyArticles(newsDf: pd.DataFrame) -> pd.DataFrame:
    if newsDf is None or newsDf.empty:
        return newsDf

    maxPerMonth = getMaxArticlesPerMonth()
    print(maxPerMonth, " max articles per month")

    newsDfCopy = newsDf.copy()
    newsDfCopy["periodGroup"] = newsDfCopy["date"].dt.to_period("M")

    sampledGroups = []
    for _, group in newsDfCopy.groupby("periodGroup"):
        if len(group) > maxPerMonth:
            sampledGroups.append(group.sample(n=maxPerMonth, random_state=42))
        else:
            sampledGroups.append(group)

    resultDf = pd.concat(sampledGroups, ignore_index=True).drop(columns=["periodGroup"])
    return resultDf.sort_values(by="date").reset_index(drop=True)


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
        score = 1.2     # Analyst ratings often have empty content... weigh heavily
    elif contentLength < 300:     
        score = 0.70   
    elif contentLength < 1000:
        score = 0.85  
    elif contentLength > 6000:
        score = 0.75     
    else:
        score = 1.0

    if numTickers <= bestMinTickers:
        return score
    return score * 0.89 ** (numTickers - bestMinTickers) 


def getContentWeight(articleRow: pd.Series, bestMinTickers=1) -> float:
    numTickers = len(articleRow["tickers"])
    contentLength = len(articleRow["content"].strip())

    if contentLength < 300:
        score = 0.16
    elif contentLength < 1000:
        score = 0.28
    elif contentLength > 6000:
        score = 0.22     
    else:
        score = 0.33    

    if numTickers <= bestMinTickers:
        return score
    return score * 0.82 ** (numTickers - bestMinTickers)


def getClassificationDf(newsDf: pd.DataFrame, bestMinTickers: int = 2, contentTruncate=1024) -> pd.DataFrame:
    # Prioritise headlines; only fall back to body content if headline is missing or empty
    headlineText = newsDf["headline"].str.strip()
    hasHeadline = headlineText.str.len() > 0

    headlines = pd.DataFrame({
        "articleIndex": newsDf.index[hasHeadline],
        "date": newsDf.loc[hasHeadline, "date"],
        "text": headlineText[hasHeadline],
        "type": "headline",
        "weight": newsDf[hasHeadline].apply(getHeadlineWeight, axis=1, bestMinTickers=bestMinTickers)
    })

    missingHeadlineMask = ~hasHeadline
    if missingHeadlineMask.any():
        contents = newsDf.loc[missingHeadlineMask & (newsDf["content"].str.strip().str.len() > 5)].copy()
        if not contents.empty:
            contentDf = pd.DataFrame({
                "articleIndex": contents.index,
                "date": contents["date"],
                "text": contents["content"].str.strip().str.slice(0, contentTruncate),
                "type": "content",
                "weight": contents.apply(getContentWeight, axis=1, bestMinTickers=bestMinTickers)
            })
            return pd.concat([headlines, contentDf], ignore_index=True)

    return headlines


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

        # Stratified monthly sampling based on chosen inference engine
        newsDf = sampleMonthlyArticles(newsDf)

        classificationDf = getClassificationDf(newsDf, bestMinTickers=2, contentTruncate=1024)
        rawPredictions = scoreTextsWithCache(classificationDf["text"].tolist(), data=data)
        if rawPredictions is None:
            return {"error": "Not available because sentiment classification models could not be loaded."}

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

        # Stratified monthly sampling based on chosen inference engine
        newsDf = sampleMonthlyArticles(newsDf)

        classificationDf = getClassificationDf(newsDf, bestMinTickers=8, contentTruncate=1024)
        classificationDf["weight"] = 1.0
        rawPredictions = scoreTextsWithCache(classificationDf["text"].tolist(), data=data)
        if rawPredictions is None:
            return {"error": "Not available because sentiment classification models could not be loaded."}

        classificationDf["sentimentLabel"] = [pred["label"].lower() for pred in rawPredictions]
        classificationDf["sentimentScore"] = [pred["score"] for pred in rawPredictions]

        classificationDf["netScore"] = classificationDf["sentimentScore"] * \
                                     classificationDf["sentimentLabel"].map({"bullish": 1, "bearish": -1, "neutral": 0})

        netSentimentScores = {}
        breakdowns = {}

        endDate = baseTs
        for windowName, startDate in windows.items():
            windowDf = classificationDf[(classificationDf["date"] >= startDate) & (classificationDf["date"] <= endDate)]

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