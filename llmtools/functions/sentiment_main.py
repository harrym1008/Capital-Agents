import hashlib
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import pandas as pd
 
from llmtools.tool_registry import DataProviders
from finbert.finbert_engines import getSentimentEngine, logitsToPredictions


SENTIMENT_LABEL_WEIGHTS = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
FULL_WEIGHT_MAX_DAYS = 60
DECAY_DURATION_DAYS = 180


# Compute SHA256 hex digest for input text caching
def getTextHash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Clear PyTorch CUDA cache and invoke garbage collection
def clearTorchCache() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Run batch headline inference through active FinBERT model engine
def scoreHeadlinesBatch(headlines: List[str], onProgressCallback: Optional[Callable] = None) -> Optional[List[Dict[str, Any]]]:
    if not headlines:
        return []

    engine = getSentimentEngine()
    if engine is None:
        return None

    try:
        logits = engine.infer(headlines, onProgressCallback=onProgressCallback)
        return logitsToPredictions(logits)
    except Exception as e:
        print(f"[Sentiment Engine] Inference error: {e}")
        return None
    finally:
        clearTorchCache()


# Score list of texts with in-memory LRU cache lookup to prevent re-computation
def scoreTextsWithCache(texts: List[str], data: Optional[DataProviders] = None, 
                        onProgressCallback: Optional[Callable] = None) -> Optional[List[Dict[str, Any]]]:
    if not texts:
        return []

    results = [None] * len(texts)
    uncachedIndices = []
    uncachedTexts = []

    for i, text in enumerate(texts):
        key = f"sent|{getTextHash(text)}"
        if data is not None:
            cachedVal = data.sentimentCache.get(key)
            if cachedVal is not None:
                results[i] = cachedVal
                if onProgressCallback:
                    onProgressCallback(1)
                continue
        uncachedIndices.append(i)
        uncachedTexts.append(text)

    if uncachedTexts:
        newPredictions = scoreHeadlinesBatch(uncachedTexts, onProgressCallback=onProgressCallback)
        if newPredictions is None:
            return None
        for idx, text, pred in zip(uncachedIndices, uncachedTexts, newPredictions):
            results[idx] = pred
            if data is not None:
                key = f"sent|{getTextHash(text)}"
                data.sentimentCache.put(key, pred)

    return results


# Map numerical net sentiment score into qualitative rating label
def deriveSentimentRating(netScore: float) -> str:
    if netScore > 0.38:
        return "Heavily optimistic"
    elif netScore > 0.24:
        return "Moderately optimistic"
    elif netScore > 0.10:
        return "Slightly optimistic"
    elif netScore > 0.04:
        return "Very slightly optimistic but near margin-of-error"
    
    elif netScore < -0.38:
        return "Heavily pessimistic"
    elif netScore < -0.24:
        return "Moderately pessimistic"
    elif netScore < -0.10:
        return "Slightly pessimistic"
    elif netScore < -0.04:
        return "Very slightly pessimistic but  near margin-of-error"
    else:
        return "Neutral sentiment"


# Convert model label prediction to signed net sentiment score in [-1.0, 1.0]
def predictionToNetScore(pred: Any) -> float:
    if not isinstance(pred, dict):
        return 0.0
    try:
        confidence = float(pred.get("score", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence * SENTIMENT_LABEL_WEIGHTS.get(str(pred.get("label", "")).lower(), 0.0)


# Aggregate sentiment scores applying linear time decay weighting over article age
def aggregateSentiment(predictions: List[Any], dates: List[Any], asOf: Any):
    try:
        base = pd.to_datetime(asOf)
        if base.tzinfo is not None:
            base = base.tz_localize(None)
    except Exception:
        return None, None

    weightedSum = 0.0
    weightSum = 0.0
    for pred, dt in zip(predictions, dates):
        if pred is None or dt is None:
            continue
        try:
            artDate = pd.to_datetime(dt)
            if artDate.tzinfo is not None:
                artDate = artDate.tz_localize(None)
            ageDays = (base - artDate).days
        except Exception:
            ageDays = 0
        try:
            ageDays = float(ageDays)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ageDays):
            continue

        if ageDays <= FULL_WEIGHT_MAX_DAYS:
            weight = 1.0
        elif ageDays <= DECAY_DURATION_DAYS:
            weight = 1.0 - (ageDays - FULL_WEIGHT_MAX_DAYS) / (DECAY_DURATION_DAYS - FULL_WEIGHT_MAX_DAYS)
        else:
            weight = 0.0

        if weight > 0:
            weightedSum += predictionToNetScore(pred) * weight
            weightSum += weight

    if weightSum <= 0:
        return None, None
    finalScore = weightedSum / weightSum
    return finalScore, deriveSentimentRating(finalScore)


# Convert timestamp to timezone-naive UTC timestamp
def normaliseTs(ts: pd.Timestamp) -> pd.Timestamp:
    tsNorm = pd.Timestamp(ts)
    if tsNorm.tzinfo is not None:
        tsNorm = tsNorm.tz_convert("UTC").tz_localize(None)
    return tsNorm
