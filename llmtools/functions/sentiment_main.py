import hashlib
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import pandas as pd
 
from llmtools.tool_registry import DataProviders
from finbert.finbert_engines import getSentimentEngine, logitsToPredictions


def getTextHash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clearTorchCache() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass



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


def deriveSentimentRating(netScore: float) -> str:
    if netScore > 0.30:
        return "Heavily optimistic"
    elif netScore > 0.20:
        return "Moderately optimistic"
    elif netScore > 0.10:
        return "Slightly optimistic"
    elif netScore < -0.30:
        return "Heavily pessimistic"
    elif netScore < -0.20:
        return "Moderately pessimistic"
    elif netScore < -0.10:
        return "Slightly pessimistic"
    else:
        return "Stable neutral sentiment"


def normaliseTs(ts: pd.Timestamp) -> pd.Timestamp:
    tsNorm = pd.Timestamp(ts)
    if tsNorm.tzinfo is not None:
        tsNorm = tsNorm.tz_convert("UTC").tz_localize(None)
    return tsNorm
