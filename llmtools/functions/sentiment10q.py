import re
import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from edgar import Filing
from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, NumberType
from dataquery.edgar_provider import EdgarDataProvider, FormType, CompanyRef
from llmtools.functions.sentiment import (
    scoreTextsWithCache,
    clearTorchCache,
    getSentimentEngine,
    logitsToPredictions,
    deriveSentimentRating
)


ABBREVIATIONS_PATTERN = (
    r"(?:Inc|Corp|Ltd|Co|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"vs|approx|e\.g|i\.e|No|Vol|pp|Fig|al|U\.S|SEC|GAAP|Q[1-4]|FY\d{2,4})\."
)


def cleanFilingText(text: str) -> str:
    text = cleanHtmlContent(text)       # Clean content like it is HTML (fixes bad UTF codes)
    text = re.sub(r"[\r\n\t]+", " ", text)      # Replace newlines and tabs with spaces
    text = re.sub(r"\s{2,}", " ", text)         # Collapse multiple spaces into one
    return text.strip()


def splitIntoSentences(text: str, minCharLen: int = 35, minWords: int = 5) -> List[str]:
    if not text:
        return []
    cleaned = cleanFilingText(text)

    # Replace dots that are part of common abbreviations with a placeholder to avoid splitting on them
    def protectAbbrevation(m: re.Match) -> str:
        return m.group(0).replace(".", "__DOT__")

    # Protect common abbreviations and decimal numbers
    protected = re.sub(ABBREVIATIONS_PATTERN, protectAbbrevation, cleaned, flags=re.IGNORECASE)
    protected = re.sub(r"(\d+)\.(\d+)", r"\1__DOT__\2", protected)

    # Split on sentence terminals followed by whitespace and capital letter/quote/digit/bracket (not on protected abbreviations)
    rawChunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(])", protected)

    sentences = []
    seen = set()
    for chunk in rawChunks:
        s = chunk.replace("__DOT__", ".").strip()   # Restore dots in abbreviations and numbers
        words = s.split()
        if len(s) >= minCharLen and len(words) >= minWords:
            alphaCount = sum(c.isalpha() for c in s)
            # String must be over 60% alphabetic chars to be considered a sentence
            # If it isnt, it is likely a table or list of numbers, skip it
            if (alphaCount / len(s)) > 0.60:
                sNormalized = re.sub(r"\s+", " ", s).strip()
                if sNormalized not in seen:
                    seen.add(sNormalized)
                    sentences.append(sNormalized)
    return sentences


def extractSectionFromTenQ(filing: Filing, data: DataProviders, sectionType: str) -> Optional[str]:
    filingObj, _ = data.edgar.downloadFilingObjects(filing)
    if filingObj is None:
        return ""
    return filingObj[sectionType]


def distillSectionSentences(rawText: str, data: DataProviders, confidenceThreshold: float = 0.24, maxWords: int = 800) -> tuple[str, list[float]]:
    if not rawText:
        return "", []

    sentences = splitIntoSentences(rawText)
    predictions = scoreTextsWithCache(sentences, data=data)
    if predictions is None:
        return "", []

    retainedSentences = []
    netScores = []

    for sent, pred in zip(sentences, predictions):
        label = pred["label"].lower()
        confidence = float(pred["score"])
        netScore = confidence if label == "bullish" else (-confidence if label == "bearish" else 0.0)
        netScores.append(netScore)

        isDecisive = (label in ["bullish", "bearish"] and confidence >= confidenceThreshold)
        if isDecisive:
            retainedSentences.append(sent)

    distilledText = " ".join(retainedSentences)
    words = distilledText.split(" ")

    if len(words) > maxWords:
        words = words[:maxWords]
        cappedText = " ".join(words) + " ... [text truncated]"
    else:
        cappedText = distilledText
    
    return cappedText, netScores


def fetchLatest10QSentiment(tool: Optional[Tool], data: DataProviders, timestamp: pd.Timestamp, ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper().strip()
    companyRef = CompanyRef(ticker)

    filing = data.edgar.getLatestFilingRef(companyRef, formType=FormType.FORM_10Q, before=timestamp)
    if filing is None:
        return f"No valid 10-Q report found for {ticker} on or before {timestamp}."

    confThreshold = 0.24
    maxWordsPerBlock = 800

    accessionNumber = filing.accession_number
    cacheKey = f"sentiment10q_mda_rf_{ticker}_{accessionNumber}_conf{confThreshold}_maxWords{maxWordsPerBlock}"
    cachedResult = data.cache.get(cacheKey)
    if cachedResult is not None:
        return cachedResult

    mdaRaw = extractSectionFromTenQ(filing, data, sectionType="Part I, Item 2")     # MD&A is typically "Part I, Item 2"
    rfRaw = extractSectionFromTenQ(filing, data, sectionType="Part II, Item 1A")    # Risk Factors is typically "Part II, Item 1A"

    if not mdaRaw and not rfRaw:
        return f"Could not extract MD&A or Risk Factors from 10-Q filing ({accessionNumber}) for {ticker}."

    # Distill MD&A and compute operational sentiment
    distilledMda, mdaNetScores = distillSectionSentences(
        mdaRaw, data=data, confidenceThreshold=confThreshold, maxWords=maxWordsPerBlock
    )
    distilledRf, rfNetScores = distillSectionSentences(
        rfRaw, data=data, confidenceThreshold=confThreshold, maxWords=maxWordsPerBlock
    )

    mdaNetMean = round(float(np.mean(mdaNetScores)), 4) if mdaNetScores else 0.0
    ratingLabel = deriveSentimentRating(mdaNetMean)

    filingPeriod = filing.period_of_report if filing else None
    filingDate = filing.filing_date if filing else None

    result = cleanData({
        "ticker": ticker,
        "company": getattr(filing, "company", None),
        "filingDate": filingDate,
        "periodEnded": filingPeriod,

        "operationalSentiment": {
            "rating": ratingLabel,
            "netScore": mdaNetMean,
        },

        "md&a": distilledMda,
        "riskFactors": distilledRf,

        "note": "The 'md&a' and 'riskFactors' texts are distilled using FinBERT to extract decisive operational and risk statements "
                "from the original full texts in the 10-Q filing. The 'operationalSentiment' is derived from the MD&A section's net sentiment score."
    })

    data.cache.put(cacheKey, result)
    return result
