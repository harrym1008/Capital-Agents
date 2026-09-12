import math
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanData, cleanNumber, NumberType
from collectors.sector_dl_client import GICS_SECTORS


def confirmBoardroomDecisionBase(
    tool: Tool, 
    ticker: str, 
    rating: str, 
    weighting: str, 
    targetKey1: str, 
    targetVal1: float, 
    targetKey2: str, 
    targetVal2: float
) -> Dict[str, Any]:
    try:
        rating = rating.upper()
        weighting = weighting.upper()

        if rating not in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            return {"error": f"Invalid rating value: {rating}. Must be one of STRONG BUY, BUY, HOLD, SELL, STRONG SELL."}
        if weighting not in ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"]:
            return {"error": f"Invalid weighting value: {weighting}. Must be one of UNDERWEIGHT, EQUAL-WEIGHT, OVERWEIGHT."}

        cleanedVal1 = cleanNumber(targetVal1, NumberType.STOCK_PRICE)[1:]
        cleanedVal2 = cleanNumber(targetVal2, NumberType.STOCK_PRICE)[1:]

        summaryLogStr = f"Boardroom decision confirmed for {ticker}: Rating: {rating}, Weighting: {weighting}, {targetKey1}: ${cleanedVal1}, {targetKey2}: ${cleanedVal2}."

        result = {
            "status": "success",
            "ticker": ticker,
            "rating": rating,
            "weighting": weighting,
            targetKey1: cleanedVal1,
            targetKey2: cleanedVal2
        }
        tool.toolLog.append(result | {"summary": summaryLogStr, "targets": [targetVal1, targetVal2]})
        tool.toolLog.append(summaryLogStr)

        return cleanData(result)
    except Exception as e:
        return {"error": f"An error occurred while confirming boardroom decision: {str(e)}"}


def confirmBoardroomDecisionImmediateTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp,
                                      ticker: str, rating: str, weighting: str, threeDayTarget: float, twoWeekTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeDayTarget", threeDayTarget, "twoWeekTarget", twoWeekTarget)


def confirmBoardroomDecisionShortTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                      ticker: str, rating: str, weighting: str, oneMonthTarget: float, threeMonthTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "oneMonthTarget", oneMonthTarget, "threeMonthTarget", threeMonthTarget)


def confirmBoardroomDecisionMediumTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                       ticker: str, rating: str, weighting: str, threeMonthTarget: float, twelveMonthTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeMonthTarget", threeMonthTarget, "twelveMonthTarget", twelveMonthTarget)


def confirmBoardroomDecisionLongTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                     ticker: str, rating: str, weighting: str, twelveMonthTarget: float, threeYearTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "twelveMonthTarget", twelveMonthTarget, "threeYearTarget", threeYearTarget)


def confirmBoardroomDecisionDistantTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                        ticker: str, rating: str, weighting: str, threeYearTarget: float, tenYearTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeYearTarget", threeYearTarget, "tenYearTarget", tenYearTarget)


def confirmSectorAllocation(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorAllocations: Dict[str, float], rationale: str) -> Dict[str, Any]:
    if not isinstance(sectorAllocations, dict) or not sectorAllocations:
        return {"error": "sectorAllocations must be a non-empty dictionary mapping sector names to percentage numbers."}

    cleanedAllocations = {}
    totalAllocated = 0.0

    for rawSector, rawPct in sectorAllocations.items():
        try:
            pctVal = round(float(rawPct), 2)
        except (ValueError, TypeError):
            return {"error": f"Allocation value for '{rawSector}' must be a valid number, got '{rawPct}'."}

        if pctVal <= 0:
            continue

        rawLower = str(rawSector).strip().lower()
        if rawLower in ["cash", "usd"]:
            cleanedAllocations["CASH"] = {
                "sector": "Cash",
                "ticker": "CASH",
                "allocationPct": pctVal
            }
            totalAllocated += pctVal
            continue

        ticker, resolvedName, note = data.sectors.resolveSector(rawSector)
        if resolvedName == "Unknown" or not ticker:
            return {"error": f"Sector '{rawSector}' could not be resolved to a valid GICS sector. Provided note: {note}"}

        cleanedAllocations[ticker] = {
            "sector": resolvedName,
            "ticker": ticker,
            "allocationPct": pctVal
        }
        totalAllocated += pctVal

    # Check if total sums to ~100%
    totalAllocated = round(totalAllocated, 2)
    if totalAllocated < 95.0 or totalAllocated > 105.0:
        return {
            "error": f"Total sector allocations must sum to approximately 100.0%. Current sum: {totalAllocated}%. Please rebalance and retry.",
            "currentSum": totalAllocated,
            "currentAllocations": cleanedAllocations
        }

    decisionRecord = {
        "sectorAllocations": cleanedAllocations,
        "totalAllocatedPct": totalAllocated,
        "sectorCount": len([k for k in cleanedAllocations if k != "CASH"]),
        "rationale": str(rationale).strip()
    }

    tool.toolLog.append(decisionRecord)
    return cleanData({
        "status": "success",
        "message": f"Sector allocation confirmed with {len(cleanedAllocations)} sectors totaling {totalAllocated}%.",
        "confirmedAllocation": decisionRecord
    })


def confirmPortfolioAllocation(
    tool: Tool, 
    data: DataProviders, 
    timestamp: pd.Timestamp, 
    positions: List[Dict[str, Any]], 
    portfolioRationale: str,
    cashWeightPct: float = 0.0,
    initialCapital: float = 100_000.0
) -> Dict[str, Any]:
    if not isinstance(positions, list) or len(positions) == 0:
        return {"error": "positions must be a non-empty list of individual stock allocation objects."}

    cleanedPositions = []
    totalStockPct = 0.0
    sectorSubtotals: Dict[str, float] = {}

    for i, pos in enumerate(positions):
        if not isinstance(pos, dict):
            return {"error": f"Position at index {i} must be a dictionary object."}

        rawTicker = str(pos.get("ticker", "")).strip().upper()
        if not rawTicker:
            return {"error": f"Position at index {i} is missing a valid 'ticker'."}

        rawPct = pos.get("weightPct") or pos.get("allocationPct") or pos.get("weight")
        try:
            pctVal = round(float(rawPct), 2)
        except (ValueError, TypeError):
            return {"error": f"Position '{rawTicker}' has invalid weight percentage: '{rawPct}'."}

        if pctVal <= 0:
            continue

        # Validate ticker existence against company database
        profile = data.tickers.getTickerProfile(rawTicker)
        if profile is None:
            return {"error": f"Ticker '{rawTicker}' does not exist in the universe or is not recognised. Please replace it with a valid traded equity."}

        companyName = profile.name or rawTicker
        rawSector = (profile.sector or "").strip()
        secTicker, resolvedSector, _ = data.sectors.resolveSector(rawSector or rawTicker)
        sectorName = resolvedSector if resolvedSector and resolvedSector != "Unknown" else (rawSector or "Equities")
        industryName = profile.industry.replace("_", " ").title() if profile.industry else "General Equities"

        rationale = str(pos.get("rationale", "")).strip()
        dollarAllocation = round(float(initialCapital) * (pctVal / 100.0), 2)
        latestPrice = None

        try:
            dayData = data.ohlcv.getSingleDayTickerData(rawTicker, timestamp)
            if dayData is not None and "close" in dayData and pd.notna(dayData["close"]):
                latestPrice = round(float(dayData["close"]), 2)
        except Exception:
            pass

        compRecord = {
            "ticker": rawTicker,
            "companyName": companyName,
            "sector": sectorName,
            "sectorEtf": secTicker,
            "industry": industryName,
            "weightPct": pctVal,
            "dollarAllocation": dollarAllocation,
            "latestPrice": latestPrice,
            "rationale": rationale
        }
        cleanedPositions.append(compRecord)
        totalStockPct += pctVal

        sectorSubtotals[sectorName] = round(sectorSubtotals.get(sectorName, 0.0) + pctVal, 2)

    try:
        cashPct = max(0.0, round(float(cashWeightPct), 2))
    except (ValueError, TypeError):
        cashPct = 0.0

    totalAllocated = round(totalStockPct + cashPct, 2)
    if totalAllocated < 95.0 or totalAllocated > 105.0:
        return {
            "error": f"Total portfolio allocation must sum to approximately 100.0% (stocks + cash). Current sum: {totalAllocated}% (Stocks: {totalStockPct}%, Cash: {cashPct}%). Please rebalance and retry.",
            "currentSum": totalAllocated,
            "currentPositions": cleanedPositions
        }

    cashDollar = round(float(initialCapital) * (cashPct / 100.0), 2)

    cashPositionRecord = {
        "weightPct": cashPct,
        "dollarAllocation": cashDollar,
        "etfSubstitute": "SPY"
    }

    portfolioRecord = {
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "initialCapital": float(initialCapital),
        "totalAllocatedPct": totalAllocated,
        "stockCount": len(cleanedPositions),
        "positions": cleanedPositions,
        "cashPosition": cashPositionRecord,
        "sectorBreakdown": sectorSubtotals,
        "portfolioRationale": str(portfolioRationale).strip()
    }

    tool.toolLog.append(portfolioRecord)
    return cleanData({
        "status": "success",
        "message": f"Portfolio creation confirmed: {len(cleanedPositions)} stock positions ({totalStockPct}%) and cash ({cashPct}%) totaling {totalAllocated}%.",
        "confirmedPortfolio": portfolioRecord
    })
