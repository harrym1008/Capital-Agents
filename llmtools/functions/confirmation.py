import json
import math
from typing import Dict, Any, List, Optional, Union
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


def distributeIntegerPercentages(weights: List[float], totalTarget: int) -> List[int]:
    if not weights or totalTarget <= 0:
        return [0] * len(weights)
    sumWeights = float(sum(weights))
    if sumWeights <= 0:
        base = totalTarget // len(weights)
        rem = totalTarget % len(weights)
        return [base + (1 if i < rem else 0) for i in range(len(weights))]

    rawFloats = [(float(w) / sumWeights) * totalTarget for w in weights]
    floors = [int(math.floor(f)) for f in rawFloats]
    remainders = [(rawFloats[i] - floors[i], i) for i in range(len(weights))]
    remainders.sort(key=lambda x: (-x[0], x[1]))
    shortfall = totalTarget - sum(floors)
    for k in range(shortfall):
        idx = remainders[k][1]
        floors[idx] += 1
    return floors


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

    # Normalise all sector allocations to exact integers totaling 100%
    keys = list(cleanedAllocations.keys())
    rawWeights = [cleanedAllocations[k]["allocationPct"] for k in keys]
    intAllocations = distributeIntegerPercentages(rawWeights, 100)
    for k, intVal in zip(keys, intAllocations):
        cleanedAllocations[k]["allocationPct"] = int(intVal)

    decisionRecord = {
        "sectorAllocations": cleanedAllocations,
        "totalAllocatedPct": 100,
        "sectorCount": len([k for k in cleanedAllocations if k != "CASH"]),
        "rationale": str(rationale).strip()
    }

    tool.toolLog.append(decisionRecord)
    return cleanData({
        "status": "success",
        "message": f"Sector allocation confirmed with {len(cleanedAllocations)} sectors totaling 100%.",
        "confirmedAllocation": decisionRecord
    })


def confirmPortfolioAllocation(
    tool: Tool, 
    data: DataProviders, 
    timestamp: pd.Timestamp, 
    sectorAllocations: Union[Dict[str, List[Dict[str, Any]]], str], 
    portfolioRationale: str,
    initialCapital: float = 100_000.0
) -> Dict[str, Any]:
    if isinstance(sectorAllocations, str):
        try:
            sectorAllocations = json.loads(sectorAllocations)
        except Exception:
            return {"error": "sectorAllocations could not be parsed as valid JSON. Provide a dictionary mapping sector names to lists of stock objects."}

    if not isinstance(sectorAllocations, dict) or len(sectorAllocations) == 0:
        return {"error": "sectorAllocations must be a non-empty dictionary mapping confirmed sector names to lists of stock positions."}

    confirmedSectorTool = tool.registry.getTool("confirmSectorAllocation") if getattr(tool, "registry", None) else None
    if not confirmedSectorTool or not confirmedSectorTool.toolLog:
        return {"error": "No confirmed sector allocation was found in the boardroom session. Sector allocation must be confirmed before finalising individual stock positions."}

    lastSectorDecision = confirmedSectorTool.toolLog[-1]
    confirmedSectorMap = lastSectorDecision.get("sectorAllocations", {})

    confirmedCashPct = 0.0
    expectedSectors: Dict[str, Dict[str, Any]] = {}

    for secKey, secVal in confirmedSectorMap.items():
        if not isinstance(secVal, dict):
            continue
        secTicker = str(secVal.get("ticker", secKey)).strip().upper()
        secName = str(secVal.get("sector", secKey)).strip()
        try:
            allocPct = round(float(secVal.get("allocationPct", 0.0)), 2)
        except (ValueError, TypeError):
            allocPct = 0.0

        if secTicker in ["CASH", "USD"] or secName.lower() in ["cash", "usd"]:
            confirmedCashPct = round(confirmedCashPct + allocPct, 2)
            continue

        if allocPct > 0:
            expectedSectors[secTicker] = {
                "sector": secName,
                "ticker": secTicker,
                "allocationPct": allocPct
            }

    providedSectors: Dict[str, List[Dict[str, Any]]] = {}

    for rawSecKey, stockList in sectorAllocations.items():
        if not isinstance(stockList, list):
            return {"error": f"Allocation for sector '{rawSecKey}' must be a list of stock objects, got {type(stockList).__name__}."}

        secTicker, resolvedName, note = data.sectors.resolveSector(str(rawSecKey).strip())
        if secTicker == "Unknown" or not secTicker:
            return {"error": f"Sector key '{rawSecKey}' could not be resolved to a recognized GICS sector. {note}"}

        secTicker = secTicker.upper()
        if secTicker not in expectedSectors:
            expectedNames = [f"{v['sector']} ({v['ticker']})" for v in expectedSectors.values()]
            return {"error": f"Sector '{rawSecKey}' ({secTicker}) is not in the confirmed sector allocations. Confirmed sectors are: {', '.join(expectedNames)}."}

        providedSectors[secTicker] = stockList

    missingSectors = [f"{v['sector']} ({v['ticker']})" for k, v in expectedSectors.items() if k not in providedSectors]
    if missingSectors:
        return {"error": f"Missing stock allocations for confirmed sector(s): {', '.join(missingSectors)}. You must provide stock holdings for all confirmed sectors."}

    portfolioRationaleStr = str(portfolioRationale or "").strip()
    portfolioWordCount = len(portfolioRationaleStr.split())
    if portfolioWordCount < 35:
        return {"error": f"Portfolio rationale is too brief ({portfolioWordCount} words). Please provide an executive rationale of approximately 100 words explaining portfolio construction and risk management."}

    cleanedPositions = []
    sectorSubtotals: Dict[str, float] = {}

    for secTicker, secInfo in expectedSectors.items():
        stockList = providedSectors[secTicker]
        sectorName = secInfo["sector"]
        sectorTargetPct = secInfo["allocationPct"]

        if len(stockList) == 0:
            return {"error": f"Sector '{sectorName}' ({secTicker}) has an empty stock list. Please provide at least one stock holding for this sector."}

        sectorSumWeight = 0.0
        sectorPositions = []

        for i, pos in enumerate(stockList):
            if not isinstance(pos, dict):
                return {"error": f"Item {i} in sector '{sectorName}' must be a dictionary object."}

            rawTicker = str(pos.get("ticker", "")).strip().upper()
            if not rawTicker:
                return {"error": f"Stock at index {i} in sector '{sectorName}' is missing a valid 'ticker'."}

            rawPerSecWeight = pos.get("perSectorWeight") if "perSectorWeight" in pos else (pos.get("weightPct") or pos.get("weight"))
            try:
                perSecVal = round(float(rawPerSecWeight), 2)
            except (ValueError, TypeError):
                return {"error": f"Stock '{rawTicker}' in sector '{sectorName}' has an invalid perSectorWeight: '{rawPerSecWeight}'."}

            if perSecVal <= 0:
                continue

            profile = data.tickers.getTickerProfile(rawTicker)
            if profile is None:
                return {"error": f"Ticker '{rawTicker}' in sector '{sectorName}' does not exist in the stock universe. Please replace it with a valid traded ticker."}

            stockSecTicker, stockResolvedSector, _ = data.sectors.resolveSector(profile.sector or "")
            if stockSecTicker and stockSecTicker != "Unknown" and stockSecTicker.upper() != secTicker:
                return {
                    "error": f"Ticker '{rawTicker}' belongs to sector '{stockResolvedSector}' ({stockSecTicker}), not '{sectorName}' ({secTicker}). All stocks must be placed in their correct GICS sector."
                }

            companyName = profile.name or rawTicker
            industryName = profile.industry.replace("_", " ").title() if profile.industry else "General Equities"

            rationale = str(pos.get("rationale", "")).strip()

            sectorSumWeight += perSecVal
            sectorPositions.append({
                "ticker": rawTicker,
                "companyName": companyName,
                "sector": sectorName,
                "sectorEtf": secTicker,
                "industry": industryName,
                "perSectorWeight": perSecVal,
                "rationale": rationale
            })

        sectorSumWeight = round(sectorSumWeight, 2)
        if sectorSumWeight < 98.0 or sectorSumWeight > 102.0:
            return {
                "error": f"The perSectorWeight values for sector '{sectorName}' sum to {sectorSumWeight}%, but must sum to 100.0%. Please rebalance the stocks within '{sectorName}' to total 100%."
            }

        sectorIntTarget = int(round(sectorTargetPct))
        stockPerSecWeights = [p["perSectorWeight"] for p in sectorPositions]
        stockIntWeights = distributeIntegerPercentages(stockPerSecWeights, sectorIntTarget)

        for pos, intWeight in zip(sectorPositions, stockIntWeights):
            dollarAllocation = round(float(initialCapital) * (intWeight / 100.0), 2)
            latestPrice = None

            try:
                dayData = data.ohlcv.getSingleDayTickerData(pos["ticker"], timestamp)
                if dayData is not None and "close" in dayData and pd.notna(dayData["close"]):
                    latestPrice = round(float(dayData["close"]), 2)
            except Exception:
                pass

            posRecord = {
                "ticker": pos["ticker"],
                "companyName": pos["companyName"],
                "sector": pos["sector"],
                "sectorEtf": pos["sectorEtf"],
                "industry": pos["industry"],
                "perSectorWeight": pos["perSectorWeight"],
                "weightPct": int(intWeight),
                "dollarAllocation": dollarAllocation,
                "latestPrice": latestPrice,
                "rationale": pos["rationale"]
            }
            cleanedPositions.append(posRecord)

        sectorSubtotals[sectorName] = sectorIntTarget

    confirmedCashInt = int(round(confirmedCashPct))
    cashDollar = round(float(initialCapital) * (confirmedCashInt / 100.0), 2)
    cashPositionRecord = {
        "weightPct": confirmedCashInt,
        "dollarAllocation": cashDollar,
        "etfSubstitute": "SPY"
    }

    totalStockPct = sum(p["weightPct"] for p in cleanedPositions)
    totalAllocated = totalStockPct + confirmedCashInt

    portfolioRecord = {
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "initialCapital": float(initialCapital),
        "totalAllocatedPct": totalAllocated,
        "stockCount": len(cleanedPositions),
        "positions": cleanedPositions,
        "cashPosition": cashPositionRecord,
        "sectorBreakdown": sectorSubtotals,
        "portfolioRationale": portfolioRationaleStr,
        "rawSectorAllocations": sectorAllocations
    }

    tool.toolLog.append(portfolioRecord)
    return cleanData({
        "status": "success",
        "message": f"Portfolio creation confirmed: {len(cleanedPositions)} stocks across {len(expectedSectors)} sectors ({totalStockPct}%) and cash ({confirmedCashInt}%) totaling {totalAllocated}%.",
        "confirmedPortfolio": portfolioRecord
    })
