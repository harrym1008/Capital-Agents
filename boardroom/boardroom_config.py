from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


# Base class for time horizons, subclasses for each boardroom mode
class TimeHorizon(str, Enum):
    @classmethod
    def fromString(cls, val: Any):
        if isinstance(val, cls):
            return val
        if not val:
            return next(iter(cls))
        valStr = str(val).strip().lower().replace("_", " ").replace("-", " ")
        for member in cls:
            if member.value.lower() == valStr or member.name.lower() == valStr:
                return member
        return next(iter(cls))


class SingleEquityTimeHorizon(TimeHorizon):
    IMMEDIATE = "immediate"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    DISTANT = "distant"

    @property
    def label(self) -> str:
        return TIME_HORIZON_INFO[self]["label"]


class PortfolioTimeHorizon(TimeHorizon):
    ONE_MONTH = "1 month"
    THREE_MONTHS = "3 months"
    SIX_MONTHS = "6 months"
    ONE_YEAR = "1 year"
    TWO_YEARS = "2 years"
    THREE_YEARS = "3 years"
    FIVE_YEARS = "5 years"
    TEN_YEARS = "10 years"
    TWENTY_YEARS = "20 years"

    @property
    def label(self) -> str:
        return self.value.title()


# Boardroom pace options for different boardroom modes
class BoardroomPace(Enum):
    ONE_SHOT = "one_shot"
    FAST = "fast"
    COMPLETE = "complete"
    PRESET = "preset"



TIME_HORIZON_INFO: Dict[SingleEquityTimeHorizon, Dict[str, Any]] = {
    SingleEquityTimeHorizon.IMMEDIATE: {
        "label": "Immediate-Term",
        "llmPriceTargets": "3-day and 2-week price targets",
        "llmFinalLinePriceTargets": "3-Day Target: $[PRICE], 2-Week Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionImmediateTerm",
        "targets": ["3d", "2w"],
    },
    SingleEquityTimeHorizon.SHORT: {
        "label": "Short-Term",
        "llmPriceTargets": "1-month and 3-month price targets",
        "llmFinalLinePriceTargets": "1-Month Target: $[PRICE], 3-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionShortTerm",
        "targets": ["1mo", "3mo"],
    },
    SingleEquityTimeHorizon.MEDIUM: {
        "label": "Medium-Term",
        "llmPriceTargets": "3-month and 12-month price targets",
        "llmFinalLinePriceTargets": "3-Month Target: $[PRICE], 12-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionMediumTerm",
        "targets": ["3mo", "12mo"],
    },
    SingleEquityTimeHorizon.LONG: {
        "label": "Long-Term",
        "llmPriceTargets": "12-month and 36-month price targets",
        "llmFinalLinePriceTargets": "12-Month Target: $[PRICE], 36-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionLongTerm",
        "targets": ["12mo", "36mo"],
    },
    SingleEquityTimeHorizon.DISTANT: {
        "label": "Distant-Term",
        "llmPriceTargets": "3-year and 10-year price targets",
        "llmFinalLinePriceTargets": "3-Year Target: $[PRICE], 10-Year Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionDistantTerm",
        "targets": ["3y", "10y"],
    }
}


# Allocation bias guidance for portfolio creation and rebalancing where 1 is max growth and 6 is max defensive (disable in UI for balanced)
ALLOCATION_BIAS_INFO: Dict[int, Dict[str, str]] = {
    1: {
        "label": "Maximum Growth Bias",
        "guidance": (
            "MANDATORY MAXIMUM GROWTH BIAS DIRECTIVE: Heavily skew all sector distributions and stock selections towards "
            "high-beta, high-momentum, innovative, and rapid capital appreciation equities. Strongly minimise "
            "defensive weightings in pursuit of maximum upside growth potential."
        ),
    },
    2: {
        "label": "Moderate Growth Bias",
        "guidance": (
            "MODERATE GROWTH BIAS DIRECTIVE: Lean towards growth-oriented, cyclical, and innovative leaders with strong "
            "revenue expansion catalysts while maintaining sensible baseline balance sheet risk controls."
        ),
    },
    3: {
        "label": "Minor Growth Bias",
        "guidance": (
            "MINOR GROWTH BIAS DIRECTIVE: Tilt slightly towards growth and cyclical opportunities while keeping a well-diversified "
            "foundation across core defensive and stable opportunities."
        ),
    },
    4: {
        "label": "Minor Defensive Bias",
        "guidance": (
            "MINOR DEFENSIVE BIAS DIRECTIVE: Tilt slightly towards capital preservation, lower-volatility companies, and dividend "
            "stability while maintaining modest participation in broad market growth."
        ),
    },
    5: {
        "label": "Moderate Defensive Bias",
        "guidance": (
            "MODERATE DEFENSIVE BIAS DIRECTIVE: Lean towards defensive, dividend-paying, capital-preserving, and lower-volatility "
            "allocations to protect against downside market drawdowns."
        ),
    },
    6: {
        "label": "Maximum Defensive Bias",
        "guidance": (
            "MANDATORY MAXIMUM DEFENSIVE BIAS DIRECTIVE: Heavily skew all sector distributions and stock selections towards "
            "maximum capital preservation, rock-solid balance sheet solvency, high dividend yield, and low-beta defensive assets. "
            "Strictly minimise speculative, high-multiple, or volatile high-beta equities."
        ),
    },
}

# Rebalance amount guidance for portfolio rebalancing where 1 is very light and 6 is maximum rebalance
REBALANCE_AMOUNT_INFO: Dict[int, Dict[str, str]] = {
    1: {
        "label": "Level 1: Very Light Rebalance",
        "guidance": (
            "MANDATORY VERY LIGHT REBALANCE MANDATE (Level 1/6): Strongly preserve the existing portfolio architecture. "
            "Keep almost all baseline holdings intact and retain their core positions. Only make very minor adjustments, "
            "subtle percentage re-weightings, or exit a holding only if facing catastrophic fundamental deterioration. "
            "Preserving existing holdings with minimal rebalancing is your top operational priority."
        ),
    },
    2: {
        "label": "Level 2: Mild Rebalance",
        "guidance": (
            "MANDATORY MILD REBALANCE MANDATE (Level 2/6): Preserve the core foundation of the baseline portfolio. "
            "Retain the vast majority of existing stock holdings, trimming slightly from lower-conviction "
            "or underperforming positions to fund modest, selective additions in favored sectors. Keep overall rebalancing low."
        ),
    },
    3: {
        "label": "Level 3: Moderate Rebalance",
        "guidance": (
            "MANDATORY MODERATE REBALANCE MANDATE (Level 3/6): Balanced rebalance strategy. Proactively realign sector weights "
            "and holdings to adapt to current macro conditions. Maintain high-conviction core baseline holdings (around 60-70% of capital) "
            "while actively trimming underperformers and introducing high-alpha new equities."
        ),
    },
    4: {
        "label": "Level 4: Substantial Rebalance",
        "guidance": (
            "MANDATORY SUBSTANTIAL REBALANCE MANDATE (Level 4/6): Meaningful rebalance and assertive capital reallocation. "
            "Do not hesitate to rotate 50%+ of the portfolio capital away from stagnant or headwind-facing baseline holdings. "
            "Retain only the absolute highest-conviction baseline stocks and actively redeploy capital into newly scouted opportunities."
        ),
    },
    5: {
        "label": "Level 5: Aggressive Rebalance",
        "guidance": (
            "MANDATORY AGGRESSIVE REBALANCE MANDATE (Level 5/6): High rebalance and extensive portfolio remodeling. "
            "Substantially reconstruct both sector allocations and constituent equities. Aggressively exit or trim any baseline "
            "holdings that do not display superior forward alpha potential, replacing up to 70-80%+ of baseline exposure."
        ),
    },
    6: {
        "label": "Level 6: Maximum Rebalance",
        "guidance": (
            "MANDATORY MAXIMUM REBALANCE MANDATE (Level 6/6): Full portfolio overhaul. Complete freedom to aggressively "
            "restructure the portfolio from the ground up with maximum rebalance scope. Prioritise optimal forward risk-adjusted return "
            "regardless of baseline inertia, replace all non-optimal holdings without hesitation."
        ),
    },
}


@dataclass(kw_only=True)
class BoardroomConfig(ABC):
    generateSummaries: bool = True
    maxIterations: int = 10
    temperature: float = 0.5
    thinkingBudget: int = 2048

    @property
    @abstractmethod
    def modeName(self) -> str:
        pass

    @abstractmethod
    def getPromptArgs(self) -> Dict[str, str]:
        pass


@dataclass(kw_only=True)
class SingleEquityRatingConfig(BoardroomConfig):
    ticker: str
    simulatedDateStr: Optional[str]
    timeHorizon: SingleEquityTimeHorizon = SingleEquityTimeHorizon.LONG
    boardroomPace: BoardroomPace = BoardroomPace.FAST

    @property
    def modeName(self) -> str:
        return "SingleEquityRating"

    def getTimeHorizonInfo(self) -> Dict[str, Any]:
        return TIME_HORIZON_INFO.get(self.timeHorizon, TIME_HORIZON_INFO[SingleEquityTimeHorizon.LONG])

    def getPromptArgs(self) -> Dict[str, Any]:
        info = self.getTimeHorizonInfo()
        return {
            "timeHorizon": self.timeHorizon.label,
            "ticker": self.ticker,
            "pacingMode": self.boardroomPace.value.title(),
            **info
        }

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "SingleEquityRatingConfig":
        ticker = data.get("ticker", "NVDA")
        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None

        timeHorizon = SingleEquityTimeHorizon.fromString(data.get("timeHorizon", "long"))

        boardroomPaceStr = data.get("boardroomPace", "fast")
        boardroomPace = BoardroomPace(boardroomPaceStr)

        maxIterations = int(data.get("maxIterations", 10))
        temperature = float(data.get("temperature", 0.5))
        generateSummaries = bool(data.get("generateSummaries", True))
        thinkingBudget = int(data.get("thinkingBudget", 2048))

        return cls(
            ticker=ticker,
            simulatedDateStr=simulatedDateStr,
            timeHorizon=timeHorizon,
            boardroomPace=boardroomPace,
            maxIterations=maxIterations,
            temperature=temperature,
            generateSummaries=generateSummaries,
            thinkingBudget=thinkingBudget
        )


@dataclass(kw_only=True)
class PortfolioCreationConfig(BoardroomConfig):
    initialCapital: float = 100_000.0
    timeHorizon: PortfolioTimeHorizon = PortfolioTimeHorizon.ONE_YEAR
    boardroomPace: BoardroomPace = BoardroomPace.COMPLETE
    simulatedDateStr: Optional[str] = None

    # Configurable portfolio constraints
    targetSectorCount: Optional[int] = None
    maxSectorAllocation: float = 40.0
    maxStockAllocation: float = 20.0
    targetStockCount: Optional[int] = None
    presetSectorAllocations: Optional[Dict[str, float]] = None
    allocationBias: Optional[int] = None

    @property
    def modeName(self) -> str:
        return "PortfolioCreation"

    def getAllocationBiasLabel(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["label"]
        return "None (Balanced Strategy)"

    def getAllocationBiasGuidance(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["guidance"]
        return "Balanced portfolio strategy with impartial equilibrium between growth upside and defensive capital preservation."

    def getPromptArgs(self) -> Dict[str, str]:
        if self.targetSectorCount is not None:
            sectorDiversityRule = f"Select around {self.targetSectorCount} distinct GICS sectors (approximately {self.targetSectorCount} sectors based on opportunity and balance)."
        else:
            sectorDiversityRule = "You have full discretion to select the optimal number of sectors (typically 2 to 6 based on market conditions)."

        if self.targetStockCount is not None:
            stockCountRule = f"around {self.targetStockCount} total stocks across the portfolio (aim for approximately {self.targetStockCount} stocks)."
        else:
            stockCountRule = "optimal discretion (typically 6 to 18 stocks balanced across confirmed sectors)"

        return {
            "initialCapital": f"${self.initialCapital:,.2f}",
            "timeHorizon": self.timeHorizon.label,
            "sectorDiversityRule": sectorDiversityRule,
            "maxSectorAllocation": f"{min(max(self.maxSectorAllocation, 20.0), 80.0):.1f}%",
            "maxStockAllocation": f"{min(max(self.maxStockAllocation, 10.0), 60.0):.1f}%",
            "targetStockCount": stockCountRule,
            "targetSectorCount": str(self.targetSectorCount) if self.targetSectorCount is not None else "Dynamic",
            "allocationBias": self.getAllocationBiasLabel(),
            "allocationBiasGuidance": self.getAllocationBiasGuidance()
        }

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "PortfolioCreationConfig":
        initialCapital = float(data.get("initialCapital", 100_000.0))
        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None

        timeHorizon = PortfolioTimeHorizon.fromString(data.get("timeHorizon", "1 year"))

        boardroomPaceStr = data.get("boardroomPace", "complete")
        if boardroomPaceStr not in ["complete", "fast", "preset"]:
            boardroomPaceStr = "complete"
        boardroomPace = BoardroomPace(boardroomPaceStr)

        targetSectorCountRaw = data.get("targetSectorCount")
        targetSectorCount = int(targetSectorCountRaw) if targetSectorCountRaw is not None and str(targetSectorCountRaw).strip() != "" else None
        if targetSectorCount is not None:
            targetSectorCount = max(1, min(11, targetSectorCount))

        maxSectorAllocation = float(data.get("maxSectorAllocation", 40.0))
        maxSectorAllocation = max(20.0, min(80.0, maxSectorAllocation))

        targetStockCountRaw = data.get("targetStockCount")
        targetStockCount = int(targetStockCountRaw) if targetStockCountRaw is not None and str(targetStockCountRaw).strip() != "" else None
        if targetStockCount is not None:
            targetStockCount = max(1, min(30, targetStockCount))

        maxStockAllocation = float(data.get("maxStockAllocation", 20.0))
        maxStockAllocation = max(10.0, min(60.0, maxStockAllocation))

        presetSectorAllocations = data.get("presetSectorAllocations")

        allocationBiasRaw = data.get("allocationBias")
        allocationBias = int(allocationBiasRaw) if allocationBiasRaw is not None and str(allocationBiasRaw).strip() != "" else None
        if allocationBias is not None:
            allocationBias = max(0, min(100, allocationBias))

        maxIterations = int(data.get("maxIterations", 10))
        temperature = float(data.get("temperature", 0.5))
        generateSummaries = bool(data.get("generateSummaries", True))
        thinkingBudget = int(data.get("thinkingBudget", 2048))

        return cls(
            initialCapital=initialCapital,
            timeHorizon=timeHorizon,
            boardroomPace=boardroomPace,
            simulatedDateStr=simulatedDateStr,
            targetSectorCount=targetSectorCount,
            maxSectorAllocation=maxSectorAllocation,
            maxStockAllocation=maxStockAllocation,
            targetStockCount=targetStockCount,
            presetSectorAllocations=presetSectorAllocations,
            allocationBias=allocationBias,
            maxIterations=maxIterations,
            temperature=temperature,
            generateSummaries=generateSummaries,
            thinkingBudget=thinkingBudget
        )


@dataclass(kw_only=True)
class PortfolioRebalancingConfig(BoardroomConfig):
    initialCapital: float = 100_000.0
    timeHorizon: PortfolioTimeHorizon = PortfolioTimeHorizon.ONE_YEAR
    boardroomPace: BoardroomPace = BoardroomPace.COMPLETE
    simulatedDateStr: Optional[str] = None

    # Baseline portfolio holdings (pure equity, no cash)
    currentPositions: Optional[list] = None
    precalculatedSectorAnalysis: Optional[Dict[str, Any]] = None

    # Configurable rebalancing constraints
    targetSectorCount: Optional[int] = None
    maxSectorAllocation: Optional[float] = None
    maxStockAllocation: Optional[float] = None
    targetStockCount: Optional[int] = None
    allocationBias: Optional[int] = None
    rebalanceAmount: int = 3

    @property
    def modeName(self) -> str:
        return "PortfolioRebalancing"

    def getRebalanceAmountLabel(self) -> str:
        amt = int(self.rebalanceAmount) if self.rebalanceAmount else 3
        return REBALANCE_AMOUNT_INFO.get(amt, REBALANCE_AMOUNT_INFO[3])["label"]

    def getRebalanceAmountGuidance(self) -> str:
        amt = int(self.rebalanceAmount) if self.rebalanceAmount else 3
        return REBALANCE_AMOUNT_INFO.get(amt, REBALANCE_AMOUNT_INFO[3])["guidance"]

    def getAllocationBiasLabel(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["label"]
        return "None (Balanced Strategy)"

    def getAllocationBiasGuidance(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["guidance"]
        return "Balanced portfolio strategy with impartial equilibrium between growth upside and defensive capital preservation."

    def formatCurrentHoldingsText(self) -> str:
        if not self.currentPositions:
            return "No existing holdings supplied."
        lines = []
        for pos in self.currentPositions:
            ticker = pos.get("ticker", "UNKNOWN")
            name = pos.get("companyName", ticker)
            sector = pos.get("sector", "Unknown Sector")
            dollarAmount = float(pos.get("dollarAmount", 0.0))
            weightPct = float(pos.get("weightPct", 0.0))
            lines.append(f"- {ticker} ({name}) | Sector: {sector} | Value: ${dollarAmount:,.2f} ({weightPct:.1f}%)")
        return "\n".join(lines)

    def formatPrecalculatedSectorsText(self) -> str:
        if not self.precalculatedSectorAnalysis:
            return "No precalculated sector breakdown available."
        secMap = self.precalculatedSectorAnalysis.get("sectors", {})
        if not secMap:
            return "No sector breakdown available."
        lines = []
        for secName, secInfo in secMap.items():
            pct = float(secInfo.get("weightPct", 0.0))
            dollars = float(secInfo.get("dollarAmount", 0.0))
            stockCount = int(secInfo.get("stockCount", 0))
            lines.append(f"- {secName}: {pct:.1f}% (${dollars:,.2f} across {stockCount} stock{'s' if stockCount != 1 else ''})")
        return "\n".join(lines)

    def getPromptArgs(self) -> Dict[str, str]:
        if self.targetSectorCount is not None:
            sectorDiversityRule = f"Aim for around {self.targetSectorCount} distinct GICS sectors in the rebalanced portfolio."
        else:
            sectorDiversityRule = "You have full discretion to retain, expand, or prune sectors based on macroeconomic and sector conditions."

        if self.targetStockCount is not None:
            stockCountRule = f"around {self.targetStockCount} total stocks across the rebalanced portfolio."
        else:
            stockCountRule = "optimal discretion (calibrated to manage risk and sector coverage effectively)"

        promptArgs = {
            "initialCapital": f"${self.initialCapital:,.2f}",
            "timeHorizon": self.timeHorizon.label,
            "currentHoldingsList": self.formatCurrentHoldingsText(),
            "precalculatedSectorsList": self.formatPrecalculatedSectorsText(),
            "sectorDiversityRule": sectorDiversityRule,
            "targetStockCount": stockCountRule,
            "targetSectorCount": str(self.targetSectorCount) if self.targetSectorCount is not None else "Dynamic",
            "allocationBias": self.getAllocationBiasLabel(),
            "allocationBiasGuidance": self.getAllocationBiasGuidance(),
            "rebalanceAmount": self.getRebalanceAmountLabel(),
            "rebalanceAmountGuidance": self.getRebalanceAmountGuidance()
        }
        if self.maxSectorAllocation is not None:
            promptArgs["maxSectorAllocation"] = f"{min(max(self.maxSectorAllocation, 20.0), 80.0):.1f}%"
        if self.maxStockAllocation is not None:
            promptArgs["maxStockAllocation"] = f"{min(max(self.maxStockAllocation, 10.0), 60.0):.1f}%"
        return promptArgs

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "PortfolioRebalancingConfig":
        positions = data.get("currentPositions") or []
        
        # Calculate initial capital strictly as sum of positions (pure equity, no cash)
        totalPosVal = sum(float(p.get("dollarAmount", 0.0) or 0.0) for p in positions)
        initialCapital = totalPosVal if totalPosVal > 0 else float(data.get("initialCapital", 100_000.0))

        # Re-compute exact weightPct for each position based on total
        cleanedPositions = []
        for p in positions:
            pDollar = float(p.get("dollarAmount", 0.0) or 0.0)
            pWeight = (pDollar / initialCapital * 100.0) if initialCapital > 0 else 0.0
            cleanedPositions.append({
                "ticker": str(p.get("ticker", "")).strip().upper(),
                "companyName": p.get("companyName") or p.get("name") or p.get("ticker", ""),
                "sector": p.get("sector") or "Unknown",
                "sectorTicker": p.get("sectorTicker") or "",
                "industry": p.get("industry") or "General Equities",
                "dollarAmount": round(pDollar, 2),
                "weightPct": round(pWeight, 2),
                "price": p.get("price")
            })

        # Precalculate sector breakdown
        sectorAgg = {}
        for p in cleanedPositions:
            sec = p.get("sector") or "Unknown"
            if sec not in sectorAgg:
                sectorAgg[sec] = {
                    "sector": sec,
                    "dollarAmount": 0.0,
                    "weightPct": 0.0,
                    "stockCount": 0,
                    "tickers": []
                }
            sectorAgg[sec]["dollarAmount"] += p["dollarAmount"]
            sectorAgg[sec]["stockCount"] += 1
            sectorAgg[sec]["tickers"].append(p["ticker"])

        for sec, sData in sectorAgg.items():
            sData["dollarAmount"] = round(sData["dollarAmount"], 2)
            sData["weightPct"] = round((sData["dollarAmount"] / initialCapital * 100.0) if initialCapital > 0 else 0.0, 2)

        precalculatedSectorAnalysis = {
            "totalCapital": initialCapital,
            "sectorCount": len(sectorAgg),
            "sectors": sectorAgg
        }

        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None
        timeHorizon = PortfolioTimeHorizon.fromString(data.get("timeHorizon", "1 year"))

        boardroomPaceStr = data.get("boardroomPace", "complete")
        if boardroomPaceStr not in ["complete", "fast"]:
            boardroomPaceStr = "complete"
        boardroomPace = BoardroomPace(boardroomPaceStr)

        targetSectorCountRaw = data.get("targetSectorCount")
        targetSectorCount = int(targetSectorCountRaw) if targetSectorCountRaw is not None and str(targetSectorCountRaw).strip() != "" else None
        if targetSectorCount is not None:
            targetSectorCount = max(1, min(11, targetSectorCount))

        maxSectorAllocationRaw = data.get("maxSectorAllocation")
        maxSectorAllocation = float(maxSectorAllocationRaw) if maxSectorAllocationRaw is not None and str(maxSectorAllocationRaw).strip() != "" else None
        if maxSectorAllocation is not None:
            maxSectorAllocation = max(20.0, min(80.0, maxSectorAllocation))

        targetStockCountRaw = data.get("targetStockCount")
        targetStockCount = int(targetStockCountRaw) if targetStockCountRaw is not None and str(targetStockCountRaw).strip() != "" else None
        if targetStockCount is not None:
            targetStockCount = max(1, min(30, targetStockCount))

        maxStockAllocationRaw = data.get("maxStockAllocation")
        maxStockAllocation = float(maxStockAllocationRaw) if maxStockAllocationRaw is not None and str(maxStockAllocationRaw).strip() != "" else None
        if maxStockAllocation is not None:
            maxStockAllocation = max(10.0, min(60.0, maxStockAllocation))

        allocationBiasRaw = data.get("allocationBias")
        allocationBias = int(allocationBiasRaw) if allocationBiasRaw is not None and str(allocationBiasRaw).strip() != "" else None
        if allocationBias is not None:
            allocationBias = max(0, min(100, allocationBias))

        rebalanceAmountRaw = data.get("rebalanceAmount")
        rebalanceAmount = int(rebalanceAmountRaw) if rebalanceAmountRaw is not None and str(rebalanceAmountRaw).strip() != "" else 3
        rebalanceAmount = max(1, min(6, rebalanceAmount))

        maxIterations = int(data.get("maxIterations", data.get("maxIters", 10)))
        temperature = float(data.get("temperature", 0.5))
        generateSummaries = bool(data.get("generateSummaries", True))
        thinkingBudget = int(data.get("thinkingBudget", 2048))

        return cls(
            initialCapital=initialCapital,
            timeHorizon=timeHorizon,
            boardroomPace=boardroomPace,
            simulatedDateStr=simulatedDateStr,
            currentPositions=cleanedPositions,
            precalculatedSectorAnalysis=precalculatedSectorAnalysis,
            targetSectorCount=targetSectorCount,
            maxSectorAllocation=maxSectorAllocation,
            maxStockAllocation=maxStockAllocation,
            targetStockCount=targetStockCount,
            allocationBias=allocationBias,
            rebalanceAmount=rebalanceAmount,
            maxIterations=maxIterations,
            temperature=temperature,
            generateSummaries=generateSummaries,
            thinkingBudget=thinkingBudget
        )


class SimulationTimestep(str, Enum):
    ONE_WEEK = "1 week"
    TWO_WEEKS = "2 weeks"
    ONE_MONTH = "1 month"
    TWO_MONTHS = "2 months"
    THREE_MONTHS = "3 months"

    @property
    def label(self) -> str:
        return self.value

    @classmethod
    def fromString(cls, val: Any) -> "SimulationTimestep":
        if isinstance(val, cls):
            return val
        try:
            return cls(str(val))
        except (ValueError, KeyError):
            return cls.ONE_MONTH


@dataclass(kw_only=True)
class AgentPortfolioSimulationConfig(BoardroomConfig):
    startDateStr: str = "2019-01-02"
    endDateStr: str = "2020-01-02"
    initialCapital: float = 100_000.0
    timestep: SimulationTimestep = SimulationTimestep.ONE_MONTH
    allocationBias: Optional[int] = None
    rebalanceAmount: int = 3
    boardroomPace: BoardroomPace = BoardroomPace.FAST

    targetSectorCount: Optional[int] = None
    targetStockCount: Optional[int] = None
    maxSectorAllocation: float = 40.0
    maxStockAllocation: float = 20.0

    @property
    def modeName(self) -> str:
        return "AgentPortfolioSimulation"

    def getRebalanceAmountLabel(self) -> str:
        amt = int(self.rebalanceAmount) if self.rebalanceAmount else 3
        return REBALANCE_AMOUNT_INFO.get(amt, REBALANCE_AMOUNT_INFO[3])["label"]

    def getRebalanceAmountGuidance(self) -> str:
        amt = int(self.rebalanceAmount) if self.rebalanceAmount else 3
        return REBALANCE_AMOUNT_INFO.get(amt, REBALANCE_AMOUNT_INFO[3])["guidance"]

    def getAllocationBiasLabel(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["label"]
        return "None (Balanced Strategy)"

    def getAllocationBiasGuidance(self) -> str:
        if self.allocationBias and self.allocationBias in ALLOCATION_BIAS_INFO:
            return ALLOCATION_BIAS_INFO[self.allocationBias]["guidance"]
        return "Balanced portfolio strategy with impartial equilibrium between growth upside and defensive capital preservation."

    def getPromptArgs(self) -> Dict[str, str]:
        if self.targetSectorCount is not None:
            sectorDiversityRule = f"Aim for around {self.targetSectorCount} distinct GICS sectors."
        else:
            sectorDiversityRule = "You have full discretion to select or adjust the optimal number of sectors (typically 2 to 6)."

        if self.targetStockCount is not None:
            stockCountRule = f"around {self.targetStockCount} total stocks across the portfolio."
        else:
            stockCountRule = "optimal discretion (typically 6 to 18 stocks balanced across confirmed sectors)"

        return {
            "initialCapital": f"${self.initialCapital:,.2f}",
            "startDate": self.startDateStr,
            "endDate": self.endDateStr,
            "timestep": self.timestep.label,
            "timeHorizon": self.timestep.label,
            "sectorDiversityRule": sectorDiversityRule,
            "targetStockCount": stockCountRule,
            "targetSectorCount": str(self.targetSectorCount) if self.targetSectorCount is not None else "Dynamic",
            "maxSectorAllocation": f"{min(max(self.maxSectorAllocation, 20.0), 80.0):.1f}%",
            "maxStockAllocation": f"{min(max(self.maxStockAllocation, 10.0), 60.0):.1f}%",
            "allocationBias": self.getAllocationBiasLabel(),
            "allocationBiasGuidance": self.getAllocationBiasGuidance(),
            "rebalanceAmount": self.getRebalanceAmountLabel(),
            "rebalanceAmountGuidance": self.getRebalanceAmountGuidance()
        }

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "AgentPortfolioSimulationConfig":
        startDateStr = str(data.get("startDate") or data.get("startDateStr") or "2019-01-02").strip()
        endDateStr = str(data.get("endDate") or data.get("endDateStr") or "2020-01-02").strip()
        initialCapital = float(data.get("initialCapital", 100_000.0))

        timestepRaw = data.get("timestep", "1 month")
        timestep = SimulationTimestep.fromString(timestepRaw)

        boardroomPaceStr = str(data.get("boardroomPace", "fast")).lower()
        if boardroomPaceStr not in ["complete", "fast"]:
            boardroomPaceStr = "fast"
        boardroomPace = BoardroomPace(boardroomPaceStr)

        targetSectorCountRaw = data.get("targetSectorCount") if data.get("targetSectorCount") is not None else data.get("targetSectors")
        targetSectorCount = int(targetSectorCountRaw) if targetSectorCountRaw is not None and str(targetSectorCountRaw).strip() != "" else None
        if targetSectorCount is not None:
            targetSectorCount = max(1, min(11, targetSectorCount))

        maxSectorAllocation = float(data.get("maxSectorAllocation", 40.0))
        maxSectorAllocation = max(20.0, min(80.0, maxSectorAllocation))

        targetStockCountRaw = data.get("targetStockCount") if data.get("targetStockCount") is not None else data.get("targetStocks")
        targetStockCount = int(targetStockCountRaw) if targetStockCountRaw is not None and str(targetStockCountRaw).strip() != "" else None
        if targetStockCount is not None:
            targetStockCount = max(1, min(50, targetStockCount))

        maxStockAllocation = float(data.get("maxStockAllocation", 20.0))
        maxStockAllocation = max(10.0, min(60.0, maxStockAllocation))

        allocationBiasRaw = data.get("allocationBias")
        allocationBias = int(allocationBiasRaw) if allocationBiasRaw is not None and str(allocationBiasRaw).strip() != "" else None
        if allocationBias is not None:
            allocationBias = max(1, min(6, allocationBias))

        rebalanceAmountRaw = data.get("rebalanceAmount")
        rebalanceAmount = int(rebalanceAmountRaw) if rebalanceAmountRaw is not None and str(rebalanceAmountRaw).strip() != "" else 3
        rebalanceAmount = max(1, min(6, rebalanceAmount))

        maxIterations = int(data.get("maxIterations", 10))
        temperature = float(data.get("temperature", 0.5))
        generateSummaries = bool(data.get("generateSummaries", True))
        thinkingBudget = int(data.get("thinkingBudget", 2048))

        return cls(
            startDateStr=startDateStr,
            endDateStr=endDateStr,
            initialCapital=initialCapital,
            timestep=timestep,
            allocationBias=allocationBias,
            rebalanceAmount=rebalanceAmount,
            boardroomPace=boardroomPace,
            targetSectorCount=targetSectorCount,
            targetStockCount=targetStockCount,
            maxSectorAllocation=maxSectorAllocation,
            maxStockAllocation=maxStockAllocation,
            maxIterations=maxIterations,
            temperature=temperature,
            generateSummaries=generateSummaries,
            thinkingBudget=thinkingBudget
        )


