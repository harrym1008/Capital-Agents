from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


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
        if self.allocationBias is None:
            return "None (Balanced Strategy)"
        bias = int(self.allocationBias)
        labels = {
            1: "Maximum Growth Bias",
            2: "Moderate Growth Bias",
            3: "Minor Growth Bias",
            4: "Minor Defensive Bias",
            5: "Moderate Defensive Bias",
            6: "Maximum Defensive Bias"
        }
        return labels.get(bias, "Balanced Strategy")

    def getAllocationBiasGuidance(self) -> str:
        if self.allocationBias is None:
            return "Balanced portfolio strategy with impartial equilibrium between growth upside and defensive capital preservation."
        bias = int(self.allocationBias)
        if bias == 1:
            return (
                "MANDATORY MAXIMUM GROWTH BIAS DIRECTIVE: Heavily skew all sector distributions and stock selections towards "
                "high-beta, high-momentum, innovative, and rapid capital appreciation equities. Strongly minimize cash reserves "
                "and defensive weightings in pursuit of maximum upside growth potential."
            )
        elif bias == 2:
            return (
                "MODERATE GROWTH BIAS DIRECTIVE: Lean towards growth-oriented, cyclical, and innovative leaders with strong "
                "revenue expansion catalysts while maintaining sensible baseline balance sheet risk controls."
            )
        elif bias == 3:
            return (
                "MINOR GROWTH BIAS DIRECTIVE: Tilt slightly towards growth and cyclical opportunities while keeping a well-diversified "
                "foundation across core defensive and stable opportunities."
            )
        elif bias == 4:
            return (
                "MINOR DEFENSIVE BIAS DIRECTIVE: Tilt slightly towards capital preservation, lower-volatility companies, and dividend "
                "stability while maintaining modest participation in broad market growth."
            )
        elif bias == 5:
            return (
                "MODERATE DEFENSIVE BIAS DIRECTIVE: Lean towards defensive, dividend-paying, capital-preserving, and lower-volatility "
                "allocations to protect against downside market drawdowns."
            )
        elif bias == 6:
            return (
                "MANDATORY MAXIMUM DEFENSIVE BIAS DIRECTIVE: Heavily skew all sector distributions and stock selections towards "
                "maximum capital preservation, rock-solid balance sheet solvency, high dividend yield, low-beta defensive assets, "
                "and tactical cash reserves. Strictly minimize speculative, high-multiple, or volatile high-beta equities."
            )
        else:
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
