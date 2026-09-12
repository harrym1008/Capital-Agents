from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TimeHorizon(Enum):
    IMMEDIATE = "immediate"
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    DISTANT = "distant"


class BoardroomPace(Enum):
    ONE_SHOT = "one_shot"
    FAST = "fast"
    COMPLETE = "complete"
    PRESET = "preset"



TIME_HORIZON_INFO: Dict[TimeHorizon, Dict[str, str]] = {
    TimeHorizon.IMMEDIATE: {
        "label": "Immediate-Term",
        "llmPriceTargets": "3-day and 2-week price targets",
        "llmFinalLinePriceTargets": "3-Day Target: $[PRICE], 2-Week Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionImmediateTerm",
        "targets": ["3d", "2w"],
    },
    TimeHorizon.SHORT: {
        "label": "Short-Term",
        "llmPriceTargets": "1-month and 3-month price targets",
        "llmFinalLinePriceTargets": "1-Month Target: $[PRICE], 3-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionShortTerm",
        "targets": ["1mo", "3mo"],
    },
    TimeHorizon.MEDIUM: {
        "label": "Medium-Term",
        "llmPriceTargets": "3-month and 12-month price targets",
        "llmFinalLinePriceTargets": "3-Month Target: $[PRICE], 12-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionMediumTerm",
        "targets": ["3mo", "12mo"],
    },
    TimeHorizon.LONG: {
        "label": "Long-Term",
        "llmPriceTargets": "12-month and 36-month price targets",
        "llmFinalLinePriceTargets": "12-Month Target: $[PRICE], 36-Month Target: $[PRICE]",
        "llmSubmitToolName": "confirmBoardroomDecisionLongTerm",
        "targets": ["12mo", "36mo"],
    },
    TimeHorizon.DISTANT: {
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
    timeHorizon: TimeHorizon
    boardroomPace: BoardroomPace

    @property
    def modeName(self) -> str:
        return "SingleEquityRating"

    def getTimeHorizonInfo(self) -> Dict[str, str]:
        return TIME_HORIZON_INFO.get(self.timeHorizon, TIME_HORIZON_INFO[TimeHorizon.LONG])

    def getPromptArgs(self) -> Dict[str, str]:
        return self.getTimeHorizonInfo()

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "SingleEquityRatingConfig":
        ticker = data.get("ticker", "NVDA")
        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None

        horizonRaw = data.get("timeHorizon", "long")
        timeHorizon = TimeHorizon(horizonRaw)

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
    timeHorizon: TimeHorizon = TimeHorizon.LONG
    boardroomPace: BoardroomPace = BoardroomPace.COMPLETE
    simulatedDateStr: Optional[str] = None

    # Configurable portfolio constraints
    targetSectorCount: Optional[int] = None
    maxSectorAllocation: float = 40.0
    maxStockAllocation: float = 20.0
    targetStockCount: Optional[int] = None
    presetSectorAllocations: Optional[Dict[str, float]] = None

    @property
    def modeName(self) -> str:
        return "PortfolioCreation"

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
            "timeHorizon": self.timeHorizon.value,
            "sectorDiversityRule": sectorDiversityRule,
            "maxSectorAllocation": f"{min(max(self.maxSectorAllocation, 20.0), 80.0):.1f}%",
            "maxStockAllocation": f"{min(max(self.maxStockAllocation, 10.0), 60.0):.1f}%",
            "targetStockCount": stockCountRule,
            "targetSectorCount": str(self.targetSectorCount) if self.targetSectorCount is not None else "Dynamic"
        }

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "PortfolioCreationConfig":
        initialCapital = float(data.get("initialCapital", 100_000.0))
        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None

        horizonRaw = data.get("timeHorizon", "long")
        timeHorizon = TimeHorizon(horizonRaw) if isinstance(horizonRaw, str) else horizonRaw

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
            maxIterations=maxIterations,
            temperature=temperature,
            generateSummaries=generateSummaries,
            thinkingBudget=thinkingBudget
        )
