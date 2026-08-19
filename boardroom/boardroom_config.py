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

    @abstractmethod
    def getPromptArgs(self) -> Dict[str, str]:
        pass


@dataclass(kw_only=True)
class SingleEquityRatingConfig(BoardroomConfig):
    ticker: str
    simulatedDateStr: Optional[str]
    timeHorizon: TimeHorizon
    boardroomPace: BoardroomPace

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

    def unpack(self) -> tuple[str, Optional[str], TimeHorizon, BoardroomPace, int, float, bool, int]:
        return self.ticker, self.simulatedDateStr, self.timeHorizon, self.boardroomPace, self.maxIterations, self.temperature, self.generateSummaries, self.thinkingBudget
