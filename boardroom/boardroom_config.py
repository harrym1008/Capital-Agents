from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TimeHorizon(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    DISTANT = "distant"


class BoardroomType(Enum):
    SINGLE_EQUITY_RATING = "single_equity_rating"


class BoardroomPace(Enum):
    ONE_SHOT = "one_shot"
    FAST = "fast"
    COMPLETE = "complete"


TIME_HORIZON_INFO: Dict[TimeHorizon, Dict[str, str]] = {
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


@dataclass
class BoardroomConfig(ABC):
    pass


@dataclass
class SingleEquityRatingConfig(BoardroomConfig):
    ticker: str
    simulatedDateStr: Optional[str]
    timeHorizon: TimeHorizon
    boardroomPace: BoardroomPace

    def getTimeHorizonInfo(self) -> Dict[str, str]:
        return TIME_HORIZON_INFO.get(self.timeHorizon, TIME_HORIZON_INFO[TimeHorizon.LONG])

    @classmethod
    def fromDict(cls, data: Dict[str, Any]) -> "SingleEquityRatingConfig":
        ticker = data.get("ticker", "NVDA")
        simulatedDateStr = data.get("simulatedDate") or data.get("simulatedDateStr") or None

        horizonRaw = data.get("timeHorizon", "long")
        if isinstance(horizonRaw, TimeHorizon):
            timeHorizon = horizonRaw
        else:
            try:
                timeHorizon = TimeHorizon(str(horizonRaw).lower())
            except ValueError:
                timeHorizon = TimeHorizon.LONG

        boardroomPaceRaw = data.get("boardroomPace") or data.get("mode") or "fast"
        if isinstance(boardroomPaceRaw, BoardroomPace):
            boardroomPace = boardroomPaceRaw
        else:
            paceStr = str(boardroomPaceRaw).lower().replace(" ", "_").replace("-", "_")
            if paceStr == "oneshot":
                paceStr = "one_shot"
            try:
                boardroomPace = BoardroomPace(paceStr)
            except ValueError:
                boardroomPace = BoardroomPace.FAST

        return cls(
            ticker=ticker,
            simulatedDateStr=simulatedDateStr,
            timeHorizon=timeHorizon,
            boardroomPace=boardroomPace
        )

    def unpack(self) -> tuple[str, Optional[str], TimeHorizon, BoardroomPace]:
        return self.ticker, self.simulatedDateStr, self.timeHorizon, self.boardroomPace
