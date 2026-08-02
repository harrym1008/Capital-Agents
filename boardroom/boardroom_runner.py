import time
import math
import pandas as pd
from typing import Any, Optional

from collectors.constants import UTC, NEW_YORK
from llm.llm_client import BaseLLMClient
from llm.client_duo import ClientDuo
from boardroom.boardroom_config import BoardroomConfig, SingleEquityRatingConfig, TimeHorizon
from boardroom.boardroom_engine import generateBoardroom
from tools.registry_builder import buildToolRegistry
from tools.lru_cacher import startPrecacheThread
from ui.ui_hooks import emitEvent


def executeBoardroomConfig(
    config: BoardroomConfig,
    boardroomClient: BaseLLMClient,
    summaryClient: Optional[BaseLLMClient] = None,
    toolRegistry: Any = None
) -> float:
    if isinstance(config, SingleEquityRatingConfig):
        return executeBoardroomRating(
            boardroomClient=boardroomClient,
            summaryClient=summaryClient,
            tickerToEval=config.ticker,
            simulatedDate=config.simulatedDateStr,
            fastMode=config.fastMode,
            timeHorizon=config.timeHorizon,
            toolRegistry=toolRegistry,
            config=config
        )
    else:
        raise NotImplementedError(f"BoardroomConfig type '{type(config).__name__}' is not supported yet.")


def executeBoardroomRating(
    boardroomClient: BaseLLMClient,
    summaryClient: Optional[BaseLLMClient] = None,
    tickerToEval: str = "NVDA",
    simulatedDate: Optional[str] = None,
    fastMode: bool = True,
    timeHorizon: TimeHorizon = TimeHorizon.LONG,
    toolRegistry: Any = None,
    config: Optional[SingleEquityRatingConfig] = None
) -> float:
    startTime = time.time()

    if config is not None:
        tickerToEval = config.ticker
        simulatedDate = config.simulatedDateStr
        fastMode = config.fastMode
        timeHorizon = config.timeHorizon
    elif not isinstance(timeHorizon, TimeHorizon):
        try:
            timeHorizon = TimeHorizon(str(timeHorizon).lower())
        except ValueError:
            timeHorizon = TimeHorizon.LONG

    if simulatedDate is None:
        simulatedDate = time.strftime("%Y-%m-%d", time.localtime())

    timestamp = pd.Timestamp(f"{simulatedDate} 09:00").tz_localize(NEW_YORK).tz_convert(UTC)

    if toolRegistry is None:
        from llm.server_manager import serverManager
        toolRegistry = serverManager.getToolRegistry()

    precacheThread = startPrecacheThread(toolRegistry, timestamp, tickerToEval, includeMacro=True)

    clientDuo = ClientDuo(boardroomClient, summaryClient)
    boardroom = generateBoardroom(toolRegistry, timestamp)
    boardroom.assignClientDuo(clientDuo)

    boardroom.execute(config=SingleEquityRatingConfig(
        ticker=tickerToEval,
        simulatedDateStr=simulatedDate,
        timeHorizon=timeHorizon,
        fastMode=fastMode
    ))

    endTime = time.time()
    elapsedSeconds = endTime - startTime
    mins = math.floor(elapsedSeconds / 60)
    secs = elapsedSeconds % 60
    totalTimeStr = f"{mins} mins {secs:.3f} secs"

    print(f"\nTotal time taken for boardroom evaluation: {totalTimeStr}")
    try:
        emitEvent("simComplete", {"totalTime": totalTimeStr})
    except Exception:
        pass
    
    return elapsedSeconds
