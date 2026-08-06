import time
import math
import pandas as pd
from typing import Any, Optional

from collectors.constants import UTC, NEW_YORK
from llm.llm_client import BaseLLMClient
from llm.client_duo import ClientDuo

from boardroom.boardroom_config import BoardroomConfig, BoardroomPace, SingleEquityRatingConfig, TimeHorizon
from boardroom.boardroom_engine import generateBoardroom

from llmtools.lru_cacher import startPrecacheThread
from llmtools.tool_registry import ToolRegistry

from ui.ui_hooks import emitEvent


def executeBoardroomConfig(
    config: BoardroomConfig,
    boardroomClient: BaseLLMClient,
    summaryClient: Optional[BaseLLMClient] = None,
    toolRegistry: ToolRegistry = None
) -> float:
    if isinstance(config, SingleEquityRatingConfig):
        return executeBoardroomRating(
            boardroomClient=boardroomClient,
            summaryClient=summaryClient,
            tickerToEval=config.ticker,
            simulatedDateStr=config.simulatedDateStr,
            boardroomPace=config.boardroomPace,
            timeHorizon=config.timeHorizon,
            toolRegistry=toolRegistry,
            config=config
        )
    else:
        raise NotImplementedError(f"BoardroomConfig type '{type(config).__name__}' is not supported yet.")


def executeBoardroomRating(
    boardroomClient: BaseLLMClient,
    summaryClient: Optional[BaseLLMClient],
    tickerToEval: str,
    simulatedDateStr: Optional[str],
    boardroomPace: BoardroomPace,
    timeHorizon: TimeHorizon,
    toolRegistry: ToolRegistry,
    config: SingleEquityRatingConfig
):
    startTime = time.time()

    if config is not None:
        tickerToEval = config.ticker
        simulatedDateStr = config.simulatedDateStr
        boardroomPace = config.boardroomPace
        timeHorizon = config.timeHorizon
    else:
        if not isinstance(timeHorizon, TimeHorizon):
            try:
                timeHorizon = TimeHorizon(str(timeHorizon).lower())
            except ValueError:
                timeHorizon = TimeHorizon.LONG

        if not isinstance(boardroomPace, BoardroomPace):
            paceStr = str(boardroomPace).lower().replace(" ", "_").replace("-", "_")
            if paceStr == "oneshot":
                paceStr = "one_shot"
            try:
                boardroomPace = BoardroomPace(paceStr)
            except ValueError:
                boardroomPace = BoardroomPace.FAST

    if simulatedDateStr is None:
        simulatedDateStr = time.strftime("%Y-%m-%d", time.localtime())

    timestamp = pd.Timestamp(f"{simulatedDateStr} 09:00").tz_localize(NEW_YORK).tz_convert(UTC)

    if toolRegistry is None:
        from llm.server_manager import serverManager
        toolRegistry = serverManager.getToolRegistry()

    precacheThread = startPrecacheThread(toolRegistry, timestamp, macroTools=False, ticker=tickerToEval)

    clientDuo = ClientDuo(boardroomClient, summaryClient)
    boardroom = generateBoardroom(toolRegistry, timestamp)
    boardroom.assignClientDuo(clientDuo)

    boardroom.execute(config=SingleEquityRatingConfig(
        ticker=tickerToEval,
        simulatedDateStr=simulatedDateStr,
        timeHorizon=timeHorizon,
        boardroomPace=boardroomPace
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
