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

    if config is None:
        raise ValueError("config must be provided for executeBoardroomRating.")

    tickerToEval = config.ticker
    simulatedDateStr = config.simulatedDateStr
    boardroomPace = config.boardroomPace
    timeHorizon = config.timeHorizon

    if simulatedDateStr is None:
        simulatedDateStr = time.strftime("%Y-%m-%d", time.localtime())

    timestamp = pd.Timestamp(f"{simulatedDateStr} 09:00").tz_localize(NEW_YORK).tz_convert(UTC)

    if toolRegistry is None:
        from llm.server_manager import serverManager
        toolRegistry = serverManager.getToolRegistry()

    # Precache tool results for non macro tools into the lru cache
    precacheThread = startPrecacheThread(toolRegistry, timestamp, macroTools=False, ticker=tickerToEval)

    clientDuo = ClientDuo(boardroomClient, summaryClient)
    boardroom = generateBoardroom(toolRegistry, timestamp)
    boardroom.assignClientDuo(clientDuo)

    boardroom.execute(config=SingleEquityRatingConfig(
        ticker=tickerToEval,
        simulatedDateStr=simulatedDateStr,
        timeHorizon=timeHorizon,
        boardroomPace=boardroomPace,
        maxIterations=config.maxIterations,
        temperature=config.temperature,
        generateSummaries=config.generateSummaries
    ))

    endTime = time.time()
    elapsedSeconds = endTime - startTime
    mins = math.floor(elapsedSeconds / 60)
    secs = elapsedSeconds % 60
    emitEvent("simComplete", {"totalTime": f"{mins} mins {secs:.3f} secs"})
    
    return elapsedSeconds
