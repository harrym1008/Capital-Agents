import time
import math
import pandas as pd
from typing import Any

from collectors.constants import UTC, NEW_YORK
from llm.llm_client import BaseLLMClient
from llm.client_duo import ClientDuo
from llm.boardroom_engine import generateBoardroom
from llm.tools.registry_builder import buildToolRegistry
from llm.tools.lru_cacher import startPrecacheThread
from ui.ui_hooks import emitEvent


def executeBoardroomRating(
    boardroomClient: BaseLLMClient,
    summaryClient: BaseLLMClient = None,
    tickerToEval: str = "NVDA",
    simulatedDate: str = None,
    fastMode: bool = True,
    toolRegistry: Any = None
) -> float:
    startTime = time.time()

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

    # precacheThread.join()

    boardroom.executeSingleEquityRating(targetTicker=tickerToEval, fastMode=fastMode)

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
