import time
import math
import threading
import traceback
import pandas as pd
from typing import Any, Optional, Tuple

from collectors.constants import UTC, NEW_YORK
from llm.llm_client import BaseLLMClient
from llm.client_duo import ClientDuo
from llm.server_manager import serverManager

from boardroom.boardroom_config import BoardroomConfig, SingleEquityRatingConfig
from boardroom.boardroom_engine import BoardroomEngine
from boardroom.boardroom_gen import generateBoardroom

from llmtools.lru_cacher import startPrecacheThread
from llmtools.tool_registry import ToolRegistry

from ui.ui_hooks import emitEvent, requestStop, resetStop, isStopRequested, SimulationStoppedException


class BoardroomManager:
    def __init__(self):
        self.activeBoardroom: Optional[BoardroomEngine] = None
        self.lastConfig: Optional[BoardroomConfig] = None
        self.toolRegistry: Optional[ToolRegistry] = None
        self.boardroomClient: Optional[BaseLLMClient] = None
        self.summaryClient: Optional[BaseLLMClient] = None
        self.activeThread: Optional[threading.Thread] = None
        self.isBoardroomRunning: bool = False
        self.stateLock = threading.Lock()

    def getToolRegistry(self) -> ToolRegistry:
        with self.stateLock:
            if self.toolRegistry is None:
                self.toolRegistry = serverManager.getToolRegistry()
            return self.toolRegistry

    def getClients(self) -> Tuple[Optional[BaseLLMClient], Optional[BaseLLMClient]]:
        with self.stateLock:
            if self.boardroomClient is not None:
                if self.summaryClient is not None:
                    return self.boardroomClient, self.summaryClient
                return self.boardroomClient, self.boardroomClient
            boardroomClient, summaryClient = serverManager.getClients()
            return boardroomClient, summaryClient

    def isBoardroomActive(self) -> bool:
        with self.stateLock:
            return bool(
                self.isBoardroomRunning and
                self.activeThread is not None and
                self.activeThread.is_alive()
            )

    def setupNewBoardroom(self, config: SingleEquityRatingConfig) -> BoardroomEngine:
        if config.simulatedDateStr is None:
            config.simulatedDateStr = time.strftime("%Y-%m-%d", time.localtime())
        timestamp = pd.Timestamp(f"{config.simulatedDateStr} 09:00").tz_localize(NEW_YORK).tz_convert(UTC)

        toolRegistry = self.getToolRegistry()
        boardroomClient, summaryClient = self.getClients()
        if not boardroomClient:
            raise RuntimeError("No active LLM server found. Please start a server from the manager setup page.")

        startPrecacheThread(toolRegistry, timestamp, macroTools=False, ticker=config.ticker)

        clientDuo = ClientDuo(boardroomClient, summaryClient)
        boardroom = generateBoardroom(toolRegistry, timestamp)
        boardroom.assignClientDuo(clientDuo)
        boardroom.lastConfig = config

        with self.stateLock:
            self.activeBoardroom = boardroom
            self.lastConfig = config

        return boardroom

    def runBoardroomTask(self, config: SingleEquityRatingConfig):
        startTime = time.time()
        try:
            boardroom = self.setupNewBoardroom(config)
            boardroom.execute(config=config)

            endTime = time.time()
            elapsedSeconds = endTime - startTime
            mins = math.floor(elapsedSeconds / 60)
            secs = elapsedSeconds % 60
            emitEvent("simComplete", {"totalTime": f"{mins} mins {secs:.3f} secs"})

        except SimulationStoppedException:
            print("Boardroom evaluation stopped by user.")
            emitEvent("simStopped", {"message": "Simulation stopped by user."})
        except Exception as e:
            if isStopRequested():
                print("Boardroom evaluation stopped by user.")
                emitEvent("simStopped", {"message": "Simulation stopped by user."})
            else:
                traceback.print_exc()
                emitEvent("error", {"message": f"{e.__class__.__name__}: {str(e)}"})
        finally:
            with self.stateLock:
                self.isBoardroomRunning = False
                self.activeThread = None
            resetStop()

    def startBoardroom(self, config: SingleEquityRatingConfig) -> Tuple[bool, str]:
        if self.isBoardroomActive():
            return False, "A boardroom simulation is already running."

        resetStop()
        with self.stateLock:
            self.isBoardroomRunning = True
            simThread = threading.Thread(
                target=self.runBoardroomTask,
                args=(config,),
                daemon=True
            )
            self.activeThread = simThread
            simThread.start()

        return True, "Boardroom simulation started."

    def stopBoardroom(self, timeout: float = 5.0) -> bool:
        requestStop()
        with self.stateLock:
            thread = self.activeThread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return not self.isBoardroomActive()

    def processQnAQuery(self, query: str) -> None:
        if self.activeBoardroom:
            self.activeBoardroom.processQnAQuery(query)

    def deleteQnATurn(self, turnIndex: int) -> bool:
        with self.stateLock:
            if self.activeBoardroom:
                return self.activeBoardroom.deleteQnATurn(turnIndex)
            return False

    def getLastBoardroom(self) -> Optional[BoardroomEngine]:
        return self.activeBoardroom

    def getLastConfig(self) -> Optional[BoardroomConfig]:
        return self.lastConfig


boardroomManager = BoardroomManager()


def executeBoardroomConfig(
    config: BoardroomConfig,
    boardroomClient: BaseLLMClient,
    summaryClient: Optional[BaseLLMClient] = None,
    toolRegistry: ToolRegistry = None
) -> float:
    if isinstance(config, SingleEquityRatingConfig):
        boardroomManager.boardroomClient = boardroomClient
        boardroomManager.summaryClient = summaryClient
        if toolRegistry is not None:
            boardroomManager.toolRegistry = toolRegistry
        startTime = time.time()
        boardroom = boardroomManager.setupNewBoardroom(config)
        boardroom.execute(config=config)

        elapsedSeconds = time.time() - startTime
        mins = math.floor(elapsedSeconds / 60)
        secs = elapsedSeconds % 60
        emitEvent("simComplete", {"totalTime": f"{mins} mins {secs:.3f} secs"})
        return elapsedSeconds
    else:
        raise NotImplementedError(f"BoardroomConfig type '{type(config).__name__}' is not supported yet.")
