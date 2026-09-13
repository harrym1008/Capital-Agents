from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from cli.ansi import ANSI
from llm.llm_client import BaseLLMClient
from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import BoardroomConfig, BoardroomPace
from ui.ui_hooks import getCurrentStage, isStopRequested, setCurrentStage, emitEvent, SimulationStoppedException


class BoardroomModelLoopException(Exception):
    pass


class BoardroomEngine(ABC):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        self.llmClient: Optional[BaseLLMClient] = None
        self.allowParallel: bool = False
        self.timestamp: pd.Timestamp = timestamp
        self.toolRegistry: ToolRegistry = toolRegistry
        self.toolRegistry.clearToolLogs()
        self.agents: Dict[str, FinancialAgent] = {}
        self.agentsList: List[FinancialAgent] = []
        self.lastConfig: Optional[BoardroomConfig] = None
        self.generate()

    @abstractmethod
    def generate(self) -> None:
        """Populates the engine's agents and tool mappings."""
        pass

    @abstractmethod
    def execute(self, config: BoardroomConfig) -> None:
        """Executes the boardroom deliberative phases for the given configuration."""
        pass

    def assignClient(self, llmClient: BaseLLMClient):
        self.llmClient = llmClient
        for agent in self.agentsList:
            agent.setClient(llmClient)
        self.allowParallel = llmClient.allowParallel

    def _runAgentsConcurrently(self, *tasks):
        if not self.allowParallel:
            return [task() for task in tasks]
        
        currentStageNum = getCurrentStage()
        
        def wrappedTask(task):
            setCurrentStage(currentStageNum)
            return task()

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(wrappedTask, task) for task in tasks]
            results = []
            try:
                for future in futures:
                    results.append(future.result())
            except SimulationStoppedException:
                for f in futures:
                    f.cancel()
                raise
            except Exception:
                if isStopRequested():
                    for f in futures:
                        f.cancel()
                    raise SimulationStoppedException("Simulation stopped by user.")
                raise
            return results

    def _getDefaultPhaseAgents(self, phaseNumber: int, pace: BoardroomPace) -> List[Dict[str, str]]:
        return []

    def _newPhaseHeader(self, phaseNumber: int, phaseName: str, pace: BoardroomPace = BoardroomPace.COMPLETE, customAgents: Optional[List[Dict[str, str]]] = None):
        setCurrentStage(phaseNumber)
        
        if phaseNumber == 0:
            tempHeader = f"{phaseName}"
        else:
            tempHeader = f"Phase {phaseNumber}: {phaseName}"
        tempHeader = f"{'|'*5} {tempHeader} {'|'*5}"
        headerLength = len(tempHeader)
        print(f"\n{ANSI.BOLD}{'-'*headerLength}\n{tempHeader}\n{'-'*headerLength}{ANSI.RESET}\n")

        if customAgents is not None:
            emitEvent("stageStart", {
                "stageNum": phaseNumber,
                "stageName": phaseName,
                "agents": customAgents
            })
            return

        agents = self._getDefaultPhaseAgents(phaseNumber, pace)
        emitEvent("stageStart", {
            "stageNum": phaseNumber,
            "stageName": phaseName,
            "agents": agents
        })

    def executeMandatedToolStage(
        self,
        agent: FinancialAgent,
        initialPrompt: str,
        mandatedToolName: str,
        config: BoardroomConfig,
        subrole: Optional[str] = None,
        maxRetries: int = 8,
        summarisationOverride: Optional[bool] = None,
        requireInitialTools: bool = True
    ) -> Tuple[str, str]:
        tool = self.toolRegistry.getTool(mandatedToolName)
        if tool:
            tool.toolLog.clear()

        currentPrompt = initialPrompt
        rawAnalysis = ""
        uiSummary = ""

        for attempt in range(maxRetries + 1):
            if isStopRequested():
                raise SimulationStoppedException("Simulation stopped by user.")

            isLastAttempt = (attempt == maxRetries)
            attemptSummaryOverride = False if not isLastAttempt else summarisationOverride

            rawAnalysis, uiSummary = agent.analyseAndReply(
                incomingMessage=currentPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                subrole=subrole,
                requireInitialTools=requireInitialTools or (attempt > 0),
                summarisationOverride=attemptSummaryOverride
            )

            if tool and len(tool.toolLog) > 0:
                # Notify agent that decision was confirmed and request short justifications & remarks
                confirmationPrompt = (
                    f"Your '{mandatedToolName}' submission has been verified, confirmed, and logged in the boardroom system.\n"
                    f"Please now provide short justifications and executive remarks around your decision-making process, "
                    f"key trade-offs considered, and final outcome for the boardroom."
                )

                remarksRaw, _ = agent.analyseAndReply(
                    incomingMessage=confirmationPrompt,
                    toolRegistry=self.toolRegistry,
                    timestamp=self.timestamp,
                    config=config,
                    subrole=subrole,
                    requireInitialTools=False,
                    summarisationOverride=False
                )

                combinedRaw = f"{rawAnalysis}\n\n{remarksRaw}".strip()
                finalUISummary = uiSummary if uiSummary else combinedRaw

                return combinedRaw, finalUISummary

            if not isLastAttempt:
                attempted = any(
                    isinstance(msg.get("tool_calls"), list) and
                    any(tc.get("function", {}).get("name") == mandatedToolName for tc in msg["tool_calls"])
                    for msg in agent.messageHistory[-6:] if isinstance(msg, dict) and msg.get("role") == "assistant"
                )
                if attempted:
                    currentPrompt = (
                        f"The '{mandatedToolName}' tool call you submitted was invalid or resulted in an error. "
                        f"You must provide a valid '{mandatedToolName}' tool call with corrected parameters to complete this boardroom stage."
                    )
                else:
                    currentPrompt = (
                        f"You did not execute the mandatory '{mandatedToolName}' tool call. "
                        f"You must upload your decision by executing the '{mandatedToolName}' tool with all required parameters."
                    )
                print(f"\n{ANSI.YELLOW}[Boardroom] Mandatory tool '{mandatedToolName}' not submitted or invalid. Retrying ({attempt + 1}/{maxRetries})...{ANSI.RESET}")

        errorMsg = (
            f"The model got stuck in a loop and failed to submit a valid '{mandatedToolName}' tool call "
            f"after {maxRetries} retries. Please try a larger parameter model or adjust generation settings."
        )
        print(f"\n{ANSI.RED}{ANSI.BOLD}[Boardroom Error] {errorMsg}{ANSI.RESET}\n")
        raise BoardroomModelLoopException(errorMsg)


def __getattr__(name: str):
    if name == "SingleEquityBoardroomEngine":
        from boardroom.engines.single_equity_rating_engine import SingleEquityBoardroomEngine
        return SingleEquityBoardroomEngine
    elif name == "PortfolioCreationBoardroomEngine":
        from boardroom.engines.portfolio_creation_engine import PortfolioCreationBoardroomEngine
        return PortfolioCreationBoardroomEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BoardroomEngine",
    "BoardroomModelLoopException",
    "SingleEquityBoardroomEngine",
    "PortfolioCreationBoardroomEngine"
]
