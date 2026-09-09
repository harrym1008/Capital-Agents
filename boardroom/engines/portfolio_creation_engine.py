import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from cli.ansi import ANSI
from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import PortfolioCreationConfig, BoardroomConfig, BoardroomPace
from boardroom.boardroom_engine import BoardroomEngine
from ui.ui_hooks import setCurrentStage, setCurrentAgent, setAgentPhase, emitEvent, SimulationStoppedException


class PortfolioCreationBoardroomEngine(BoardroomEngine):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        super().__init__(toolRegistry, timestamp)
        self.confirmedSectorAllocation: Optional[Dict[str, Any]] = None
        self.lastConfig: Optional[PortfolioCreationConfig] = None

    def generate(self) -> None:
        timestampStr = self.timestamp.strftime("%Y-%m-%d")
        toolMap = self.toolRegistry.getToolMap()

        self.macroAnalyst = FinancialAgent(
            agentRole="Macro Analyst",
            tools=[
                toolMap["fetchMacroContext"],
                toolMap["fetchMacroNews"],
                toolMap["fetchMacroSentimentHistory"],
                toolMap["fetchAllSectorRankings"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.CYAN,
            dateStr=timestampStr
        )

        self.bullAnalyst = FinancialAgent(
            agentRole="Bullish Value Analyst",
            tools=[
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["fetchAllSectorRankings"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.GREEN,
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["fetchAllSectorRankings"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.RED,
            dateStr=timestampStr
        )

        self.portManager = FinancialAgent(
            agentRole="Impartial Portfolio Manager",
            tools=[
                toolMap["fetchAllSectorRankings"],
                toolMap["fetchSectorPerformance"],
                toolMap["confirmSectorAllocation"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.MAGENTA,
            dateStr=timestampStr
        )

        self.agents = {
            "macroAnalyst": self.macroAnalyst,
            "bullAnalyst": self.bullAnalyst,
            "bearAnalyst": self.bearAnalyst,
            "portManager": self.portManager
        }
        self.agentsList = list(self.agents.values())

        if self.llmClient is not None:
            for agent in self.agentsList:
                agent.setClient(self.llmClient)

    def executePortfolioCreation(self, config: PortfolioCreationConfig):
        if self.llmClient is None:
            raise ValueError("LLM client is not assigned. Please assign a client before executing the boardroom.")

        self.llmClient.newTask()
        self.confirmedSectorAllocation = None

        startTime = datetime.now()
        dateStr = config.simulatedDateStr if config.simulatedDateStr else self.timestamp.strftime("%Y-%m-%d")
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()

        print(f"\n{'='*70}\nStarting Portfolio Creation Boardroom (Pace: {pace.value.upper()})\nCapital: {promptArgs['initialCapital']} | Sector Constraint: {promptArgs['sectorDiversityRule']} (Max: {promptArgs['maxSectorAllocation']})\n{'='*70}")

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        phase1Agents = [{"role": "Macro Analyst", "color": self.macroAnalyst.color, "name": "Macro Strategist"}]
        self._newPhaseHeader(1, "Macro Environment Analysis", pace, customAgents=phase1Agents)

        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and examine macro sector rotations to guide portfolio asset creation.\n"
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Time Horizon: {promptArgs['timeHorizon']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime and leading/lagging sectors.\n"
            f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
            f"3. Provide broad sector allocation guidance to prepare the boardroom for sector allocation decisions."
        )

        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        bullishSectorRaw, bullishSectorSummary = None, None
        bearishSectorRaw, bearishSectorSummary = None, None

        if pace == BoardroomPace.COMPLETE:
            # Phase 2: Sector Allocation Analysis (Bullish & Bearish in parallel)
            phase2Agents = [
                {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
            ]
            self._newPhaseHeader(2, "Sector Allocation Analysis", pace, customAgents=phase2Agents)

            bullPrompt = (
                f"Macro Analysis Context:\n{macroRaw}\n\n"
                f"Task: Propose a growth and cyclical sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
                f"Constraints: Max single sector allocation is {promptArgs['maxSectorAllocation']}. {promptArgs['sectorDiversityRule']}\n"
                f"1. Use 'fetchSectorPerformance' to evaluate high-conviction growth and cyclical sectors.\n"
                f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
                f"3. Highlight growth catalysts and upside drivers."
            )

            bearPrompt = (
                f"Macro Analysis Context:\n{macroRaw}\n\n"
                f"Task: Propose a defensive, risk-managed sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
                f"Constraints: Max single sector allocation is {promptArgs['maxSectorAllocation']}. {promptArgs['sectorDiversityRule']}\n"
                f"1. Use 'fetchSectorPerformance' to evaluate defensive, non-cyclical, and capital-preserving sectors.\n"
                f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
                f"3. Highlight vulnerabilities, drawdown risks, and defensive hedges."
            )

            def runBullSector():
                return self.bullAnalyst.analyseAndReply(
                    incomingMessage=bullPrompt,
                    toolRegistry=self.toolRegistry,
                    timestamp=self.timestamp,
                    config=config,
                    requireInitialTools=True
                )

            def runBearSector():
                return self.bearAnalyst.analyseAndReply(
                    incomingMessage=bearPrompt,
                    toolRegistry=self.toolRegistry,
                    timestamp=self.timestamp,
                    config=config,
                    requireInitialTools=True
                )

            results = self._runAgentsConcurrently(runBullSector, runBearSector)
            (bullishSectorRaw, bullishSectorSummary) = results[0]
            (bearishSectorRaw, bearishSectorSummary) = results[1]

        # Phase 3: Sector Allocation Decision (Impartial Portfolio Manager)
        phase3Agents = [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
        self._newPhaseHeader(3, "Sector Allocation Decision", pace, customAgents=phase3Agents)

        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")
        origPmTools = list(self.portManager.tools)
        if confirmSectorTool:
            confirmSectorTool.toolLog.clear()
            if confirmSectorTool not in self.portManager.tools:
                self.portManager.tools.append(confirmSectorTool)

        if pace == BoardroomPace.COMPLETE:
            pmSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Bullish Sector Proposal:\n{bullishSectorRaw}\n\n"
                f"Bearish Sector Proposal:\n{bearishSectorRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, reconcile the Bullish and Bearish proposals to make the definitive sector allocation decision for {promptArgs['initialCapital']}.\n"
                f"Mandatory Constraints:\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
                f"- Allocations must sum to 100.0%.\n\n"
                f"Execute the 'confirmSectorAllocation' tool with your exact sector dictionary and rationale."
            )
        else:
            # Fast pace: PM decides directly from macro context
            pmSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, determine the executive sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
                f"Mandatory Constraints:\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
                f"- Allocations must sum to 100.0%.\n\n"
                f"Execute the 'confirmSectorAllocation' tool with your exact sector dictionary and rationale."
            )

        pmSectorRaw, pmSectorSummary = self.portManager.analyseAndReply(
            incomingMessage=pmSectorPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        if confirmSectorTool:
            self.portManager.tools = origPmTools
            if confirmSectorTool.toolLog:
                self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        # Log completion of Phase 3
        print(f"\n[Phase 3 Complete] Confirmed Sector Allocation: {self.confirmedSectorAllocation}")
        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        self.lastConfig = config


    def execute(self, config: PortfolioCreationConfig) -> None:
        self.executePortfolioCreation(config)
        
