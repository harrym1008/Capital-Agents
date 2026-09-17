import os
import re
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
from collectors.sector_dl_client import GICS_SECTORS



class PortfolioCreationBoardroomEngine(BoardroomEngine):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        super().__init__(toolRegistry, timestamp)
        self.confirmedSectorAllocation: Optional[Dict[str, Any]] = None
        self.confirmedPortfolioAllocation: Optional[Dict[str, Any]] = None
        self.lastConfig: Optional[PortfolioCreationConfig] = None
        self.fullConvSummary: str = ""

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
                toolMap["fetchAllSectorsAnalysis"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.GREEN,
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.RED,
            dateStr=timestampStr
        )

        self.portManager = FinancialAgent(
            agentRole="Impartial Portfolio Manager",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                toolMap["confirmSectorAllocation"],
                toolMap["confirmPortfolioAllocation"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.MAGENTA,
            dateStr=timestampStr
        )

        self.growthHunter = FinancialAgent(
            agentRole="Growth Stock Hunter",
            tools=[
                toolMap["fetchStocksInSector"],
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.GREEN,
            dateStr=timestampStr
        )

        self.valueHunter = FinancialAgent(
            agentRole="Value/Defensive Stock Hunter",
            tools=[
                toolMap["fetchStocksInSector"],
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchCashFlowStatement"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.BLUE,
            dateStr=timestampStr
        )

        self.aggRiskAnalyst = FinancialAgent(
            agentRole="Aggressive Risk Analyst",
            tools=[
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.YELLOW,
            dateStr=timestampStr
        )

        self.consRiskAnalyst = FinancialAgent(
            agentRole="Conservative Risk Analyst",
            tools=[
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.BLUE,
            dateStr=timestampStr
        )

        self.agents = {
            "macroAnalyst": self.macroAnalyst,
            "bullAnalyst": self.bullAnalyst,
            "bearAnalyst": self.bearAnalyst,
            "portManager": self.portManager,
            "growthHunter": self.growthHunter,
            "valueHunter": self.valueHunter,
            "aggRiskAnalyst": self.aggRiskAnalyst,
            "consRiskAnalyst": self.consRiskAnalyst
        }
        self.agentsList = list(self.agents.values())

        if self.llmClient is not None:
            for agent in self.agentsList:
                agent.setClient(self.llmClient)

    def _getDefaultPhaseAgents(self, phaseNumber: int, pace: BoardroomPace) -> List[Dict[str, str]]:
        isFast = pace == BoardroomPace.FAST

        if phaseNumber == 1:
            return [{"role": "Macro Analyst", "color": self.macroAnalyst.colorName, "name": "Macro Strategist"}]
        elif phaseNumber == 2:
            return [
                {"role": "Bullish Value Analyst", "color": self.bullAnalyst.colorName, "name": "Bullish Analyst"},
                {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.colorName, "name": "Bearish Analyst"}
            ]
        elif phaseNumber == 3:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.colorName, "name": "Portfolio Manager"}]
        elif phaseNumber == 4:
            return [
                {"role": "Growth Stock Hunter", "color": self.growthHunter.colorName, "name": "Growth Stock Hunter"},
                {"role": "Value/Defensive Stock Hunter", "color": self.valueHunter.colorName, "name": "Defensive Stock Hunter"}
            ]
        elif phaseNumber == 5:
            return [
                {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.colorName, "name": "Aggressive Risk Analyst"},
                {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.colorName, "name": "Conservative Risk Analyst"}
            ]
        elif phaseNumber == 6:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.colorName, "name": "Portfolio Manager"}]

        return []

    def _formatSectorsContext(self) -> str:
        if not self.confirmedSectorAllocation:
            return "No confirmed sector allocation available."
        confirmed = self.confirmedSectorAllocation.get("confirmedAllocation", self.confirmedSectorAllocation)
        sectorsDict = confirmed.get("sectorAllocations", {})
        lines = [f"- {secInfo.get('sector', secKey)} ({secKey}): {int(round(float(secInfo.get('allocationPct', 0))))}%" for secKey, secInfo in sectorsDict.items()]
        return "\n".join(lines)


    def executeFastPortfolioCreation(self, config: PortfolioCreationConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Fast Portfolio Creation Boardroom (Pace: FAST)")

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and examine macro sector rotations to guide portfolio asset creation.\n"
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Time Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime and leading/lagging sectors.\n"
            f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        # Phase 3: Sector Allocation Decision (Portfolio Manager decides directly from Macro)
        self._newPhaseHeader(3, "Sector Allocation Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")

        pmSectorPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, determine the executive sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
            f"- Allocations must sum to approximately 100.0%.\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Review market breadth (% > 50 SMA), constituent divergence (median constituent return vs ETF return), relative strength channels (1Y/3Y vs SPY), 10Y Treasury beta, and constituent FinBERT sentiment.\n"
            f"2. Execute the 'confirmSectorAllocation' tool with your 'sectorAllocations' dictionary (e.g. {{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) and executive 'rationale'."
        )
        pmSectorRaw, pmSectorUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=8,
            requireInitialTools=True
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        confirmedSectorsText = self._formatSectorsContext()

        # Phase 4: Stock Scouting (Growth Hunter & Value Hunter concurrently)
        self._newPhaseHeader(4, "Stock Scouting", pace)
        growthPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
            f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive, dividend, and value equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
            f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='defensive' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, valuation, and margin of safety rationale."
        )

        (growthRaw, growthUISummary), (valueRaw, valueUISummary) = self._runAgentsConcurrently(
            lambda: self.growthHunter.analyseAndReply(
                incomingMessage=growthPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: self.valueHunter.analyseAndReply(
                incomingMessage=valuePrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        # Phase 6: Final Executive Decision (Portfolio Manager synthesizes directly from Hunters in Fast mode)
        self._newPhaseHeader(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")

        pmFinalPrompt = (
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Growth Stock Hunter Scouting:\n{growthRaw}\n\n"
            f"Value/Defensive Stock Hunter Scouting:\n{valueRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, make the final executive decision to construct the portfolio for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary. Every confirmed non-cash sector must contain stock holdings.\n"
            f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. 60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
            f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
        )

        pmFinalRaw, pmFinalUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=8,
            requireInitialTools=True
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation
        })

        self._recordAndSaveSession(
            pace=pace,
            startTime=startTime,
            config=config,
            macroUISummary=macroUISummary,
            macroRaw=macroRaw,
            growthUISummary=growthUISummary,
            growthRaw=growthRaw,
            valueUISummary=valueUISummary,
            valueRaw=valueRaw,
            pmFinalUISummary=pmFinalUISummary,
            pmFinalRaw=pmFinalRaw,
            isFast=True
        )

    def executeCompletePortfolioCreation(self, config: PortfolioCreationConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Complete Portfolio Creation Boardroom (Pace: COMPLETE)\n{'='*70}")

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and examine macro sector rotations to guide portfolio asset creation.\n"
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Time Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime and leading/lagging sectors.\n"
            f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        # Phase 2: Sector Allocation Analysis (Bullish & Bearish Analysts in parallel)
        self._newPhaseHeader(2, "Sector Allocation Analysis", pace)
        bullPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a growth and cyclical sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Assess market breadth (% > 50 SMA), constituent divergence (median constituent return vs ETF return), relative strength channels (1Y/3Y vs SPY), 10Y Treasury beta, and constituent FinBERT sentiment.\n"
            f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
            f"3. Highlight growth catalysts, healthy breadth participation, and upside drivers."
        )

        bearPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a defensive, risk-managed sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Scrutinize narrow rallies (where ETF return >> median constituent return), deteriorating breadth (< 50% above 50 SMA), high channel valuation extremes, 10Y Treasury yield vulnerability, and bearish sentiment.\n"
            f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
            f"3. Highlight vulnerabilities, drawdown risks, and defensive hedges."
        )

        (bullSectorRaw, bullSectorUISummary), (bearSectorRaw, bearSectorUISummary) = self._runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(
                incomingMessage=bullPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: self.bearAnalyst.analyseAndReply(
                incomingMessage=bearPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        # Phase 3: Sector Allocation Decision (Impartial Portfolio Manager reconciles Bull & Bear)
        self._newPhaseHeader(3, "Sector Allocation Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")

        pmSectorPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Bullish Sector Proposal:\n{bullSectorRaw}\n\n"
            f"Bearish Sector Proposal:\n{bearSectorRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the Bullish and Bearish proposals to make the definitive sector allocation decision for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
            f"- Allocations must sum to approximately 100.0%.\n\n"
            f"1. You may call 'fetchAllSectorsAnalysis' to inspect or cross-verify the underlying breadth, constituent divergence, and sentiment metrics cited by the Bull and Bear (returns instantly from cache).\n"
            f"2. Execute the 'confirmSectorAllocation' tool with your 'sectorAllocations' dictionary (e.g. {{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) and executive 'rationale'."
        )
        pmSectorRaw, pmSectorUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=8,
            requireInitialTools=True
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        confirmedSectorsText = self._formatSectorsContext()

        # Phase 4: Stock Scouting (Growth Hunter & Value Hunter in parallel)
        self._newPhaseHeader(4, "Stock Scouting", pace)
        growthPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
            f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive, dividend, and value equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
            f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='defensive' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, valuation, and margin of safety rationale."
        )

        (growthRaw, growthUISummary), (valueRaw, valueUISummary) = self._runAgentsConcurrently(
            lambda: self.growthHunter.analyseAndReply(
                incomingMessage=growthPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: self.valueHunter.analyseAndReply(
                incomingMessage=valuePrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        # Phase 5: Stock Allocation Proposals (Aggressive & Conservative Risk Analysts in parallel)
        self._newPhaseHeader(5, "Stock Allocation Proposals", pace)
        aggProposalPrompt = (
            f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final aggressive individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- Sector Alignment: Group your stock proposals strictly under each confirmed sector above.\n"
            f"- Per-Sector Weighting: Inside each confirmed sector, propose high-beta growth stocks with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining catalysts and beta strategy.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
        )

        consProposalPrompt = (
            f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final conservative individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- Sector Alignment: Group your stock proposals strictly under each confirmed sector above.\n"
            f"- Per-Sector Weighting: Inside each confirmed sector, propose defensive, low-volatility equities with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining margin of safety and downside protection.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
        )

        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                incomingMessage=aggProposalPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=False
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                incomingMessage=consProposalPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=False
            )
        )

        # Phase 6: Final Executive Decision (Impartial Portfolio Manager reconciles Risk proposals)
        self._newPhaseHeader(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")

        pmFinalPrompt = (
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
            f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the Aggressive and Conservative proposals to make the definitive portfolio construction decision for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary. Every confirmed non-cash sector must contain stock holdings.\n"
            f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. 60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
            f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
        )

        pmFinalRaw, pmFinalUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=8,
            requireInitialTools=True
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation
        })

        self._recordAndSaveSession(
            pace=pace,
            startTime=startTime,
            config=config,
            macroUISummary=macroUISummary,
            macroRaw=macroRaw,
            growthUISummary=growthUISummary,
            growthRaw=growthRaw,
            valueUISummary=valueUISummary,
            valueRaw=valueRaw,
            pmFinalUISummary=pmFinalUISummary,
            pmFinalRaw=pmFinalRaw,
            bullSectorUISummary=bullSectorUISummary,
            bullSectorRaw=bullSectorRaw,
            bearSectorUISummary=bearSectorUISummary,
            bearSectorRaw=bearSectorRaw,
            aggProposalUISummary=aggProposalUISummary,
            aggProposalRaw=aggProposalRaw,
            consProposalUISummary=consProposalUISummary,
            consProposalRaw=consProposalRaw,
            isFast=False
        )

    def executePresetPortfolioCreation(self, config: PortfolioCreationConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Pre-set Sector Allocation Portfolio Creation Boardroom (Pace: PRESET)\n{'='*70}")

        # Populate confirmed sector allocation directly from user pre-set inputs
        presetMap = config.presetSectorAllocations or {}
        cleanedAllocations = {}

        for ticker, info in GICS_SECTORS.items():
            rawPct = presetMap.get(ticker, presetMap.get(info.name, 0.0))
            try:
                pctVal = round(float(rawPct), 2)
            except (ValueError, TypeError):
                pctVal = 0.0

            if pctVal > 0:
                cleanedAllocations[ticker] = {
                    "sector": info.name,
                    "ticker": ticker,
                    "allocationPct": pctVal
                }

        if cleanedAllocations:
            from llmtools.functions.confirmation import distributeIntegerPercentages
            keys = list(cleanedAllocations.keys())
            rawWeights = [cleanedAllocations[k]["allocationPct"] for k in keys]
            intAllocations = distributeIntegerPercentages(rawWeights, 100)
            for k, intVal in zip(keys, intAllocations):
                cleanedAllocations[k]["allocationPct"] = int(intVal)

        totalAllocated = sum(item["allocationPct"] for item in cleanedAllocations.values())
        self.confirmedSectorAllocation = {
            "sectorAllocations": cleanedAllocations,
            "totalAllocatedPct": totalAllocated,
            "sectorCount": len(cleanedAllocations),
            "rationale": "Pre-set sector allocation configured by user."
        }

        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")
        if confirmSectorTool:
            confirmSectorTool.toolLog.append(self.confirmedSectorAllocation)

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        confirmedSectorsText = self._formatSectorsContext()
        activeSectorsSummary = ", ".join([f"{item['sector']} ({int(item['allocationPct'])}%)" for item in cleanedAllocations.values()])

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis to contextualize and guide portfolio equity selection.\n"
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Time Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
            f"MANDATED PRE-SET SECTOR ALLOCATION (EXECUTIVE DIRECTIVE):\n"
            f"{confirmedSectorsText}\n\n"
            f"CRITICAL DIRECTIVE ON SECTOR ALLOCATION:\n"
            f"The boardroom executive mandate has ALREADY established and locked the portfolio sector allocation to: {activeSectorsSummary}. "
            f"Do NOT propose an alternative sector allocation, and do NOT advise underweighting or avoiding the mandated sector(s). "
            f"Instead, analyze current macroeconomic indicators, market regime, inflation, yields, and sector rotation to assess "
            f"how the broader economic backdrop specifically impacts the mandated sector(s) ({activeSectorsSummary}), and evaluate which equity "
            f"attributes, fundamental factors, and risk exposures are best positioned to navigate prevailing market conditions within the mandated sector(s).\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime and economic indicators.\n"
            f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
            f"3. Deliver strategic equity selection guidance tailored strictly to the mandated sectors: {activeSectorsSummary}."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        # Stages 2 and 3 are skipped because the sector allocation is already pre-set

        # Phase 4: Stock Scouting (Growth Hunter & Value Hunter concurrently)
        self._newPhaseHeader(4, "Stock Scouting", pace)
        growthPrompt = (
            f"MANDATED SECTOR ALLOCATIONS (STRICT & BINDING):\n{confirmedSectorsText}\n\n"
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities STRICTLY within the confirmed sectors above.\n"
            f"MANDATORY CONSTRAINTS:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- TARGET STOCK COUNT: The target total stock count across the entire portfolio is {promptArgs['targetStockCount']}. "
            f"Keep your shortlisted candidate count focused and calibrated so the boardroom does not exceed this count.\n"
            f"- STRICT SECTOR BOUNDARY: You MUST ONLY scout candidate equities belonging to the confirmed sectors ({activeSectorsSummary}). "
            f"Do NOT scout or propose equities from any other sectors, regardless of any general macro commentary.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"MANDATED SECTOR ALLOCATIONS (STRICT & BINDING):\n{confirmedSectorsText}\n\n"
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive and value equities STRICTLY within the confirmed sectors above.\n"
            f"MANDATORY CONSTRAINTS:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- TARGET STOCK COUNT: The target total stock count across the entire portfolio is {promptArgs['targetStockCount']}. "
            f"Keep your shortlisted candidate count focused and calibrated so the boardroom does not exceed this count.\n"
            f"- STRICT SECTOR BOUNDARY: You MUST ONLY scout candidate equities belonging to the confirmed sectors ({activeSectorsSummary}). "
            f"Do NOT scout or propose equities from any other sectors, regardless of any general macro commentary.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='defensive' for each confirmed sector to screen candidate equities.\n"
            f"2. You MUST afterwards use 'fetchBatchStockOverviews' to retrieve much more detailed information on your top high-conviction picks.\n"
            f"3. Present your candidate table with tickers, industry, valuation, and margin of safety rationale."
        )

        (growthRaw, growthUISummary), (valueRaw, valueUISummary) = self._runAgentsConcurrently(
            lambda: self.growthHunter.analyseAndReply(
                incomingMessage=growthPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: self.valueHunter.analyseAndReply(
                incomingMessage=valuePrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        # Phase 5: Stock Allocation Proposals (Aggressive & Conservative Risk Analysts in parallel)
        self._newPhaseHeader(5, "Stock Allocation Proposals", pace)
        aggProposalPrompt = (
            f"Confirmed Pre-Set Sector Allocations (MANDATORY & BINDING):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final aggressive individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"MANDATORY CONSTRAINTS:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- STRICT SECTOR ALIGNMENT: Group stocks strictly under each confirmed sector ({activeSectorsSummary}).\n"
            f"- Per-Sector Weighting: Inside each confirmed sector, propose high-beta growth stocks with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining catalysts and beta strategy.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
        )

        consProposalPrompt = (
            f"Confirmed Pre-Set Sector Allocations (MANDATORY & BINDING):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final conservative individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"MANDATORY CONSTRAINTS:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- STRICT SECTOR ALIGNMENT: Group stocks strictly under each confirmed sector ({activeSectorsSummary}).\n"
            f"- Per-Sector Weighting: Inside each confirmed sector, propose defensive, low-volatility equities with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining margin of safety and downside protection.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
        )

        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = self._runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(
                incomingMessage=aggProposalPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=False
            ),
            lambda: self.consRiskAnalyst.analyseAndReply(
                incomingMessage=consProposalPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=False
            )
        )

        # Phase 6: Final Executive Decision (Impartial Portfolio Manager reconciles Risk proposals)
        self._newPhaseHeader(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")

        pmFinalPrompt = (
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
            f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the Aggressive and Conservative proposals to make the definitive portfolio construction decision for {promptArgs['initialCapital']}.\n"
            f"MANDATORY CONSTRAINTS:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
            f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary ({activeSectorsSummary}). Every confirmed sector must contain stock holdings.\n"
            f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. 60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
            f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
        )

        pmFinalRaw, pmFinalUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=8,
            requireInitialTools=True
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation
        })

        self._recordAndSaveSession(
            pace=pace,
            startTime=startTime,
            config=config,
            macroUISummary=macroUISummary,
            macroRaw=macroRaw,
            growthUISummary=growthUISummary,
            growthRaw=growthRaw,
            valueUISummary=valueUISummary,
            valueRaw=valueRaw,
            pmFinalUISummary=pmFinalUISummary,
            pmFinalRaw=pmFinalRaw,
            bullSectorUISummary=None,
            bullSectorRaw=None,
            bearSectorUISummary=None,
            bearSectorRaw=None,
            aggProposalUISummary=aggProposalUISummary,
            aggProposalRaw=aggProposalRaw,
            consProposalUISummary=consProposalUISummary,
            consProposalRaw=consProposalRaw,
            isFast=False
        )

    def _recordAndSaveSession(
        self,
        pace: BoardroomPace,
        startTime: datetime,
        config: PortfolioCreationConfig,
        macroUISummary: str,
        macroRaw: str,
        growthUISummary: str,
        growthRaw: str,
        valueUISummary: str,
        valueRaw: str,
        pmFinalUISummary: str,
        pmFinalRaw: str,
        bullSectorUISummary: Optional[str] = None,
        bullSectorRaw: Optional[str] = None,
        bearSectorUISummary: Optional[str] = None,
        bearSectorRaw: Optional[str] = None,
        aggProposalUISummary: Optional[str] = None,
        aggProposalRaw: Optional[str] = None,
        consProposalUISummary: Optional[str] = None,
        consProposalRaw: Optional[str] = None,
        isFast: bool = False
    ):
        endTime = datetime.now()
        timeTaken = endTime - startTime
        timeStr = f"{timeTaken.seconds // 60} mins {timeTaken.seconds % 60} secs"

        print()
        self._newPhaseHeader(0, f"Final Portfolio Creation Summary", pace)

        separator = f"\n{ANSI.BOLD}{ANSI.DIM}{'-'*70}{ANSI.RESET}\n"

        if isFast:
            shortSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Summary:\n{ANSI.RESET}{macroUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Summary:\n{ANSI.RESET}{growthUISummary}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Summary:\n{ANSI.RESET}{valueUISummary}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalUISummary}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )
            fullSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Analysis:\n{ANSI.RESET}{macroRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Scouting:\n{ANSI.RESET}{growthRaw}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Scouting:\n{ANSI.RESET}{valueRaw}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalRaw}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )
        elif pace == BoardroomPace.PRESET:
            presetSectorText = self._formatSectorsContext()
            shortSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Summary:\n{ANSI.RESET}{macroUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{ANSI.MAGENTA}Pre-set Sector Allocation:\n{ANSI.RESET}{presetSectorText}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Summary:\n{ANSI.RESET}{growthUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Summary:\n{ANSI.RESET}{valueUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Proposal Summary:\n{ANSI.RESET}{aggProposalUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Proposal Summary:\n{ANSI.RESET}{consProposalUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalUISummary}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )
            fullSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Analysis:\n{ANSI.RESET}{macroRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{ANSI.MAGENTA}Pre-set Sector Allocation:\n{ANSI.RESET}{presetSectorText}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Scouting:\n{ANSI.RESET}{growthRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Scouting:\n{ANSI.RESET}{valueRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Allocation Proposal:\n{ANSI.RESET}{aggProposalRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Allocation Proposal:\n{ANSI.RESET}{consProposalRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalRaw}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )
        else:
            shortSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Summary:\n{ANSI.RESET}{macroUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Sector Summary:\n{ANSI.RESET}{bullSectorUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Sector Summary:\n{ANSI.RESET}{bearSectorUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Summary:\n{ANSI.RESET}{growthUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Summary:\n{ANSI.RESET}{valueUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Risk Proposal Summary:\n{ANSI.RESET}{aggProposalUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Risk Proposal Summary:\n{ANSI.RESET}{consProposalUISummary}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalUISummary}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )
            fullSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Analysis:\n{ANSI.RESET}{macroRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.bullAnalyst.color}Bullish Sector Thesis:\n{ANSI.RESET}{bullSectorRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.bearAnalyst.color}Bearish Sector Thesis:\n{ANSI.RESET}{bearSectorRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Scouting:\n{ANSI.RESET}{growthRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Scouting:\n{ANSI.RESET}{valueRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.aggRiskAnalyst.color}Aggressive Allocation Proposal:\n{ANSI.RESET}{aggProposalRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.consRiskAnalyst.color}Conservative Allocation Proposal:\n{ANSI.RESET}{consProposalRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalRaw}\n"
                f"\nConfirmed Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio creation: {timeStr}\n"
            )

        print(shortSummary)

        os.makedirs("output", exist_ok=True)
        if isFast:
            filenamePace = "fast"
        elif pace == BoardroomPace.PRESET:
            filenamePace = "preset"
        else:
            filenamePace = "complete"
        filepath = os.path.join("output", f"portfolio_{filenamePace}_{startTime.strftime('%Y-%m-%d_%H-%M-%S')}.ans")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fullSummary)

        self.lastConfig = config
        self.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', fullSummary)

    def execute(self, config: PortfolioCreationConfig) -> None:
        if self.llmClient is None:
            raise ValueError("LLM client is not assigned. Please assign a client before executing the boardroom.")

        self.toolRegistry.clearToolLogs()
        self.llmClient.newTask()
        self.confirmedSectorAllocation = None
        self.confirmedPortfolioAllocation = None

        if config.boardroomPace == BoardroomPace.FAST:
            self.executeFastPortfolioCreation(config)
        elif config.boardroomPace == BoardroomPace.PRESET:
            self.executePresetPortfolioCreation(config)
        else:
            self.executeCompletePortfolioCreation(config)

