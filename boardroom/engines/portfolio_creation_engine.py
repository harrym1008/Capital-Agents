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
                toolMap["fetchAllSectorsPerformance"],
                toolMap["fetchAllSectorProfiles"],
                toolMap["fetchAllSectorRankings"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.GREEN,
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchAllSectorsPerformance"],
                toolMap["fetchAllSectorProfiles"],
                toolMap["fetchAllSectorRankings"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
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
                toolMap["fetchBatchFinnhubMetrics"],
                toolMap["fetchFinnhubCompanyFundamentals"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyValuationMetrics"],
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
                toolMap["fetchBatchFinnhubMetrics"],
                toolMap["fetchFinnhubCompanyFundamentals"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchBalanceSheet"],
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
                toolMap["fetchBatchFinnhubMetrics"],
                toolMap["fetchFinnhubCompanyFundamentals"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            ansiColor=ANSI.YELLOW,
            dateStr=timestampStr
        )

        self.consRiskAnalyst = FinancialAgent(
            agentRole="Conservative Risk Analyst",
            tools=[
                toolMap["fetchBatchFinnhubMetrics"],
                toolMap["fetchFinnhubCompanyFundamentals"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyValuationMetrics"],
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
                {"role": "Growth Stock Hunter", "color": self.growthHunter.colorName, "name": "Growth Hunter"},
                {"role": "Value/Defensive Stock Hunter", "color": self.valueHunter.colorName, "name": "Value Hunter"}
            ]
        elif phaseNumber == 5:
            return [
                {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.colorName, "name": "Aggressive Risk"},
                {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.colorName, "name": "Conservative Risk"}
            ]
        elif phaseNumber == 6:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.colorName, "name": "Portfolio Manager"}]

        return []

    def _formatSectorsContext(self) -> str:
        if not self.confirmedSectorAllocation:
            return "No confirmed sector allocation available."
        confirmed = self.confirmedSectorAllocation.get("confirmedAllocation", self.confirmedSectorAllocation)
        sectorsDict = confirmed.get("sectorAllocations", {})
        lines = [f"- {secInfo.get('sector', secKey)} ({secKey}): {secInfo.get('allocationPct', 0.0)}%" for secKey, secInfo in sectorsDict.items()]
        return "\n".join(lines)


    def executeFastPortfolioCreation(self, config: PortfolioCreationConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Fast Portfolio Creation Boardroom (Pace: FAST)\nCapital: {promptArgs['initialCapital']} | Sector Rule: {promptArgs['sectorDiversityRule']}\n{'='*70}")

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
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

        # Phase 3: Sector Allocation Decision (Portfolio Manager decides directly from Macro)
        self._newPhaseHeader(3, "Sector Allocation Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")

        pmSectorPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, determine the executive sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
            f"- Allocations must sum to approximately 100.0%.\n\n"
            f"Execute the 'confirmSectorAllocation' tool with your exact sector dictionary and rationale."
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
            f"Confirmed Portfolio Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities across the confirmed sectors.\n"
            f"Portfolio Constraints: Max single stock allocation: {promptArgs['maxStockAllocation']}. Target count: {promptArgs['targetStockCount']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
            f"2. Use 'fetchBatchFinnhubMetrics' or 'fetchFinnhubCompanyFundamentals' to retrieve fast valuation and margins on your top 2-3 high-conviction picks per sector.\n"
            f"3. Present your candidate table with tickers, industry, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive, dividend, and value equities across the confirmed sectors.\n"
            f"Portfolio Constraints: Max single stock allocation: {promptArgs['maxStockAllocation']}. Target count: {promptArgs['targetStockCount']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='defensive' or 'value' for each confirmed sector to screen candidate equities.\n"
            f"2. Use 'fetchBatchFinnhubMetrics' or 'fetchFinnhubCompanyFundamentals' to evaluate valuation, debt-to-equity, and margins on your top 2-3 high-conviction picks per sector.\n"
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
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Growth Stock Hunter Scouting:\n{growthRaw}\n\n"
            f"Value/Defensive Stock Hunter Scouting:\n{valueRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, make the final executive decision to construct the portfolio for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Respect confirmed sector totals: stock holdings per sector must sum to the sector's allocated percentage.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Target stock count: {promptArgs['targetStockCount']}.\n"
            f"- Total portfolio allocation (stocks + optional cash buffer) must equal 100.0%.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your exact positions array, portfolioRationale, and cashWeightPct."
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

        print(f"\n{'='*70}\nStarting Complete Portfolio Creation Boardroom (Pace: COMPLETE)\nCapital: {promptArgs['initialCapital']} | Sector Rule: {promptArgs['sectorDiversityRule']}\n{'='*70}")

        # Phase 1: Macro Environment Analysis (Macro Strategist)
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
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

        # Phase 2: Sector Allocation Analysis (Bullish & Bearish Analysts in parallel)
        self._newPhaseHeader(2, "Sector Allocation Analysis", pace)
        bullPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a growth and cyclical sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Constraints: Max single sector allocation is {promptArgs['maxSectorAllocation']}. {promptArgs['sectorDiversityRule']}\n"
            f"1. Use 'fetchAllSectorsPerformance', 'fetchAllSectorProfiles', and 'fetchAllSectorRankings' to comprehensively assess all 11 GICS sectors.\n"
            f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
            f"3. Highlight growth catalysts and upside drivers."
        )

        bearPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a defensive, risk-managed sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Constraints: Max single sector allocation is {promptArgs['maxSectorAllocation']}. {promptArgs['sectorDiversityRule']}\n"
            f"1. Use 'fetchAllSectorsPerformance', 'fetchAllSectorProfiles', and 'fetchAllSectorRankings' to comprehensively assess all 11 GICS sectors.\n"
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
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
            f"- Allocations must sum to approximately 100.0%.\n\n"
            f"Execute the 'confirmSectorAllocation' tool with your exact sector dictionary and rationale."
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
            f"Confirmed Portfolio Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities across the confirmed sectors.\n"
            f"Portfolio Constraints: Max single stock allocation: {promptArgs['maxStockAllocation']}. Target count: {promptArgs['targetStockCount']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
            f"2. Use 'fetchBatchFinnhubMetrics' or 'fetchFinnhubCompanyFundamentals' to retrieve fast valuation and margins on your top 2-3 high-conviction picks per sector.\n"
            f"3. Present your candidate table with tickers, industry, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Confirmed Portfolio Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive, dividend, and value equities across the confirmed sectors.\n"
            f"Portfolio Constraints: Max single stock allocation: {promptArgs['maxStockAllocation']}. Target count: {promptArgs['targetStockCount']}.\n\n"
            f"1. Use 'fetchStocksInSector' with style='defensive' or 'value' for each confirmed sector to screen candidate equities.\n"
            f"2. Use 'fetchBatchFinnhubMetrics' or 'fetchFinnhubCompanyFundamentals' to evaluate valuation, debt-to-equity, and margins on your top 2-3 high-conviction picks per sector.\n"
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
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Value Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final aggressive individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Constraints: Max single stock allocation is {promptArgs['maxStockAllocation']}. Stock holdings per sector must adhere to confirmed sector limits.\n"
            f"Propose a comprehensive portfolio table (% per asset, dollar allocation, and role) overweighting high-beta growth drivers summing to 100.0%."
        )

        consProposalPrompt = (
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Growth Candidates Scouted:\n{growthRaw}\n\n"
            f"Value Candidates Scouted:\n{valueRaw}\n\n"
            f"Task: Construct your final conservative individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Constraints: Max single stock allocation is {promptArgs['maxStockAllocation']}. Stock holdings per sector must adhere to confirmed sector limits.\n"
            f"Propose a comprehensive portfolio table (% per asset, dollar allocation, and role) emphasizing dividend stability, lower volatility, and margin of safety summing to 100.0%."
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
            f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
            f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the Aggressive and Conservative proposals to make the definitive portfolio construction decision for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Respect confirmed sector totals: stock holdings per sector must sum to the sector's allocated percentage.\n"
            f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"- Target stock count: {promptArgs['targetStockCount']}.\n"
            f"- Total portfolio allocation (stocks + optional cash buffer) must equal 100.0%.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your exact positions array, portfolioRationale, and cashWeightPct."
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
        filenamePace = "fast" if isFast else "complete"
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
        else:
            self.executeCompletePortfolioCreation(config)
