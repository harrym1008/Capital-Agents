import os
import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from cli.ansi import ANSI
from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import PortfolioRebalancingConfig, BoardroomConfig, BoardroomPace
from boardroom.boardroom_engine import BoardroomEngine
from ui.ui_hooks import setCurrentStage, setCurrentAgent, setAgentPhase, emitEvent, SimulationStoppedException
from collectors.sector_dl_client import GICS_SECTORS



class PortfolioRebalancingBoardroomEngine(BoardroomEngine):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        super().__init__(toolRegistry, timestamp)
        self.confirmedSectorAllocation: Optional[Dict[str, Any]] = None
        self.confirmedPortfolioAllocation: Optional[Dict[str, Any]] = None
        self.lastConfig: Optional[PortfolioRebalancingConfig] = None
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

    def executeFastPortfolioRebalancing(self, config: PortfolioRebalancingConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Fast Portfolio Rebalancing Boardroom (Pace: FAST)")

        # Phase 1: Macro & Baseline Portfolio Health Audit
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and audit the existing baseline portfolio against prevailing market conditions.\n"
            f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
            f"Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n\n"
            f"CURRENT BASELINE PORTFOLIO HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTOR BREAKDOWN:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime, inflation, rates, and leading/lagging sectors.\n"
            f"2. Audit the current portfolio: Assess which sectors are overconcentrated or facing headwinds, and identify which emerging or resilient sectors are underrepresented.\n"
            f"3. Deliver your macro narrative, economic indicator table, market regime classification, and portfolio vulnerability diagnosis calibrated to the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        # Phase 3: Sector Rebalancing Decision
        self._newPhaseHeader(3, "Sector Rebalancing Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")

        pmSectorPrompt = (
            f"Macro & Portfolio Audit Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE PORTFOLIO HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTOR BREAKDOWN:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"Task: As the Impartial Portfolio Manager, determine the target rebalanced sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Allocations must sum to approximately 100.0% across equity sectors.\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Review market breadth, constituent divergence, relative strength channels, and FinBERT sentiment.\n"
            f"2. Compare your target sector weights directly against the baseline sector breakdown. You MUST explain the rationale for each difference (e.g. why specific sectors were expanded, trimmed, eliminated, or newly introduced) in accordance with the Rebalance Mandate.\n"
            f"3. Execute the 'confirmSectorAllocation' tool with your target 'sectorAllocations' dictionary (e.g. {{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) and executive 'rationale' detailing these sector-level shifts."
        )
        pmSectorConfirmationPrompt = (
            f"Your sector allocation has been verified and logged.\n"
            f"Please now provide your detailed executive remarks explaining the sector changes you are making compared to the original baseline portfolio.\n"
            f"Specifically reference the original baseline sector breakdown, detail each sector difference/delta (weights increased, decreased, eliminated, or newly added), "
            f"and explain the economic and risk rationale behind each difference in alignment with the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )
        pmSectorRaw, pmSectorUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=8,
            requireInitialTools=True,
            confirmationPrompt=pmSectorConfirmationPrompt
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        confirmedSectorsText = self._formatSectorsContext()

        # Phase 4: Stock Holdings Audit (Growth & Value Hunters concurrently)
        self._newPhaseHeader(4, "Stock Holdings Audit", pace)
        growthPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, evaluate existing growth holdings and scout high-conviction momentum/growth replacements strictly within the target sectors above.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"1. Audit existing baseline growth holdings: recommend which to KEEP vs. REPLACE or TRIM based on current growth catalysts and the rebalance mandate.\n"
            f"2. Use 'fetchStocksInSector' with style='growth' for confirmed sectors to screen replacement or expansion candidates.\n"
            f"3. Use 'fetchBatchStockOverviews' on top conviction picks and present your candidate table."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, evaluate existing defensive/value holdings and scout high-conviction defensive replacements strictly within the target sectors above.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"1. Audit existing baseline defensive/value holdings: recommend which to KEEP vs. REPLACE or TRIM based on margin of safety and the rebalance mandate.\n"
            f"2. Use 'fetchStocksInSector' with style='defensive' for confirmed sectors to screen candidate equities.\n"
            f"3. Use 'fetchBatchStockOverviews' on top conviction picks and present your candidate table."
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

        # Phase 6: Final Executive Rebalancing Decision
        self._newPhaseHeader(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")
        if confirmPortTool:
            origConfirmPortFunc = confirmPortTool.function
            def boundConfirmPortfolioAllocation(*args, **kwargs):
                if "initialCapital" not in kwargs or not kwargs["initialCapital"]:
                    kwargs["initialCapital"] = config.initialCapital
                return origConfirmPortFunc(*args, **kwargs)
            confirmPortTool.function = boundConfirmPortfolioAllocation

        pmFinalPrompt = (
            f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
            f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"Target Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Growth Stock Hunter Scouting:\n{growthRaw}\n\n"
            f"Value/Defensive Stock Hunter Scouting:\n{valueRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, construct the definitive rebalanced portfolio for {promptArgs['initialCapital']}.\n"
            f"You may retain strong existing holdings from the baseline portfolio, adjust their weights, replace underperforming names, and introduce high-conviction new additions in accordance with the Rebalance Mandate ({promptArgs['rebalanceAmount']}).\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across confirmed sectors.\n"
            f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary.\n"
            f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages summing strictly to 100% of that sector.\n"
            f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
            f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100-150 words detailing the rebalancing strategy, explicitly comparing the rebalanced portfolio to the original baseline holdings and explaining the differences.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary, 'portfolioRationale', and 'initialCapital'={config.initialCapital}."
        )

        pmFinalConfirmationPrompt = (
            f"Your rebalanced portfolio allocation has been verified, logged, and confirmed.\n"
            f"Please now provide your definitive executive report explaining the changes you have made to the portfolio.\n"
            f"Explicitly reference the original baseline portfolio holdings: identify which original stocks were retained (and whether their allocation was weighted up or down), "
            f"which original stocks were trimmed or completely exited, which new stocks were added, and clearly explain the strategic rationale behind all differences in light of the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )

        pmFinalRaw, pmFinalUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=8,
            requireInitialTools=True,
            confirmationPrompt=pmFinalConfirmationPrompt
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation,
            "originalPositions": config.currentPositions
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

    def executeCompletePortfolioRebalancing(self, config: PortfolioRebalancingConfig):
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        startTime = datetime.now()

        print(f"\n{'='*70}\nStarting Complete Portfolio Rebalancing Boardroom (Pace: COMPLETE)\n{'='*70}")

        # Phase 1: Macro & Baseline Portfolio Health Audit
        self._newPhaseHeader(1, "Macro Environment Analysis", pace)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and audit the existing baseline portfolio against prevailing market conditions.\n"
            f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
            f"Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n\n"
            f"CURRENT BASELINE PORTFOLIO HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTOR BREAKDOWN:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyze market regime, inflation, rates, and leading/lagging sectors.\n"
            f"2. Audit the current portfolio: Assess which sectors are overconcentrated or facing headwinds, and identify which emerging or resilient sectors are underrepresented.\n"
            f"3. Deliver your macro narrative, economic indicator table, market regime classification, and portfolio vulnerability diagnosis calibrated to the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )
        macroRaw, macroUISummary = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True
        )

        # Phase 2: Sector Rebalancing Analysis (Bull & Bear concurrently)
        self._newPhaseHeader(2, "Sector Rebalancing Analysis", pace)
        bullPrompt = (
            f"Macro & Portfolio Audit Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTORS:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"Task: Propose an opportunistic, growth-oriented sector rebalancing allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors.\n"
            f"2. Propose which existing sectors to expand or maintain and identify new growth sectors to introduce under the rebalance mandate.\n"
            f"3. Present a clear table of percentage allocations across your selected sectors summing to 100.0%."
        )

        bearPrompt = (
            f"Macro & Portfolio Audit Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTORS:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"Task: Propose a defensive, risk-managed sector rebalancing allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Scrutinize narrow rallies and deteriorating breadth.\n"
            f"2. Propose defensive trims or exits from overextended or vulnerable sectors in the current portfolio under the rebalance mandate.\n"
            f"3. Present a clear table of percentage allocations across your selected sectors summing to 100.0%."
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

        # Phase 3: Sector Rebalancing Decision
        self._newPhaseHeader(3, "Sector Rebalancing Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")

        pmSectorPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTORS:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"Bullish Sector Proposal:\n{bullSectorRaw}\n\n"
            f"Bearish Sector Proposal:\n{bearSectorRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the Bullish and Bearish proposals to make the definitive target sector allocation decision for {promptArgs['initialCapital']}.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Allocations must sum to approximately 100.0% across equity sectors.\n\n"
            f"1. You may call 'fetchAllSectorsAnalysis' to verify metrics.\n"
            f"2. Compare your target sector weights directly against the baseline sector breakdown. You MUST explain the rationale for each difference (e.g. why specific sectors were expanded, trimmed, eliminated, or newly introduced) in accordance with the Rebalance Mandate.\n"
            f"3. Execute the 'confirmSectorAllocation' tool with your target 'sectorAllocations' dictionary and executive 'rationale' detailing these sector-level shifts."
        )
        pmSectorConfirmationPrompt = (
            f"Your sector allocation has been verified and logged.\n"
            f"Please now provide your detailed executive remarks explaining the sector changes you are making compared to the original baseline portfolio.\n"
            f"Specifically reference the original baseline sector breakdown, detail each sector difference/delta (weights increased, decreased, eliminated, or newly added), "
            f"and explain the economic and risk rationale behind each difference in alignment with the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )
        pmSectorRaw, pmSectorUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=8,
            requireInitialTools=True,
            confirmationPrompt=pmSectorConfirmationPrompt
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation
        })

        confirmedSectorsText = self._formatSectorsContext()

        # Phase 4: Stock Holdings Audit (Growth & Value Hunters concurrently)
        self._newPhaseHeader(4, "Stock Holdings Audit", pace)
        growthPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Confirmed Target Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, evaluate existing growth holdings and scout high-conviction growth equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"1. Audit existing baseline growth holdings: recommend which to KEEP vs. REPLACE or TRIM based on current growth catalysts and the rebalance mandate.\n"
            f"2. Use 'fetchStocksInSector' with style='growth' for confirmed sectors to screen candidate equities.\n"
            f"3. Use 'fetchBatchStockOverviews' on top conviction picks and present your candidate table."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Confirmed Target Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Value/Defensive Stock Hunter, evaluate existing defensive/value holdings and scout high-conviction defensive equities across the confirmed sectors.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"1. Audit existing baseline defensive/value holdings: recommend which to KEEP vs. REPLACE or TRIM based on margin of safety and the rebalance mandate.\n"
            f"2. Use 'fetchStocksInSector' with style='defensive' for confirmed sectors to screen candidate equities.\n"
            f"3. Use 'fetchBatchStockOverviews' on top conviction picks and present your candidate table."
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

        # Phase 5: Rebalancing Stock Proposals (Aggressive & Conservative Analysts concurrently)
        self._newPhaseHeader(5, "Stock Reallocation Proposals", pace)
        aggProposalPrompt = (
            f"Target Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Growth Candidates Scouted in Phase 4:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted in Phase 4:\n{valueRaw}\n\n"
            f"Task: Review and scrutinize BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from an aggressive, high-upside risk perspective against the CURRENT BASELINE HOLDINGS. Construct your final aggressive stock rebalancing proposal for {promptArgs['initialCapital']}.\n\n"
            f"Instructions:\n"
            f"1. Evaluate candidate equities from both hunters against baseline positions: identify which baseline stocks to retain or expand for upside momentum, and which to replace with high-beta scouted picks while respecting the Rebalance Mandate.\n"
            f"2. Group stock proposals strictly under confirmed target sectors, assigning whole integer percentages summing strictly to 100% per sector.\n"
            f"3. Provide a concise 25-35 word rationale per stock explaining catalysts and beta strategy.\n"
            f"4. DO NOT ask challenge questions or create Q&A items. Output your comprehensive analysis and structured allocation proposal.\n\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target Stock Count: {promptArgs['targetStockCount']}.\n"
            f"- Sector Alignment: Group stock proposals strictly under confirmed target sectors.\n"
            f"- Per-Sector Weighting: Propose high-beta equities with whole integer percentages summing strictly to 100% per sector.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector, indicating retention or replacement of baseline holdings."
        )

        consProposalPrompt = (
            f"Target Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Growth Candidates Scouted in Phase 4:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted in Phase 4:\n{valueRaw}\n\n"
            f"Task: Review and scrutinize BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from a conservative, risk-managed capital-preservation perspective against the CURRENT BASELINE HOLDINGS. Construct your final conservative stock rebalancing proposal for {promptArgs['initialCapital']}.\n\n"
            f"Instructions:\n"
            f"1. Evaluate candidate equities from both hunters against baseline positions: identify core baseline holdings to preserve for safety and solvency, and determine which scouted defensive picks offer superior margin of safety while respecting the Rebalance Mandate.\n"
            f"2. Group stock proposals strictly under confirmed target sectors, assigning whole integer percentages summing strictly to 100% per sector.\n"
            f"3. Provide a concise 25-35 word rationale per stock explaining downside protection and margin of safety.\n"
            f"4. DO NOT ask challenge questions or create Q&A items. Output your comprehensive analysis and structured allocation proposal.\n\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target Stock Count: {promptArgs['targetStockCount']}.\n"
            f"- Sector Alignment: Group stock proposals strictly under confirmed target sectors.\n"
            f"- Per-Sector Weighting: Propose defensive equities with whole integer percentages summing strictly to 100% per sector.\n\n"
            f"Propose a comprehensive portfolio table grouped by confirmed sector, indicating retention or replacement of baseline holdings."
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

        # Phase 6: Final Executive Decision
        self._newPhaseHeader(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")
        if confirmPortTool:
            origConfirmPortFunc = confirmPortTool.function
            def boundConfirmPortfolioAllocation(*args, **kwargs):
                if "initialCapital" not in kwargs or not kwargs["initialCapital"]:
                    kwargs["initialCapital"] = config.initialCapital
                return origConfirmPortFunc(*args, **kwargs)
            confirmPortTool.function = boundConfirmPortfolioAllocation

        pmFinalPrompt = (
            f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
            f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"Confirmed Target Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
            f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile the proposals to construct the definitive rebalanced portfolio for {promptArgs['initialCapital']}.\n"
            f"You may retain strong existing holdings from the baseline portfolio, adjust their weights, replace underperforming names, and introduce high-conviction new additions in accordance with the Rebalance Mandate ({promptArgs['rebalanceAmount']}).\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across confirmed sectors.\n"
            f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary.\n"
            f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages summing strictly to 100% of that sector.\n"
            f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
            f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100-150 words detailing the rebalancing strategy, explicitly comparing the rebalanced portfolio to the original baseline holdings and explaining the differences.\n\n"
            f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary, 'portfolioRationale', and 'initialCapital'={config.initialCapital}."
        )

        pmFinalConfirmationPrompt = (
            f"Your rebalanced portfolio allocation has been verified, logged, and confirmed.\n"
            f"Please now provide your definitive executive report explaining the changes you have made to the portfolio.\n"
            f"Explicitly reference the original baseline portfolio holdings: identify which original stocks were retained (and whether their allocation was weighted up or down), "
            f"which original stocks were trimmed or completely exited, which new stocks were added, and clearly explain the strategic rationale behind all differences in light of the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )

        pmFinalRaw, pmFinalUISummary = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=8,
            requireInitialTools=True,
            confirmationPrompt=pmFinalConfirmationPrompt
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation,
            "originalPositions": config.currentPositions
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
        config: PortfolioRebalancingConfig,
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
        self._newPhaseHeader(0, f"Final Portfolio Rebalancing Summary", pace)

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
                f"\nConfirmed Rebalanced Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio rebalancing: {timeStr}\n"
            )
            fullSummary = (
                f"\n{ANSI.BOLD}{self.macroAnalyst.color}Macro Strategist Analysis:\n{ANSI.RESET}{macroRaw}\n"
                f"{separator}"
                f"\n{ANSI.BOLD}{self.growthHunter.color}Growth Stock Hunter Scouting:\n{ANSI.RESET}{growthRaw}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.valueHunter.color}Value/Defensive Stock Hunter Scouting:\n{ANSI.RESET}{valueRaw}\n"
                f"\n{separator}\n"
                f"\n{ANSI.BOLD}{self.portManager.color}Final Executive Decision:\n{ANSI.RESET}{pmFinalRaw}\n"
                f"\nConfirmed Rebalanced Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio rebalancing: {timeStr}\n"
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
                f"\nConfirmed Rebalanced Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio rebalancing: {timeStr}\n"
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
                f"\nConfirmed Rebalanced Portfolio: {self.confirmedPortfolioAllocation}\n"
                f"Time taken for portfolio rebalancing: {timeStr}\n"
            )

        print(shortSummary)

        os.makedirs("output", exist_ok=True)
        filenamePace = "fast" if isFast else "complete"
        filepath = os.path.join("output", f"rebalance_{filenamePace}_{startTime.strftime('%Y-%m-%d_%H-%M-%S')}.ans")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fullSummary)

        self.lastConfig = config
        self.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', fullSummary)

    def execute(self, config: PortfolioRebalancingConfig) -> None:
        if self.llmClient is None:
            raise ValueError("LLM client is not assigned. Please assign a client before executing the boardroom.")

        self.toolRegistry.clearToolLogs()
        self.llmClient.newTask()
        self.confirmedSectorAllocation = None
        self.confirmedPortfolioAllocation = None

        if config.boardroomPace == BoardroomPace.FAST:
            self.executeFastPortfolioRebalancing(config)
        else:
            self.executeCompletePortfolioRebalancing(config)
