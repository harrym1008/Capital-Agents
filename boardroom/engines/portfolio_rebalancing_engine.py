import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import PortfolioRebalancingConfig, BoardroomPace
from boardroom.boardroom_engine import BoardroomEngine
from boardroom.boardroom_fsm import BoardroomContext, BoardroomStage, BoardroomFSM
from ui.ui_hooks import setCurrentStage, emitEvent, SimulationStoppedException


# PORTFOLIO REBALANCING stages for the FSM

# Stage 1: Macro Environment Analysis
class RebalanceMacroStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 1):
        super().__init__(
            stageId="macroAnalysis",
            phaseNumber=phaseNumber,
            phaseName="Macro Environment Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        promptArgs = config.getPromptArgs()

        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis and audit the existing baseline portfolio against prevailing market conditions.\n"
            f"Judge holdings on forward regime exposure rather than punishing brief price softness.\n"
            f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
            f"Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n\n"
            f"CURRENT BASELINE PORTFOLIO HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTOR BREAKDOWN:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyse market regime, inflation, rates, and leading/lagging sectors.\n"
            f"2. Audit the current portfolio: Assess which sectors are overconcentrated or facing headwinds, "
            f"and identify which emerging or resilient sectors are underrepresented.\n"
            f"3. Deliver your macro narrative, economic indicator table, market regime classification, "
            f"and portfolio vulnerability diagnosis calibrated to the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )

        macroRaw, macroUISummary = engine.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=engine.toolRegistry,
            timestamp=engine.timestamp,
            config=config,
            requireInitialTools=True
        )

        context.set("macroRaw", macroRaw)
        context.set("macroUISummary", macroUISummary)
        return None

# Stage 2: Sector Rebalancing Debate
class RebalanceSectorDebateStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 2):
        super().__init__(
            stageId="sectorDebate",
            phaseNumber=phaseNumber,
            phaseName="Sector Rebalancing Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")

        bullPrompt = (
            f"Macro & Portfolio Audit Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"PRECALCULATED BASELINE SECTORS:\n{promptArgs['precalculatedSectorsList']}\n\n"
            f"Task: Propose an opportunistic, growth-oriented sector rebalancing allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Weigh whether incumbents can recover and compound before urging rotation on recent softness.\n"
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
            f"Distinguish temporary price weakness from genuine solvency deterioration before urging exits.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Scrutinise narrow rallies and deteriorating breadth.\n"
            f"2. Propose defensive trims or exits from overextended or vulnerable sectors in the current portfolio under the rebalance mandate.\n"
            f"3. Present a clear table of percentage allocations across your selected sectors summing to 100.0%."
        )

        (bullSectorRaw, bullSectorUISummary), (bearSectorRaw, bearSectorUISummary) = engine.runAgentsConcurrently(
            lambda: engine.bullAnalyst.analyseAndReply(
                incomingMessage=bullPrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: engine.bearAnalyst.analyseAndReply(
                incomingMessage=bearPrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        context.set("bullSectorRaw", bullSectorRaw)
        context.set("bullSectorUISummary", bullSectorUISummary)
        context.set("bearSectorRaw", bearSectorRaw)
        context.set("bearSectorUISummary", bearSectorUISummary)
        return None

# Stage 3: Sector Rebalancing Decision
class RebalanceSectorDecisionStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 3):
        super().__init__(
            stageId="sectorDecision",
            phaseNumber=phaseNumber,
            phaseName="Sector Rebalancing Decision"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")
        confirmSectorTool = engine.toolRegistry.getTool("confirmSectorAllocation")

        if pace == BoardroomPace.FAST:
            pmSectorPrompt = (
                f"Macro & Portfolio Audit Context:\n{macroRaw}\n\n"
                f"CURRENT BASELINE PORTFOLIO HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
                f"PRECALCULATED BASELINE SECTOR BREAKDOWN:\n{promptArgs['precalculatedSectorsList']}\n\n"
                f"Task: As the Impartial Portfolio Manager, determine the target rebalanced sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
                f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Allocations must sum to approximately 100.0% across equity sectors.\n\n"
                f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. "
                f"Review market breadth, constituent divergence, relative strength channels, and FinBERT sentiment.\n"
                f"2. Compare your target sector weights directly against the baseline sector breakdown. You MUST explain the rationale for each difference (e.g. "
                f"why specific sectors were expanded, trimmed, eliminated, or newly introduced) in accordance with the Rebalance Mandate.\n"
                f"3. Execute the 'confirmSectorAllocation' tool with your target 'sectorAllocations' dictionary (e.g. "
                f"{{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) "
                f"and executive 'rationale' detailing these sector-level shifts."
            )
        else:
            bullSectorRaw = context.get("bullSectorRaw", "")
            bearSectorRaw = context.get("bearSectorRaw", "")
            pmSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
                f"PRECALCULATED BASELINE SECTORS:\n{promptArgs['precalculatedSectorsList']}\n\n"
                f"Bullish Sector Proposal:\n{bullSectorRaw}\n\n"
                f"Bearish Sector Proposal:\n{bearSectorRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, reconcile the Bullish and Bearish proposals to "
                f"make the definitive target sector allocation decision for {promptArgs['initialCapital']}.\n"
                f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Allocations must sum to approximately 100.0% across equity sectors.\n\n"
                f"1. You may call 'fetchAllSectorsAnalysis' to verify metrics.\n"
                f"2. Compare your target sector weights directly against the baseline sector breakdown. You MUST explain the rationale for each difference (e.g. "
                f"why specific sectors were expanded, trimmed, eliminated, or newly introduced) in accordance with the Rebalance Mandate.\n"
                f"3. Execute the 'confirmSectorAllocation' tool with your target 'sectorAllocations' dictionary and executive 'rationale' detailing these sector-level shifts."
            )

        pmSectorConfirmationPrompt = (
            f"Your sector allocation has been verified and logged.\n"
            f"Please now provide your detailed executive remarks explaining the sector changes you are making compared to the original baseline portfolio.\n"
            f"Specifically reference the original baseline sector breakdown, detail each sector difference/delta (weights increased, decreased, eliminated, or newly added), "
            f"and explain the economic and risk rationale behind each difference in alignment with the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )

        pmSectorRaw, pmSectorUISummary = engine.executeMandatedToolStage(
            agent=engine.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=10,
            requireInitialTools=True,
            confirmationPrompt=pmSectorConfirmationPrompt
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            engine.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]
            context.set("confirmedSectorAllocation", engine.confirmedSectorAllocation)

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": engine.confirmedSectorAllocation
        })

        return None


# Stage 4: Stock Holdings Audit and Candidate Screening
class RebalanceStockAuditStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 4):
        super().__init__(
            stageId="stockAudit",
            phaseNumber=phaseNumber,
            phaseName="Stock Holdings Audit"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")
        confirmedSectorsText = engine.formatSectorsContext()

        growthPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, evaluate existing growth holdings and scout "
            f"high-conviction momentum/growth replacements strictly within the target sectors above.\n"
            f"Think as a growth owner: do not recommend replacing an incumbent on a modest dip alone; require a clearly superior forward path.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"CRITICAL MULTI-STEP EXECUTION MANDATE (STRICTLY REQUIRED):\n"
            f"1. Audit existing baseline growth holdings: recommend which to KEEP vs. REPLACE or TRIM based on current growth catalysts and the rebalance mandate.\n"
            f"2. STEP 1 (Screening): Call 'fetchStocksInSector' with style='growth' for confirmed sectors to screen replacement or expansion candidate equities.\n"
            f"3. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after calling 'fetchStocksInSector', you MUST call 'fetchBatchStockOverviews' "
            f"with a list of your shortlisted candidate tickers. DO NOT finalise or output your "
            f"candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
            f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, EV/EBITDA, "
            f"profitability margins, revenue/EPS growth, leverage, news sentiment, and company summary) "
            f"via 'fetchBatchStockOverviews' before making any final recommendations.\n"
            f"PORTFOLIO CONTINUITY MANDATE: Your shortlisted candidate tickers fed into 'fetchBatchStockOverviews' MUST include ALL "
            f"of the companies that exist right now inside CURRENT BASELINE HOLDINGS above. Extract every ticker dynamically "
            f"from that list at run time and combine those incumbents with newly screened candidates, "
            f"so each current holding is re-valued on real metrics before any KEEP, TRIM or REPLACE verdict.\n"
            f"4. STEP 3: Present your structured candidate table with tickers, company name, industry, real quantitative valuation/growth metrics, momentum, and growth catalysts."
        )

        valuePrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Defensive Stock Hunter, evaluate existing defensive/value holdings and "
            f"scout high-conviction defensive replacements strictly within the target sectors above.\n"
            f"Think as a defensive steward: retain sound incumbents through soft patches unless overview metrics show the cushion is gone.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}.\n"
            f"- Sector Boundary: Focus strictly on confirmed target sectors.\n\n"
            f"CRITICAL MULTI-STEP EXECUTION MANDATE (STRICTLY REQUIRED):\n"
            f"1. Audit existing baseline defensive/value holdings: recommend which to KEEP vs. REPLACE or TRIM based on margin of safety and the rebalance mandate.\n"
            f"2. STEP 1 (Screening): Call 'fetchStocksInSector' with style='defensive' for confirmed sectors to screen candidate equities.\n"
            f"3. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after calling 'fetchStocksInSector', you MUST call 'fetchBatchStockOverviews' "
            f"with a list of your shortlisted candidate tickers. DO NOT finalise or output your "
            f"candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
            f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, debt-to-equity, "
            f"current/quick ratios, profitability margins, dividend yield, and company summary) "
            f"via 'fetchBatchStockOverviews' to verify solvency and true margin of safety.\n"
            f"PORTFOLIO CONTINUITY MANDATE: Your shortlisted candidate tickers fed into 'fetchBatchStockOverviews' MUST include ALL "
            f"of the companies that exist right now inside CURRENT BASELINE HOLDINGS above. Extract every ticker dynamically "
            f"from that list at run time and combine those incumbents with newly screened candidates, "
            f"so each current holding is re-valued on real metrics before any KEEP, TRIM or REPLACE verdict.\n"
            f"4. STEP 3: Present your structured candidate table with tickers, company name, "
            f"industry, real quantitative valuation/solvency metrics, and margin of safety rationale."
        )

        (growthRaw, growthUISummary), (valueRaw, valueUISummary) = engine.runAgentsConcurrently(
            lambda: engine.growthHunter.analyseAndReply(
                incomingMessage=growthPrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=True
            ),
            lambda: engine.valueHunter.analyseAndReply(
                incomingMessage=valuePrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=True
            )
        )

        context.set("growthRaw", growthRaw)
        context.set("growthUISummary", growthUISummary)
        context.set("valueRaw", valueRaw)
        context.set("valueUISummary", valueUISummary)
        return None


# Stage 5: Stock Reallocation Proposals
class RebalanceStockProposalStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 5):
        super().__init__(
            stageId="stockProposals",
            phaseNumber=phaseNumber,
            phaseName="Stock Reallocation Proposals"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        promptArgs = config.getPromptArgs()
        growthRaw = context.get("growthRaw", "")
        valueRaw = context.get("valueRaw", "")
        confirmedSectorsText = engine.formatSectorsContext()

        aggProposalPrompt = (
            f"Target Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
            f"Growth Candidates Scouted in Phase 4:\n{growthRaw}\n\n"
            f"Defensive Candidates Scouted in Phase 4:\n{valueRaw}\n\n"
            f"Task: Review and scrutinise BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from an aggressive, "
            f"high-upside risk perspective against the CURRENT BASELINE HOLDINGS. "
            f"Construct your final aggressive stock rebalancing proposal for {promptArgs['initialCapital']}.\n"
            f"Seek alpha through disciplined upgrades, not restless trading; model recovery as well as momentum for each incumbent.\n\n"
            f"Instructions:\n"
            f"1. Evaluate candidate equities from both hunters against baseline positions: identify which baseline stocks to retain or "
            f"expand for upside momentum, and which to replace with high-beta scouted picks while respecting the Rebalance Mandate.\n"
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
            f"Task: Review and scrutinise BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from a conservative, risk-managed capital-preservation "
            f"perspective against the CURRENT BASELINE HOLDINGS. Construct your final conservative stock rebalancing proposal for {promptArgs['initialCapital']}.\n"
            f"Default to keeping proven incumbents; demand clear evidence of impaired safety before endorsing turnover.\n\n"
            f"Instructions:\n"
            f"1. Evaluate candidate equities from both hunters against baseline positions: identify core baseline holdings to preserve for safety and solvency, "
            f"and determine which scouted defensive picks offer superior margin of safety while respecting the Rebalance Mandate.\n"
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

        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = engine.runAgentsConcurrently(
            lambda: engine.aggRiskAnalyst.analyseAndReply(
                incomingMessage=aggProposalPrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=False
            ),
            lambda: engine.consRiskAnalyst.analyseAndReply(
                incomingMessage=consProposalPrompt,
                toolRegistry=engine.toolRegistry,
                timestamp=engine.timestamp,
                config=config,
                requireInitialTools=False
            )
        )

        context.set("aggProposalRaw", aggProposalRaw)
        context.set("aggProposalUISummary", aggProposalUISummary)
        context.set("consProposalRaw", consProposalRaw)
        context.set("consProposalUISummary", consProposalUISummary)
        return None


# Stage 6: Final Portfolio Rebalancing Decision
class RebalanceFinalDecisionStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 6):
        super().__init__(
            stageId="finalDecision",
            phaseNumber=phaseNumber,
            phaseName="Final Executive Decision"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioRebalancingBoardroomEngine" = context.engine
        config: PortfolioRebalancingConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        confirmedSectorsText = engine.formatSectorsContext()
        confirmPortTool = engine.toolRegistry.getTool("confirmPortfolioAllocation")

        if confirmPortTool:
            origConfirmPortFunc = confirmPortTool.function
            def boundConfirmPortfolioAllocation(*args, **kwargs):
                if "initialCapital" not in kwargs or not kwargs["initialCapital"]:
                    kwargs["initialCapital"] = config.initialCapital
                return origConfirmPortFunc(*args, **kwargs)
            confirmPortTool.function = boundConfirmPortfolioAllocation

        if pace == BoardroomPace.FAST:
            growthRaw = context.get("growthRaw", "")
            valueRaw = context.get("valueRaw", "")
            pmFinalPrompt = (
                f"Total Portfolio Capital: {promptArgs['initialCapital']}\n"
                f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
                f"Target Sector Allocations:\n{confirmedSectorsText}\n\n"
                f"CURRENT BASELINE HOLDINGS:\n{promptArgs['currentHoldingsList']}\n\n"
                f"Growth Stock Hunter Scouting:\n{growthRaw}\n\n"
                f"Defensive Stock Hunter Scouting:\n{valueRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, construct the definitive rebalanced portfolio for {promptArgs['initialCapital']}.\n"
                f"Minimise needless turnover: retain incumbents whose overview metrics remain sound, justifying every exit against the Rebalance Mandate.\n"
                f"You may retain strong existing holdings from the baseline portfolio, adjust their weights, replace underperforming names, "
                f"and introduce high-conviction new additions in accordance with the Rebalance Mandate ({promptArgs['rebalanceAmount']}).\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
                f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across confirmed sectors.\n"
                f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary.\n"
                f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages summing strictly to 100% of that sector.\n"
                f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
                f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100-150 words detailing the rebalancing strategy, "
                f"explicitly comparing the rebalanced portfolio to the original baseline holdings and explaining the differences.\n\n"
                f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary, 'portfolioRationale', and 'initialCapital'={config.initialCapital}."
            )
        else:
            aggProposalRaw = context.get("aggProposalRaw", "")
            consProposalRaw = context.get("consProposalRaw", "")
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
                f"Minimise needless turnover: retain incumbents whose overview metrics remain sound, justifying every exit against the Rebalance Mandate.\n"
                f"You may retain strong existing holdings from the baseline portfolio, adjust their weights, replace underperforming names, "
                f"and introduce high-conviction new additions in accordance with the Rebalance Mandate ({promptArgs['rebalanceAmount']}).\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Rebalance Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
                f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across confirmed sectors.\n"
                f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary.\n"
                f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages summing strictly to 100% of that sector.\n"
                f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
                f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100-150 words detailing the rebalancing strategy, "
                f"explicitly comparing the rebalanced portfolio to the original baseline holdings and explaining the differences.\n\n"
                f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary, 'portfolioRationale', and 'initialCapital'={config.initialCapital}."
            )

        pmFinalConfirmationPrompt = (
            f"Your rebalanced portfolio allocation has been verified, logged, and confirmed.\n"
            f"Please now provide your definitive executive report explaining the changes you have made to the portfolio.\n"
            f"Explicitly reference the original baseline portfolio holdings: identify which original stocks were retained (and whether their allocation was weighted up or down), "
            f"which original stocks were trimmed or completely exited, which new stocks were added, "
            f"and clearly explain the strategic rationale behind all differences in light of the Rebalance Mandate ({promptArgs['rebalanceAmount']})."
        )

        pmFinalRaw, pmFinalUISummary = engine.executeMandatedToolStage(
            agent=engine.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=10,
            requireInitialTools=True,
            confirmationPrompt=pmFinalConfirmationPrompt
        )

        if confirmPortTool and confirmPortTool.toolLog:
            engine.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]
            context.set("confirmedPortfolioAllocation", engine.confirmedPortfolioAllocation)

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": engine.confirmedPortfolioAllocation,
            "originalPositions": config.currentPositions
        })

        return None


# The main complete engine
class PortfolioRebalancingBoardroomEngine(BoardroomEngine):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        super().__init__(toolRegistry, timestamp)
        self.confirmedSectorAllocation: Optional[Dict[str, Any]] = None
        self.confirmedPortfolioAllocation: Optional[Dict[str, Any]] = None

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
            color="cyan",
            dateStr=timestampStr
        )

        self.bullAnalyst = FinancialAgent(
            agentRole="Bullish Value Analyst",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                toolMap["executePythonCalculation"]
            ],
            color="green",
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                toolMap["executePythonCalculation"]
            ],
            color="red",
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
            color="magenta",
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
                toolMap["fetchShortInterestHistory"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="green",
            dateStr=timestampStr
        )

        self.valueHunter = FinancialAgent(
            agentRole="Defensive Stock Hunter",
            tools=[
                toolMap["fetchStocksInSector"],
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchShortInterestHistory"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="blue",
            dateStr=timestampStr
        )

        self.aggRiskAnalyst = FinancialAgent(
            agentRole="Aggressive Risk Analyst",
            tools=[
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchShortInterestHistory"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="yellow",
            dateStr=timestampStr
        )

        self.consRiskAnalyst = FinancialAgent(
            agentRole="Conservative Risk Analyst",
            tools=[
                toolMap["fetchBatchStockOverviews"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchShortInterestHistory"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="blue",
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

    def getDefaultPhaseAgents(self, phaseNumber: int, pace: BoardroomPace) -> List[Dict[str, str]]:
        if phaseNumber == 1:
            return [{"role": "Macro Analyst", "color": self.macroAnalyst.color, "name": "Macro Strategist"}]
        elif phaseNumber == 2:
            return [
                {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
            ]
        elif phaseNumber == 3:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
        elif phaseNumber == 4:
            return [
                {"role": "Growth Stock Hunter", "color": self.growthHunter.color, "name": "Growth Stock Hunter"},
                {"role": "Defensive Stock Hunter", "color": self.valueHunter.color, "name": "Defensive Stock Hunter"}
            ]
        elif phaseNumber == 5:
            return [
                {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.color, "name": "Aggressive Risk Analyst"},
                {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.color, "name": "Conservative Risk Analyst"}
            ]
        elif phaseNumber == 6:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]

        return []

    def formatSectorsContext(self) -> str:
        # Return the formatted confirmed sector allocation as a string for LLM prompts
        if not self.confirmedSectorAllocation:
            return "No confirmed sector allocation available."
        confirmed = self.confirmedSectorAllocation.get("confirmedAllocation", self.confirmedSectorAllocation)
        sectorsDict = confirmed.get("sectorAllocations", {})
        lines = [f"- {secInfo.get('sector', secKey)} ({secKey}): {int(round(float(secInfo.get('allocationPct', 0))))}%" for secKey, secInfo in sectorsDict.items()]
        return "\n".join(lines)

    # Build the FSM for this rebalancing engine
    def buildFSM(self, config: PortfolioRebalancingConfig) -> BoardroomFSM:
        pace = config.boardroomPace

        if pace == BoardroomPace.FAST:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title="Fast Portfolio Rebalancing Boardroom (Pace: FAST)")
            fsm.addStage(RebalanceMacroStage(phaseNumber=1), nextStage="sectorDecision")
            fsm.addStage(RebalanceSectorDecisionStage(phaseNumber=3), nextStage="stockAudit")
            fsm.addStage(RebalanceStockAuditStage(phaseNumber=4), nextStage="finalDecision")
            fsm.addStage(RebalanceFinalDecisionStage(phaseNumber=6), nextStage=None)
            return fsm
        else:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title="Complete Portfolio Rebalancing Boardroom (Pace: COMPLETE)")
            fsm.addStage(RebalanceMacroStage(phaseNumber=1), nextStage="sectorDebate")
            fsm.addStage(RebalanceSectorDebateStage(phaseNumber=2), nextStage="sectorDecision")
            fsm.addStage(RebalanceSectorDecisionStage(phaseNumber=3), nextStage="stockAudit")
            fsm.addStage(RebalanceStockAuditStage(phaseNumber=4), nextStage="stockProposals")
            fsm.addStage(RebalanceStockProposalStage(phaseNumber=5), nextStage="finalDecision")
            fsm.addStage(RebalanceFinalDecisionStage(phaseNumber=6), nextStage=None)
            return fsm


    def execute(self, config: PortfolioRebalancingConfig) -> None:
        self.toolRegistry.clearToolLogs()
        self.llmClient.newTask()
        self.confirmedSectorAllocation = None
        self.confirmedPortfolioAllocation = None
        self.lastConfig = config

        context = BoardroomContext(engine=self, config=config)
        fsm = self.buildFSM(config)
        fsm.run(context)
