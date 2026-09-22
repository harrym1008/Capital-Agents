import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from collectors.sector_dl_client import GICS_SECTORS
from llmtools.tool_registry import ToolRegistry
from llmtools.functions.confirmation import distributeIntegerPercentages
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import PortfolioCreationConfig, BoardroomPace
from boardroom.boardroom_engine import BoardroomEngine
from boardroom.boardroom_fsm import BoardroomContext, BoardroomStage, BoardroomFSM
from ui.ui_hooks import setCurrentStage, emitEvent, SimulationStoppedException


# PORTFOLIO CREATION stages for the BoardroomFSM

# Stage 0: Preset Sector Allocation Setup if chosen
class CreationPresetSectorSetupStage(BoardroomStage):
    def __init__(self):
        super().__init__(
            stageId="presetSectorSetup",
            phaseNumber=0,
            phaseName="Preset Sector Allocation Setup"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config

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
            keys = list(cleanedAllocations.keys())
            rawWeights = [cleanedAllocations[k]["allocationPct"] for k in keys]
            intAllocations = distributeIntegerPercentages(rawWeights, 100)
            for k, intVal in zip(keys, intAllocations):
                cleanedAllocations[k]["allocationPct"] = int(intVal)

        totalAllocated = sum(item["allocationPct"] for item in cleanedAllocations.values())
        confirmedAllocation = {
            "sectorAllocations": cleanedAllocations,
            "totalAllocatedPct": totalAllocated,
            "sectorCount": len(cleanedAllocations),
            "rationale": "Pre-set sector allocation configured by user."
        }
        engine.confirmedSectorAllocation = confirmedAllocation
        context.set("confirmedSectorAllocation", confirmedAllocation)

        confirmSectorTool = engine.toolRegistry.getTool("confirmSectorAllocation")
        if confirmSectorTool:
            confirmSectorTool.toolLog.append(confirmedAllocation)

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": confirmedAllocation
        })

        return None


# 1. Macro Analysis Stage
class CreationMacroStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 1):
        super().__init__(
            stageId="macroAnalysis",
            phaseNumber=phaseNumber,
            phaseName="Macro Environment Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()

        if pace == BoardroomPace.PRESET:
            confirmedSectorsText = engine.formatSectorsContext()
            cleanedAllocations = (engine.confirmedSectorAllocation or {}).get("sectorAllocations", {})
            activeSectorsSummary = ", ".join([f"{item['sector']} ({int(item['allocationPct'])}%)" for item in cleanedAllocations.values()])

            macroPrompt = (
                f"Task: Conduct top-down macroeconomic analysis to contextualise and guide portfolio equity selection.\n"
                f"Judge regimes on forward fundamentals, not yesterday's price noise.\n"
                f"Initial Capital: {promptArgs['initialCapital']}\n"
                f"Time Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
                f"MANDATED PRE-SET SECTOR ALLOCATION (EXECUTIVE DIRECTIVE):\n"
                f"{confirmedSectorsText}\n\n"
                f"CRITICAL DIRECTIVE ON SECTOR ALLOCATION:\n"
                f"The boardroom executive mandate has ALREADY established and locked the portfolio sector allocation to: {activeSectorsSummary}. "
                f"Do NOT propose an alternative sector allocation, and do NOT advise underweighting or avoiding the mandated sector(s). "
                f"Instead, analyse current macroeconomic indicators, market regime, inflation, yields, and sector rotation to assess "
                f"how the broader economic backdrop specifically impacts the mandated sector(s) ({activeSectorsSummary}), and evaluate which equity "
                f"attributes, fundamental factors, and risk exposures are best positioned to navigate prevailing market conditions within the mandated sector(s).\n\n"
                f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyse market regime and economic indicators.\n"
                f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
                f"3. Deliver strategic equity selection guidance tailored strictly to the mandated sectors: {activeSectorsSummary}."
            )
        else:
            macroPrompt = (
                f"Task: Conduct top-down macroeconomic analysis and examine macro sector rotations to guide portfolio asset creation.\n"
                f"Judge regimes on forward fundamentals, not yesterday's price noise.\n"
                f"Initial Capital: {promptArgs['initialCapital']}\n"
                f"Time Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
                f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyse market regime and leading/lagging sectors.\n"
                f"2. Output your economic indicator table, macro narrative, and market regime classification.\n"
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


# 2. Sector Allocation Debate Stage with Bull and Bear Analysts
class CreationSectorDebateStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 2):
        super().__init__(
            stageId="sectorDebate",
            phaseNumber=phaseNumber,
            phaseName="Sector Allocation Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")

        bullPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a growth and cyclical sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Champion durable compounding over fleeting momentum, judging each sector across upside, sideways and stress paths.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Assess market breadth (% > 50 SMA), "
            f"constituent divergence (median constituent return vs ETF return), "
            f"relative strength channels (1Y/3Y vs SPY), 10Y Treasury beta, and constituent FinBERT sentiment.\n"
            f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
            f"3. Highlight growth catalysts, healthy breadth participation, and upside drivers."
        )

        bearPrompt = (
            f"Macro Analysis Context:\n{macroRaw}\n\n"
            f"Task: Propose a defensive, risk-managed sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Demand evidence of resilience through the cycle rather than reacting to recent price softness alone.\n"
            f"Mandatory Constraints:\n"
            f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Scrutinise narrow rallies (where ETF return >> median constituent return), "
            f"deteriorating breadth (< 50% above 50 SMA), high channel valuation extremes, 10Y Treasury yield vulnerability, and bearish sentiment.\n"
            f"2. Present a clear table of percentage allocations across your selected sectors summing to 100.0%.\n"
            f"3. Highlight vulnerabilities, drawdown risks, and defensive hedges."
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


# 3. Sector Allocation Decision Stage with Portfolio Manager
class CreationSectorDecisionStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 3):
        super().__init__(
            stageId="sectorDecision",
            phaseNumber=phaseNumber,
            phaseName="Sector Allocation Decision"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")

        confirmSectorTool = engine.toolRegistry.getTool("confirmSectorAllocation")

        if pace == BoardroomPace.FAST:
            pmSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, determine the executive sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
                f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
                f"- Allocations must sum to approximately 100.0%.\n\n"
                f"1. You MUST first call 'fetchAllSectorsAnalysis' to evaluate all 11 GICS sectors. Review market breadth (% > 50 SMA), "
                f"constituent divergence (median constituent return vs ETF return), "
                f"relative strength channels (1Y/3Y vs SPY), 10Y Treasury beta, and constituent FinBERT sentiment.\n"
                f"2. Execute the 'confirmSectorAllocation' tool with your 'sectorAllocations' dictionary (e.g. "
                f"{{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) and executive 'rationale'."
            )
        else:
            bullSectorRaw = context.get("bullSectorRaw", "")
            bearSectorRaw = context.get("bearSectorRaw", "")
            pmSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Bullish Sector Proposal:\n{bullSectorRaw}\n\n"
                f"Bearish Sector Proposal:\n{bearSectorRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, reconcile the Bullish and Bearish proposals "
                f"to make the definitive sector allocation decision for {promptArgs['initialCapital']}.\n"
                f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- {promptArgs['sectorDiversityRule']}\n"
                f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
                f"- Allocations must sum to approximately 100.0%.\n\n"
                f"1. You may call 'fetchAllSectorsAnalysis' to inspect or cross-verify the underlying breadth, "
                f"constituent divergence, and sentiment metrics cited by the Bull and Bear (returns instantly from cache).\n"
                f"2. Execute the 'confirmSectorAllocation' tool with your 'sectorAllocations' dictionary (e.g. "
                f"{{'information_technology': 35, 'financials': 25, 'health_care': 20, 'consumer_discretionary': 20}}) and executive 'rationale'."
            )

        pmSectorRaw, pmSectorUISummary = engine.executeMandatedToolStage(
            agent=engine.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=10,
            requireInitialTools=True
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            engine.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]
            context.set("confirmedSectorAllocation", engine.confirmedSectorAllocation)

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": engine.confirmedSectorAllocation
        })

        return None


# 4. Stock Scouting Stage with Growth and Value Analysts
class CreationStockScoutingStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 4):
        super().__init__(
            stageId="stockScouting",
            phaseNumber=phaseNumber,
            phaseName="Stock Scouting"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        macroRaw = context.get("macroRaw", "")
        confirmedSectorsText = engine.formatSectorsContext()

        if pace == BoardroomPace.PRESET:
            cleanedAllocations = (engine.confirmedSectorAllocation or {}).get("sectorAllocations", {})
            activeSectorsSummary = ", ".join([f"{item['sector']} ({int(item['allocationPct'])}%)" for item in cleanedAllocations.values()])
            growthPrompt = (
                f"MANDATED SECTOR ALLOCATIONS (STRICT & BINDING):\n{confirmedSectorsText}\n\n"
                f"Macro Context:\n{macroRaw}\n\n"
                f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities STRICTLY within the confirmed sectors above.\n"
                f"Favour forward earnings power and catalyst runway over a single soft quarter, weighing recovery and acceleration scenarios.\n"
                f"MANDATORY CONSTRAINTS:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- TARGET STOCK COUNT: The target total stock count across the entire portfolio is {promptArgs['targetStockCount']}. "
                f"Keep your shortlisted candidate count focused and calibrated so the boardroom does not exceed this count.\n"
                f"- STRICT SECTOR BOUNDARY: You MUST ONLY scout candidate equities belonging to the confirmed sectors ({activeSectorsSummary}). "
                f"Do NOT scout or propose equities from any other sectors, regardless of any general macro commentary.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"CRITICAL MULTI-STEP TOOL EXECUTION MANDATE (STRICTLY REQUIRED):\n"
                f"1. STEP 1 (Screening): Call 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
                f"2. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after obtaining sector candidates from 'fetchStocksInSector', "
                f"you MUST call 'fetchBatchStockOverviews' with a list of your shortlisted candidate tickers (e.g. top 5-10 stocks across sectors). "
                f"DO NOT finalise or output your candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
                f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, EV/EBITDA, "
                f"profitability margins, revenue/EPS growth, leverage, news sentiment, and company summary) "
                f"via 'fetchBatchStockOverviews' to assess business quality and upside catalysts.\n"
                f"3. STEP 3: Present your structured candidate table with tickers, company name, "
                f"industry, real quantitative valuation/growth metrics, momentum, and growth catalysts."
            )

            valuePrompt = (
                f"MANDATED SECTOR ALLOCATIONS (STRICT & BINDING):\n{confirmedSectorsText}\n\n"
                f"Macro Context:\n{macroRaw}\n\n"
                f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive and value equities STRICTLY within the confirmed sectors above.\n"
                f"Require proven solvency and margin of safety through the cycle, tolerating modest underperformance where the balance sheet endures.\n"
                f"MANDATORY CONSTRAINTS:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- TARGET STOCK COUNT: The target total stock count across the entire portfolio is {promptArgs['targetStockCount']}. "
                f"Keep your shortlisted candidate count focused and calibrated so the boardroom does not exceed this count.\n"
                f"- STRICT SECTOR BOUNDARY: You MUST ONLY scout candidate equities belonging to the confirmed sectors ({activeSectorsSummary}). "
                f"Do NOT scout or propose equities from any other sectors, regardless of any general macro commentary.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"CRITICAL MULTI-STEP TOOL EXECUTION MANDATE (STRICTLY REQUIRED):\n"
                f"1. STEP 1 (Screening): Call 'fetchStocksInSector' with style='defensive' for each confirmed sector to screen candidate equities.\n"
                f"2. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after obtaining sector candidates from 'fetchStocksInSector', "
                f"you MUST call 'fetchBatchStockOverviews' with a list of your shortlisted candidate tickers (e.g. top 5-10 stocks across sectors). "
                f"DO NOT finalise or output your candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
                f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, debt-to-equity, "
                f"current/quick ratios, profitability margins, dividend yield, and company summary) "
                f"via 'fetchBatchStockOverviews' to verify solvency and true margin of safety.\n"
                f"3. STEP 3: Present your structured candidate table with tickers, company name, "
                f"industry, real quantitative valuation/solvency metrics, and margin of safety rationale."
            )
        else:
            growthPrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"Task: As the Growth Stock Hunter, scout high-conviction growth and momentum equities across the confirmed sectors.\n"
                f"Favour forward earnings power and catalyst runway over a single soft quarter, weighing recovery and acceleration scenarios.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
                f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"CRITICAL MULTI-STEP TOOL EXECUTION MANDATE (STRICTLY REQUIRED):\n"
                f"1. STEP 1 (Screening): Call 'fetchStocksInSector' with style='growth' for each confirmed sector to screen candidate equities.\n"
                f"2. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after obtaining sector candidates from 'fetchStocksInSector', "
                f"you MUST call 'fetchBatchStockOverviews' with a list of your shortlisted candidate tickers (e.g. top 5-10 stocks across sectors). "
                f"DO NOT finalise or output your candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
                f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, EV/EBITDA, "
                f"profitability margins, revenue/EPS growth, leverage, news sentiment, and company summary) "
                f"via 'fetchBatchStockOverviews' to assess business quality and upside catalysts.\n"
                f"3. STEP 3: Present your structured candidate table with tickers, company name, "
                f"industry, real quantitative valuation/growth metrics, momentum, and growth catalysts."
            )

            valuePrompt = (
                f"Macro Context:\n{macroRaw}\n\n"
                f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"Task: As the Value/Defensive Stock Hunter, scout high-conviction defensive, dividend, and value equities across the confirmed sectors.\n"
                f"Require proven solvency and margin of safety through the cycle, tolerating modest underperformance where the balance sheet endures.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target stock count across entire portfolio: {promptArgs['targetStockCount']}. Keep candidate selections focused and calibrated to this target.\n"
                f"- Sector Boundary: Scout candidate equities ONLY within the confirmed sectors above.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"CRITICAL MULTI-STEP TOOL EXECUTION MANDATE (STRICTLY REQUIRED):\n"
                f"1. STEP 1 (Screening): Call 'fetchStocksInSector' with style='defensive' for each confirmed sector to screen candidate equities.\n"
                f"2. STEP 2 (MANDATORY IMMEDIATE TOOL CALL): Immediately after obtaining sector candidates from 'fetchStocksInSector', "
                f"you MUST call 'fetchBatchStockOverviews' with a list of your shortlisted candidate tickers (e.g. top 5-10 stocks across sectors). "
                f"DO NOT finalise or output your candidate table after Step 1 alone! 'fetchStocksInSector' only provides heuristic scores. "
                f"You MUST inspect real financial metrics (valuation multiples like P/E, P/B, debt-to-equity, "
                f"current/quick ratios, profitability margins, dividend yield, and company summary) "
                f"via 'fetchBatchStockOverviews' to verify solvency and true margin of safety.\n"
                f"3. STEP 3: Present your structured candidate table with tickers, company name, "
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


# 5. Stock Allocation Proposal Stage with Aggressive and Conservative Analysts
class CreationStockProposalStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 5):
        super().__init__(
            stageId="stockProposals",
            phaseNumber=phaseNumber,
            phaseName="Stock Allocation Proposals"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        growthRaw = context.get("growthRaw", "")
        valueRaw = context.get("valueRaw", "")
        confirmedSectorsText = engine.formatSectorsContext()

        if pace == BoardroomPace.PRESET:
            cleanedAllocations = (engine.confirmedSectorAllocation or {}).get("sectorAllocations", {})
            activeSectorsSummary = ", ".join([f"{item['sector']} ({int(item['allocationPct'])}%)" for item in cleanedAllocations.values()])
            aggProposalPrompt = (
                f"Confirmed Pre-Set Sector Allocations (MANDATORY & BINDING):\n{confirmedSectorsText}\n\n"
                f"Growth Candidates Scouted:\n{growthRaw}\n\n"
                f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
                f"Task: Construct your final aggressive individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
                f"Press for genuine upside optionality while respecting evidence; do not churn holdings on noise.\n"
                f"MANDATORY CONSTRAINTS:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- STRICT SECTOR ALIGNMENT: Group stocks strictly under each confirmed sector ({activeSectorsSummary}).\n"
                f"- Per-Sector Weighting: Inside each confirmed sector, propose high-beta growth stocks with "
                f"whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
                f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining catalysts and beta strategy.\n\n"
                f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, "
                f"Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
            )

            consProposalPrompt = (
                f"Confirmed Pre-Set Sector Allocations (MANDATORY & BINDING):\n{confirmedSectorsText}\n\n"
                f"Growth Candidates Scouted:\n{growthRaw}\n\n"
                f"Defensive Candidates Scouted:\n{valueRaw}\n\n"
                f"Task: Construct your final conservative individual stock allocation proposal for {promptArgs['initialCapital']}.\n"
                f"Protect compounding first: prefer seasoned cash generators unless a challenger clearly improves safety.\n"
                f"MANDATORY CONSTRAINTS:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- STRICT SECTOR ALIGNMENT: Group stocks strictly under each confirmed sector ({activeSectorsSummary}).\n"
                f"- Per-Sector Weighting: Inside each confirmed sector, propose defensive, "
                f"low-volatility equities with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
                f"- Stock Justifications: Include a concise 25-35 word rationale per stock explaining margin of safety and downside protection.\n\n"
                f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, "
                f"Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
            )
        else:
            aggProposalPrompt = (
                f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"Growth Candidates Scouted in Phase 4:\n{growthRaw}\n\n"
                f"Defensive Candidates Scouted in Phase 4:\n{valueRaw}\n\n"
                f"Task: Review and scrutinise BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from an aggressive, "
                f"high-upside risk perspective. Construct your final aggressive individual stock allocation proposal for {promptArgs['initialCapital']}.\n\n"
                f"Instructions:\n"
                f"1. Scrutinise candidates from both hunters, identifying high-beta opportunities with "
                f"strong upside momentum and earnings catalysts, while challenging overly sluggish picks.\n"
                f"2. Group your stock proposals strictly under each confirmed sector above, "
                f"assigning whole integer 'perSectorWeight' percentages summing strictly to 100% per sector.\n"
                f"3. Provide a concise 25-35 word rationale per stock explaining catalysts and beta strategy.\n"
                f"4. DO NOT ask challenge questions or create Q&A items. Output your comprehensive analysis and structured allocation proposal.\n\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- Sector Alignment: Group your stock proposals strictly under each confirmed sector above.\n"
                f"- Per-Sector Weighting: Inside each confirmed sector, propose high-beta growth stocks with "
                f"whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, "
                f"Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
            )

            consProposalPrompt = (
                f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"Growth Candidates Scouted in Phase 4:\n{growthRaw}\n\n"
                f"Defensive Candidates Scouted in Phase 4:\n{valueRaw}\n\n"
                f"Task: Review and scrutinise BOTH the Growth Hunter candidates and Defensive/Value Hunter candidates from a conservative, "
                f"risk-managed capital-preservation perspective. Construct your final conservative individual stock allocation proposal for {promptArgs['initialCapital']}.\n\n"
                f"Instructions:\n"
                f"1. Scrutinise candidates from both hunters, prioritising rock-solid balance sheets, dividend reliability, "
                f"low volatility, and margin of safety, while challenging speculative or overvalued picks.\n"
                f"2. Group your stock proposals strictly under each confirmed sector above, "
                f"assigning whole integer 'perSectorWeight' percentages summing strictly to 100% per sector.\n"
                f"3. Provide a concise 25-35 word rationale per stock explaining margin of safety and downside protection.\n"
                f"4. DO NOT ask challenge questions or create Q&A items. Output your comprehensive analysis and structured allocation proposal.\n\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target Stock Count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- Sector Alignment: Group your stock proposals strictly under each confirmed sector above.\n"
                f"- Per-Sector Weighting: Inside each confirmed sector, propose defensive, "
                f"low-volatility equities with whole integer 'perSectorWeight' percentages summing strictly to 100% for that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n\n"
                f"Propose a comprehensive portfolio table grouped by confirmed sector (Stock, Sector, "
                f"Per-Sector Weight %, Dollar Allocation, Investment Role, and 25-35 word Rationale)."
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


# 6. Final Portfolio Decision Stage with Portfolio Manager
class CreationFinalDecisionStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 6):
        super().__init__(
            stageId="finalDecision",
            phaseNumber=phaseNumber,
            phaseName="Final Executive Decision"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "PortfolioCreationBoardroomEngine" = context.engine
        config: PortfolioCreationConfig = context.config
        pace = config.boardroomPace
        promptArgs = config.getPromptArgs()
        confirmedSectorsText = engine.formatSectorsContext()
        confirmPortTool = engine.toolRegistry.getTool("confirmPortfolioAllocation")

        if pace == BoardroomPace.FAST:
            growthRaw = context.get("growthRaw", "")
            valueRaw = context.get("valueRaw", "")
            pmFinalPrompt = (
                f"Initial Capital: {promptArgs['initialCapital']}\n"
                f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
                f"Growth Stock Hunter Scouting:\n{growthRaw}\n\n"
                f"Value/Defensive Stock Hunter Scouting:\n{valueRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, make the final executive decision to construct the portfolio for {promptArgs['initialCapital']}.\n"
                f"Balance conviction with humility: size positions for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary. Every confirmed sector must contain stock holdings.\n"
                f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. "
                f"60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
                f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
                f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
                f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed "
                f"sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
            )
        elif pace == BoardroomPace.PRESET:
            aggProposalRaw = context.get("aggProposalRaw", "")
            consProposalRaw = context.get("consProposalRaw", "")
            cleanedAllocations = (engine.confirmedSectorAllocation or {}).get("sectorAllocations", {})
            activeSectorsSummary = ", ".join([f"{item['sector']} ({int(item['allocationPct'])}%)" for item in cleanedAllocations.values()])
            pmFinalPrompt = (
                f"Initial Capital: {promptArgs['initialCapital']}\n"
                f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
                f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
                f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, reconcile the Aggressive and Conservative proposals "
                f"to make the definitive portfolio construction decision for {promptArgs['initialCapital']}.\n"
                f"Balance conviction with humility: size positions for bull, base and bear paths rather than a single forecast.\n"
                f"MANDATORY CONSTRAINTS:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' "
                f"dictionary ({activeSectorsSummary}). Every confirmed sector must contain stock holdings.\n"
                f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. "
                f"60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
                f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
                f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
                f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed "
                f"sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
            )
        else:
            aggProposalRaw = context.get("aggProposalRaw", "")
            consProposalRaw = context.get("consProposalRaw", "")
            pmFinalPrompt = (
                f"Initial Capital: {promptArgs['initialCapital']}\n"
                f"Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"Confirmed Sector Allocations:\n{confirmedSectorsText}\n\n"
                f"Aggressive Risk Allocation Proposal:\n{aggProposalRaw}\n\n"
                f"Conservative Risk Allocation Proposal:\n{consProposalRaw}\n\n"
                f"Task: As the Impartial Portfolio Manager, reconcile the Aggressive and Conservative proposals "
                f"to make the definitive portfolio construction decision for {promptArgs['initialCapital']}.\n"
                f"Balance conviction with humility: size positions for bull, base and bear paths rather than a single forecast.\n"
                f"Mandatory Constraints:\n"
                f"- Target Investment Horizon: {promptArgs['timeHorizon']}\n"
                f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
                f"- Target stock count: Aim for around {promptArgs['targetStockCount']} stocks across the confirmed sectors.\n"
                f"- Sector Allocation Structure: Group stocks by confirmed sector into the 'sectorAllocations' dictionary. Every confirmed sector must contain stock holdings.\n"
                f"- Per-Sector Weighting: Inside each sector, assign whole integer 'perSectorWeight' percentages (e.g. "
                f"60, 40, not decimals) to chosen stocks such that they strictly sum to 100% of that sector.\n"
                f"- Max single stock allocation: {promptArgs['maxStockAllocation']}.\n"
                f"- Stock Justifications: Provide a 25-35 word rationale for each equity holding.\n"
                f"- Portfolio Rationale: Provide an executive portfolioRationale of approximately 100 words.\n\n"
                f"Execute the 'confirmPortfolioAllocation' tool with your 'sectorAllocations' dictionary (mapping each confirmed "
                f"sector to its list of stocks with 'ticker', 'perSectorWeight', and 'rationale') and 'portfolioRationale'."
            )

        pmFinalRaw, pmFinalUISummary = engine.executeMandatedToolStage(
            agent=engine.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=10,
            requireInitialTools=True
        )

        if confirmPortTool and confirmPortTool.toolLog:
            engine.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]
            context.set("confirmedPortfolioAllocation", engine.confirmedPortfolioAllocation)

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": engine.confirmedPortfolioAllocation
        })

        return None



# The main complete engine
class PortfolioCreationBoardroomEngine(BoardroomEngine):
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
            agentRole="Value/Defensive Stock Hunter",
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
                {"role": "Value/Defensive Stock Hunter", "color": self.valueHunter.color, "name": "Defensive Stock Hunter"}
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


    # Build the Finite State Machine for the portfolio creation process 
    def buildFSM(self, config: PortfolioCreationConfig) -> BoardroomFSM:
        pace = config.boardroomPace

        if pace == BoardroomPace.FAST:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title="Fast Portfolio Creation Boardroom (Pace: FAST)")
            fsm.addStage(CreationMacroStage(phaseNumber=1), nextStage="sectorDecision")
            fsm.addStage(CreationSectorDecisionStage(phaseNumber=3), nextStage="stockScouting")
            fsm.addStage(CreationStockScoutingStage(phaseNumber=4), nextStage="finalDecision")
            fsm.addStage(CreationFinalDecisionStage(phaseNumber=6), nextStage=None)
            return fsm
        elif pace == BoardroomPace.PRESET:
            fsm = BoardroomFSM(initialStageId="presetSectorSetup", title="Pre-set Sector Allocation Portfolio Creation Boardroom (Pace: PRESET)")
            fsm.addStage(CreationPresetSectorSetupStage(), nextStage="macroAnalysis")
            fsm.addStage(CreationMacroStage(phaseNumber=1), nextStage="stockScouting")
            fsm.addStage(CreationStockScoutingStage(phaseNumber=4), nextStage="stockProposals")
            fsm.addStage(CreationStockProposalStage(phaseNumber=5), nextStage="finalDecision")
            fsm.addStage(CreationFinalDecisionStage(phaseNumber=6), nextStage=None)
            return fsm
        else:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title="Complete Portfolio Creation Boardroom (Pace: COMPLETE)")
            fsm.addStage(CreationMacroStage(phaseNumber=1), nextStage="sectorDebate")
            fsm.addStage(CreationSectorDebateStage(phaseNumber=2), nextStage="sectorDecision")
            fsm.addStage(CreationSectorDecisionStage(phaseNumber=3), nextStage="stockScouting")
            fsm.addStage(CreationStockScoutingStage(phaseNumber=4), nextStage="stockProposals")
            fsm.addStage(CreationStockProposalStage(phaseNumber=5), nextStage="finalDecision")
            fsm.addStage(CreationFinalDecisionStage(phaseNumber=6), nextStage=None)
            return fsm


    def execute(self, config: PortfolioCreationConfig) -> None:
        self.toolRegistry.clearToolLogs()
        self.llmClient.newTask()
        self.confirmedSectorAllocation = None
        self.confirmedPortfolioAllocation = None
        self.lastConfig = config

        context = BoardroomContext(engine=self, config=config)
        fsm = self.buildFSM(config)
        fsm.run(context)
