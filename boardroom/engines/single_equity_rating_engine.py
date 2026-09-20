import re
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent
from llm.agents.agent_prompts import buildSpokespersonSysPrompt, buildSpecialistQnASysPrompt
from boardroom.boardroom_config import SingleEquityRatingConfig, SingleEquityTimeHorizon, BoardroomPace
from boardroom.boardroom_engine import BoardroomEngine
from boardroom.boardroom_fsm import BoardroomContext, BoardroomStage, BoardroomFSM
from ui.ui_hooks import isStopRequested, setCurrentStage, setCurrentAgent, setAgentPhase, emitEvent, SimulationStoppedException


def getActiveSpecialistRoles(pace: BoardroomPace) -> List[str]:
    if pace == BoardroomPace.ONE_SHOT:
        return ["One-Shot Analyst"]
    elif pace == BoardroomPace.FAST:
        return [
            "Macro Analyst",
            "Bullish Value Analyst",
            "Bearish Risk Analyst",
            "Impartial Portfolio Manager"
        ]
    else:
        return [
            "Macro Analyst",
            "Bullish Value Analyst",
            "Bearish Risk Analyst",
            "Aggressive Risk Analyst",
            "Conservative Risk Analyst",
            "Impartial Portfolio Manager"
        ]


# SINGLE EQUITY BOARDROOM stages for its FSM

# Stage 1 only for One-Shot 
class SingleEquityOneShotAnalysisStage(BoardroomStage):
    def __init__(self):
        super().__init__(
            stageId="oneShotAnalysis",
            phaseNumber=1,
            phaseName="One-Shot Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()

        oneShotPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Time Horizon: {timeHorizonInfo['label']}\n\n"
            f"Task: Conduct your complete analysis of macro conditions, single-stock reserach, risk assessment and final "
            f"executive decision in one go for the ticker: {targetTicker} over the {timeHorizonInfo['label']} time horizon.\n"
            f"Execute your data tools (macro, financials, valuation, statements, stock performance, news) to retrieve hard facts. "
            f"Present your final executive decision with explicit rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}."
        )

        oneShotRaw, _ = engine.oneShotAnalyst.analyseAndReply(
            oneShotPrompt, engine.toolRegistry, engine.timestamp, config, subrole="analysis", requireInitialTools=True
        )
        context.set("oneShotRaw", oneShotRaw)
        return None


# One shot analysis upload
class SingleEquityOneShotUploadStage(BoardroomStage):
    def __init__(self):
        super().__init__(
            stageId="oneShotUpload",
            phaseNumber=2,
            phaseName="Decision Upload"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]
        oneShotRaw = context.get("oneShotRaw", "")

        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Decision:\n{oneShotRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )

        origTools = list(engine.oneShotAnalyst.tools)
        engine.oneShotAnalyst.clearTools()
        engine.oneShotAnalyst.addTool(finalSubmitToolName, engine.toolRegistry)
        _, _ = engine.executeMandatedToolStage(
            agent=engine.oneShotAnalyst,
            initialPrompt=uploadPrompt,
            mandatedToolName=finalSubmitToolName,
            config=config,
            subrole="upload",
            maxRetries=10,
            summarisationOverride=False,
            requireInitialTools=True
        )
        engine.oneShotAnalyst.tools = origTools

        try:
            formattedExecutiveDecision = engine.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."

        convSummary = (
            f"\n{engine.oneShotAnalyst.color}Final Executive Decision:\n{oneShotRaw}\n"
            f"\n{formattedExecutiveDecision}\n"
        )
        engine.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', convSummary)
        engine.lastConfig = config
        if formattedExecutiveDecision and formattedExecutiveDecision != "Decision not found.":
            engine.portManagerFinalOutput = f"{oneShotRaw}\n\n{formattedExecutiveDecision}"
        else:
            engine.portManagerFinalOutput = oneShotRaw

        return None


# 1. Macro Analysis Stage with Macro Analyst
class SingleEquityMacroStage(BoardroomStage):
    def __init__(self, phaseNumber: int = 1):
        super().__init__(
            stageId="macroAnalysis",
            phaseNumber=phaseNumber,
            phaseName="Macro Environment Analysis"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        timeHorizonInfo = config.getTimeHorizonInfo()

        macroPrompt = (
            f"Time Horizon: {timeHorizonInfo['label']}\n\n"
            f"Task: Conduct top-down macroeconomic analysis for the US financial markets over the {timeHorizonInfo['label']} time horizon.\n"
            f"Use your macro-specific tools to retrieve economic indicators, headlines, and sentiment history. "
            f"Present a narrative macro summary and explicitly output your overall market regime classification as BULLISH, BEARISH, or NEUTRAL."
        )
        macroRaw, macroUISummary = engine.macroAnalyst.analyseAndReply(
            macroPrompt, engine.toolRegistry, engine.timestamp, config, subrole=None, requireInitialTools=True
        )

        context.set("macroRaw", macroRaw)
        context.set("macroUISummary", macroUISummary)
        return None


# 2. Initial equity research stage with Bullish and Bearish Analysts
class SingleEquityResearchStage(BoardroomStage):
    def __init__(self, targetTicker: str, phaseNumber: int = 2):
        super().__init__(
            stageId="specialistResearch",
            phaseNumber=phaseNumber,
            phaseName=f"Specialist Research on {targetTicker}"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()
        macroRaw = context.get("macroRaw", "")

        researchPrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Time Horizon: {timeHorizonInfo['label']}\n\n"
            f"Task: Conduct single-stock research on ticker {targetTicker} over the {timeHorizonInfo['label']} time horizon.\n"
            f"Execute your data tools (valuation metrics, financial statements, stock price performance, company profile, etc.) to retrieve hard facts. "
            f"Present your thesis and state: explicit rating ({{permittedRatings}}), OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT weight, and preliminary {timeHorizonInfo['llmPriceTargets']}."
        )

        (bullThesisRaw, bullThesisUISummary), (bearThesisRaw, bearThesisUISummary) = engine.runAgentsConcurrently(
            lambda: engine.bullAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="BUY/HOLD"), engine.toolRegistry, engine.timestamp,
                config, subrole="research", requireInitialTools=True
            ),
            lambda: engine.bearAnalyst.analyseAndReply(
                researchPrompt.format(permittedRatings="HOLD/SELL"), engine.toolRegistry, engine.timestamp,
                config, subrole="research", requireInitialTools=True
            )
        )

        context.set("bullThesisRaw", bullThesisRaw)
        context.set("bullThesisUISummary", bullThesisUISummary)
        context.set("bearThesisRaw", bearThesisRaw)
        context.set("bearThesisUISummary", bearThesisUISummary)
        return None


# 3. Thesis critiquing stage with Aggressive and Conservative Risk Analysts
class SingleEquityDebateStage(BoardroomStage):
    def __init__(self, targetTicker: str, phaseNumber: int = 3):
        super().__init__(
            stageId="seniorRiskDebate",
            phaseNumber=phaseNumber,
            phaseName=f"Senior Risk Debate on {targetTicker}"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        macroRaw = context.get("macroRaw", "")
        bearThesisRaw = context.get("bearThesisRaw", "")
        bullThesisRaw = context.get("bullThesisRaw", "")

        aggDebatePrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Bearish Analyst's Thesis on {targetTicker}:\n{bearThesisRaw}\n\n"
            f"Task: Challenge the Bearish Analyst's stance on {targetTicker}.\n"
            f"Formulate 2-3 quantitative questions challenging their downside assumptions."
        )
        consDebatePrompt = (
            f"Macroeconomic Context:\n{macroRaw}\n\n"
            f"Bullish Analyst's Thesis on {targetTicker}:\n{bullThesisRaw}\n\n"
            f"Task: Challenge the Bullish Analyst's stance on {targetTicker}.\n"
            f"Formulate 2-3 quantitative questions challenging their upside assumptions."
        )

        (aggQuestionsRaw, aggQuestionsUISummary), (consQuestionsRaw, consQuestionsUISummary) = engine.runAgentsConcurrently(
            lambda: engine.aggRiskAnalyst.analyseAndReply(
                aggDebatePrompt, engine.toolRegistry, engine.timestamp, config, "critique", requireInitialTools=False
            ),
            lambda: engine.consRiskAnalyst.analyseAndReply(
                consDebatePrompt, engine.toolRegistry, engine.timestamp, config, "critique", requireInitialTools=False
            )
        )

        context.set("aggQuestionsRaw", aggQuestionsRaw)
        context.set("aggQuestionsUISummary", aggQuestionsUISummary)
        context.set("consQuestionsRaw", consQuestionsRaw)
        context.set("consQuestionsUISummary", consQuestionsUISummary)
        return None


# Bullish and bearish analyst's defense stage
class SingleEquityDefenseStage(BoardroomStage):
    def __init__(self, targetTicker: str, phaseNumber: int = 4):
        super().__init__(
            stageId="analystDefense",
            phaseNumber=phaseNumber,
            phaseName=f"Analyst Defense on {targetTicker}"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        consQuestionsRaw = context.get("consQuestionsRaw", "")
        aggQuestionsRaw = context.get("aggQuestionsRaw", "")

        bullDefensePrompt = (
            f"Questions Posed by Conservative Risk Analyst:\n{consQuestionsRaw}\n\n"
            f"Task: Defend your bullish thesis and price targets for {targetTicker}.\n"
            f"Answer each question quantitatively using your tools or Python models if needed. Revise your thesis, targets, or rating if substantiated deficiencies were highlighted."
        )
        bearDefensePrompt = (
            f"Questions Posed by Aggressive Risk Analyst:\n{aggQuestionsRaw}\n\n"
            f"Task: Defend your bearish risk analysis and price targets for {targetTicker}.\n"
            f"Answer each question quantitatively using your tools or Python models if needed. Revise your risk assessment, targets, or rating if substantiated upside catalysts were highlighted."
        )

        (bullDefenseRaw, bullDefenseUISummary), (bearDefenseRaw, bearDefenseUISummary) = engine.runAgentsConcurrently(
            lambda: engine.bullAnalyst.analyseAndReply(
                bullDefensePrompt, engine.toolRegistry, engine.timestamp, config, "defense", requireInitialTools=False
            ),
            lambda: engine.bearAnalyst.analyseAndReply(
                bearDefensePrompt, engine.toolRegistry, engine.timestamp, config, "defense", requireInitialTools=False
            )
        )

        context.set("bullDefenseRaw", bullDefenseRaw)
        context.set("bullDefenseUISummary", bullDefenseUISummary)
        context.set("bearDefenseRaw", bearDefenseRaw)
        context.set("bearDefenseUISummary", bearDefenseUISummary)
        return None


# 5. Aggressive and Conservative Risk Analyst's proposal stage
class SingleEquityProposalStage(BoardroomStage):
    def __init__(self, targetTicker: str, phaseNumber: int = 5):
        super().__init__(
            stageId="riskProposals",
            phaseNumber=phaseNumber,
            phaseName=f"Q&A-Based Proposals on {targetTicker}"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()
        bearDefenseRaw = context.get("bearDefenseRaw", "")
        bullDefenseRaw = context.get("bullDefenseRaw", "")

        aggProposalPrompt = (
            f"Bearish Analyst's Defense:\n{bearDefenseRaw}\n\n"
            f"Task: Formulate your final aggressive allocation proposal for {targetTicker}.\n"
            f"Propose your {timeHorizonInfo['llmPriceTargets']} and position weight (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), justifying your high-upside growth assumptions."
        )
        consProposalPrompt = (
            f"Bullish Analyst's Defense:\n{bullDefenseRaw}\n\n"
            f"Task: Formulate your final conservative allocation proposal for {targetTicker}.\n"
            f"Propose your {timeHorizonInfo['llmPriceTargets']} and position weight (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), incorporating a robust margin of safety."
        )

        (aggProposalRaw, aggProposalUISummary), (consProposalRaw, consProposalUISummary) = engine.runAgentsConcurrently(
            lambda: engine.aggRiskAnalyst.analyseAndReply(
                aggProposalPrompt, engine.toolRegistry, engine.timestamp, config, subrole="proposal", requireInitialTools=False
            ),
            lambda: engine.consRiskAnalyst.analyseAndReply(
                consProposalPrompt, engine.toolRegistry, engine.timestamp, config, subrole="proposal", requireInitialTools=False
            )
        )

        context.set("aggProposalRaw", aggProposalRaw)
        context.set("aggProposalUISummary", aggProposalUISummary)
        context.set("consProposalRaw", consProposalRaw)
        context.set("consProposalUISummary", consProposalUISummary)
        return None


# 6. Conclusion and decision stage with Portfolio Manager
class SingleEquityDecisionStage(BoardroomStage):
    def __init__(self, targetTicker: str, phaseNumber: int):
        super().__init__(
            stageId="finalExecutiveDecision",
            phaseNumber=phaseNumber,
            phaseName=f"Final Executive Decision on {targetTicker}"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        pace = config.boardroomPace
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]
        macroRaw = context.get("macroRaw", "")

        if pace == BoardroomPace.FAST:
            bullThesisRaw = context.get("bullThesisRaw", "")
            bearThesisRaw = context.get("bearThesisRaw", "")
            managerPrompt = (
                f"Target Asset: {targetTicker}\n"
                f"Time Horizon: {timeHorizonInfo['label']}\n\n"
                f"Macro Conditions:\n{macroRaw}\n\n"
                f"Aggressive Allocation Case:\n{bullThesisRaw}\n\n"
                f"Conservative Allocation Case:\n{bearThesisRaw}\n\n"
                f"Task: Produce the final executive investment decision for {targetTicker} over the {timeHorizonInfo['label']} time horizon.\n"
                f"Weigh upside potential against solvency risks. You MUST verify your final price targets using the 'calculateDistFromCurrPrice' tool. "
                f"Include a definitive rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}."
            )
            requireTools = False
        else:
            aggProposalRaw = context.get("aggProposalRaw", "")
            consProposalRaw = context.get("consProposalRaw", "")
            managerPrompt = (
                f"Target Asset: {targetTicker}\n"
                f"Time Horizon: {timeHorizonInfo['label']}\n\n"
                f"Macro Conditions:\n{macroRaw}\n\n"
                f"Aggressive Allocation Case:\n{aggProposalRaw}\n\n"
                f"Conservative Allocation Case:\n{consProposalRaw}\n\n"
                f"Task: Produce the final executive investment decision for {targetTicker} over the {timeHorizonInfo['label']} time horizon.\n"
                f"Weigh upside potential against solvency risks. You MUST verify your final price targets using the 'calculateDistFromCurrPrice' tool. "
                f"Include a definitive rating (BUY/HOLD/SELL), weighting (OVERWEIGHT/EQUAL-WEIGHT/UNDERWEIGHT), and {timeHorizonInfo['llmFinalLinePriceTargets']}."
            )
            requireTools = True

        engine.portManager.removeTool(finalSubmitToolName)
        finalDecisionRaw, finalDecisionUISummary = engine.portManager.analyseAndReply(
            managerPrompt, engine.toolRegistry, engine.timestamp, config, subrole="decision", requireInitialTools=requireTools
        )

        context.set("finalDecisionRaw", finalDecisionRaw)
        context.set("finalDecisionUISummary", finalDecisionUISummary)
        return None


# 7. Mandated tool stage upload for the portfolio manager's final decision
class SingleEquityDecisionUploadStage(BoardroomStage):
    def __init__(self, phaseNumber: int):
        super().__init__(
            stageId="decisionUpload",
            phaseNumber=phaseNumber,
            phaseName="Decision Upload"
        )

    def execute(self, context: BoardroomContext) -> Optional[str]:
        engine: "SingleEquityBoardroomEngine" = context.engine
        config: SingleEquityRatingConfig = context.config
        pace = config.boardroomPace
        targetTicker = config.ticker
        timeHorizonInfo = config.getTimeHorizonInfo()
        finalSubmitToolName = timeHorizonInfo["llmSubmitToolName"]
        finalDecisionRaw = context.get("finalDecisionRaw", "")

        uploadPrompt = (
            f"Target Asset: {targetTicker}\n"
            f"Final Decision Summary:\n{finalDecisionRaw}\n\n"
            f"Task: Upload and log the final boardroom verdict for {targetTicker}.\n"
            f"Execute the {finalSubmitToolName} tool with ticker='{targetTicker}', rating, weighting and the {timeHorizonInfo['llmFinalLinePriceTargets']} based on your final decision."
        )

        engine.portManager.clearTools()
        engine.portManager.addTool(finalSubmitToolName, engine.toolRegistry)
        _, _ = engine.executeMandatedToolStage(
            agent=engine.portManager,
            initialPrompt=uploadPrompt,
            mandatedToolName=finalSubmitToolName,
            config=config,
            subrole="upload",
            maxRetries=10,
            summarisationOverride=False,
            requireInitialTools=True
        )
        engine.portManager.tools = []

        try:
            formattedExecutiveDecision = engine.toolRegistry.getTool(finalSubmitToolName).toolLog[-1]
        except Exception:
            formattedExecutiveDecision = "Decision not found."

        if pace == BoardroomPace.FAST:
            convSummary = (
                f"\nMacro Analyst Response:\n{context.get('macroUISummary', '')}\n\n"
                f"\nBullish Analyst Response:\n{context.get('bullThesisUISummary', '')}\n\n"
                f"\nBearish Analyst Response:\n{context.get('bearThesisUISummary', '')}\n\n"
                f"\nFinal Executive Response:\n{context.get('finalDecisionUISummary', '')}\n\n"
                f"\n{formattedExecutiveDecision}\n"
            )
        else:
            convSummary = (
                f"\nMacro Analyst Response:\n{context.get('macroUISummary', '')}\n\n"
                f"======================"
                f"\nBearish Analyst Response:\n{context.get('bearThesisUISummary', '')}\n\n"
                f"\nAggressive Risk Analyst Critique:\n{context.get('aggQuestionsUISummary', '')}\n\n"
                f"\nBearish Analyst Defense:\n{context.get('bearDefenseUISummary', '')}\n\n"
                f"\nAggressive Risk Analyst Proposal:\n{context.get('aggProposalUISummary', '')}\n\n"
                f"======================"
                f"\nBullish Analyst Response:\n{context.get('bullThesisUISummary', '')}\n\n"
                f"\nConservative Risk Analyst Critique:\n{context.get('consQuestionsUISummary', '')}\n\n"
                f"\nBullish Analyst Defense:\n{context.get('bullDefenseUISummary', '')}\n\n"
                f"\nConservative Risk Analyst Proposal:\n{context.get('consProposalUISummary', '')}\n\n"
                f"======================"
                f"\nFinal Executive Response:\n{context.get('finalDecisionUISummary', '')}\n\n"
                f"\n{formattedExecutiveDecision}\n"
            )

        engine.fullConvSummary = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', convSummary)
        engine.lastConfig = config
        if formattedExecutiveDecision and formattedExecutiveDecision != "Decision not found.":
            engine.portManagerFinalOutput = f"{finalDecisionRaw}\n\n{formattedExecutiveDecision}"
        else:
            engine.portManagerFinalOutput = finalDecisionRaw

        return None


# The main complete engine including managing the QnA system
class SingleEquityBoardroomEngine(BoardroomEngine):
    def __init__(
            self, 
            toolRegistry: ToolRegistry,
            timestamp: pd.Timestamp
        ):
        super().__init__(toolRegistry, timestamp)

        # For managing the QnA turns:
        self.lastConfig: Optional[SingleEquityRatingConfig] = None
        self.fullConvSummary: str = ""
        self.portManagerFinalOutput: Optional[str] = None
        self.qnaHistory: list = []
        self.qnaTurns: list = []

        self.configureTransferToolSchema(BoardroomPace.COMPLETE)

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
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchLatest10QSentiment"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchTickerSentimentHistory"],
                toolMap["fetchSentimentDivergence"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="green",
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchLatest10QSentiment"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchTickerSentimentHistory"],
                toolMap["fetchSentimentDivergence"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="red",
            dateStr=timestampStr
        )

        self.aggRiskAnalyst = FinancialAgent(
            agentRole="Aggressive Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchLatest10QSentiment"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchTickerSentimentHistory"],
                toolMap["fetchSentimentDivergence"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="yellow",
            dateStr=timestampStr
        )

        self.consRiskAnalyst = FinancialAgent(
            agentRole="Conservative Risk Analyst",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchLatest10QSentiment"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchTickerSentimentHistory"],
                toolMap["fetchSentimentDivergence"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="blue",
            dateStr=timestampStr
        )

        self.portManager = FinancialAgent(
            agentRole="Impartial Portfolio Manager",
            tools=[
                toolMap["fetchCompanyProfile"],
                toolMap["executePythonCalculation"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["confirmBoardroomDecisionLongTerm"]
            ],
            color="magenta",
            dateStr=timestampStr
        )

        self.oneShotAnalyst = FinancialAgent(
            agentRole="One-Shot Analyst",
            tools=[
                toolMap["fetchMacroContext"],
                toolMap["fetchMacroNews"],
                toolMap["fetchMacroSentimentHistory"],
                toolMap["fetchAllSectorRankings"],
                toolMap["fetchSectorPerformance"],
                toolMap["fetchSectorProfile"],
                toolMap["fetchCompanyProfile"],
                toolMap["fetchCompanyValuationMetrics"],
                toolMap["fetchIncomeStatement"],
                toolMap["fetchBalanceSheet"],
                toolMap["fetchCashFlowStatement"],
                toolMap["fetchLatest10QSentiment"],
                toolMap["fetchStockPricePerformance"],
                toolMap["fetchCompanyRecentNews"],
                toolMap["fetchTickerSentimentHistory"],
                toolMap["fetchSentimentDivergence"],
                toolMap["calculateDistFromCurrPrice"],
                toolMap["executePythonCalculation"]
            ],
            color="cyan",
            dateStr=timestampStr
        )

        self.spokesperson = FinancialAgent(
            agentRole="Boardroom Spokesperson",
            tools=[],
            color="cyan",
            dateStr=timestampStr
        )

        self.agents = {
            "macroAnalyst": self.macroAnalyst,
            "bullAnalyst": self.bullAnalyst,
            "bearAnalyst": self.bearAnalyst,
            "aggRiskAnalyst": self.aggRiskAnalyst,
            "consRiskAnalyst": self.consRiskAnalyst,
            "portManager": self.portManager,
            "oneShotAnalyst": self.oneShotAnalyst,
            "spokesperson": self.spokesperson
        }
        self.agentsList = list(self.agents.values())

        self.specialistMap = {
            "Macro Analyst": self.macroAnalyst,
            "Bullish Value Analyst": self.bullAnalyst,
            "Bearish Risk Analyst": self.bearAnalyst,
            "Aggressive Risk Analyst": self.aggRiskAnalyst,
            "Conservative Risk Analyst": self.consRiskAnalyst,
            "Impartial Portfolio Manager": self.portManager,
            "One-Shot Analyst": self.oneShotAnalyst
        }

        if self.llmClient is not None:
            for agent in self.agentsList:
                agent.setClient(self.llmClient)


    def getDefaultPhaseAgents(self, phaseNumber: int, pace: BoardroomPace) -> List[Dict[str, str]]:
        if pace == BoardroomPace.ONE_SHOT:
            analystColor = self.oneShotAnalyst.color if self.oneShotAnalyst is not None else self.macroAnalyst.color
            return [{"role": "One-Shot Analyst", "color": analystColor, "name": "One-Shot Analyst"}]

        isFast = pace == BoardroomPace.FAST
        if phaseNumber == 1:
            return [{"role": "Macro Analyst", "color": self.macroAnalyst.color, "name": "Macro Analyst"}]
        elif phaseNumber == 2:
            return [
                {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
            ]
        elif phaseNumber == 3:
            if isFast:
                return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
            else:
                return [
                    {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.color, "name": "Aggressive Risk Analyst"},
                    {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.color, "name": "Conservative Risk Analyst"}
                ]
        elif phaseNumber == 4:
            if isFast:
                return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
            else:
                return [
                    {"role": "Bullish Value Analyst", "color": self.bullAnalyst.color, "name": "Bullish Analyst"},
                    {"role": "Bearish Risk Analyst", "color": self.bearAnalyst.color, "name": "Bearish Analyst"}
                ]
        elif phaseNumber == 5:
            return [
                {"role": "Aggressive Risk Analyst", "color": self.aggRiskAnalyst.color, "name": "Aggressive Risk Analyst"},
                {"role": "Conservative Risk Analyst", "color": self.consRiskAnalyst.color, "name": "Conservative Risk Analyst"}
            ]
        elif phaseNumber == 6:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
        elif phaseNumber == 7:
            return [{"role": "Impartial Portfolio Manager", "color": self.portManager.color, "name": "Portfolio Manager"}]
        return []




    def buildFSM(self, config: SingleEquityRatingConfig) -> BoardroomFSM:
        pace = config.boardroomPace
        targetTicker = config.ticker

        if pace == BoardroomPace.ONE_SHOT:
            fsm = BoardroomFSM(initialStageId="oneShotAnalysis", title=f"One-Shot Boardroom Evaluation for: {targetTicker}")
            fsm.addStage(SingleEquityOneShotAnalysisStage(), nextStage="oneShotUpload")
            fsm.addStage(SingleEquityOneShotUploadStage(), nextStage=None)
            return fsm
        elif pace == BoardroomPace.FAST:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title=f"Fast Boardroom Evaluation for: {targetTicker}")
            fsm.addStage(SingleEquityMacroStage(phaseNumber=1), nextStage="specialistResearch")
            fsm.addStage(SingleEquityResearchStage(targetTicker=targetTicker, phaseNumber=2), nextStage="finalExecutiveDecision")
            fsm.addStage(SingleEquityDecisionStage(targetTicker=targetTicker, phaseNumber=3), nextStage="decisionUpload")
            fsm.addStage(SingleEquityDecisionUploadStage(phaseNumber=4), nextStage=None)
            return fsm
        else:
            fsm = BoardroomFSM(initialStageId="macroAnalysis", title=f"Live Boardroom Evaluation for: {targetTicker}")
            fsm.addStage(SingleEquityMacroStage(phaseNumber=1), nextStage="specialistResearch")
            fsm.addStage(SingleEquityResearchStage(targetTicker=targetTicker, phaseNumber=2), nextStage="seniorRiskDebate")
            fsm.addStage(SingleEquityDebateStage(targetTicker=targetTicker, phaseNumber=3), nextStage="analystDefense")
            fsm.addStage(SingleEquityDefenseStage(targetTicker=targetTicker, phaseNumber=4), nextStage="riskProposals")
            fsm.addStage(SingleEquityProposalStage(targetTicker=targetTicker, phaseNumber=5), nextStage="finalExecutiveDecision")
            fsm.addStage(SingleEquityDecisionStage(targetTicker=targetTicker, phaseNumber=6), nextStage="decisionUpload")
            fsm.addStage(SingleEquityDecisionUploadStage(phaseNumber=7), nextStage=None)
            return fsm


    def execute(self, config: SingleEquityRatingConfig):
        if self.llmClient is None:
            raise ValueError("LLM client is not assigned. Please assign a client before executing the boardroom.")

        self.toolRegistry.clearToolLogs()
        self.configureTransferToolSchema(config.boardroomPace)
        self.llmClient.newTask()
        self.lastConfig = config

        context = BoardroomContext(engine=self, config=config)
        fsm = self.buildFSM(config)
        fsm.run(context)


    # Non standard QnA management methods for this engine only 

    def deleteQnATurn(self, turnIndex: int) -> bool:
        if 0 <= turnIndex < len(self.qnaTurns):
            targetTurn = self.qnaTurns[turnIndex]
            
            # Prune spokesperson message history
            if self.spokesperson:
                spokespersonTargetLen = targetTurn.get("spokespersonHistoryLen", 0)
                self.spokesperson.messageHistory = self.spokesperson.messageHistory[:spokespersonTargetLen]
            
            # Prune specialist message histories
            for role, targetLen in targetTurn.get("specialistHistoryLens", {}).items():
                agent = self.specialistMap.get(role)
                if agent is not None:
                    agent.messageHistory = agent.messageHistory[:targetLen]
            
            # Prune qnaTurns list
            self.qnaTurns = self.qnaTurns[:turnIndex]
            return True
        return False

    def configureTransferToolSchema(self, pace: BoardroomPace) -> None:
        activeRoles = getActiveSpecialistRoles(pace)
        transferTool = self.toolRegistry.getTool("transferToAgent") if self.toolRegistry else None
        if transferTool:
            transferTool.paramSchema = {
                "type": "object",
                "properties": {
                    "agentRole": {
                        "type": "string",
                        "enum": activeRoles,
                        "description": f"The exact name of the active specialist boardroom agent to transfer to ({', '.join(activeRoles)})."
                    },
                    "transferMessage": {
                        "type": "string",
                        "description": "A clear, concise instruction or summary of the question for the specialist agent to answer."
                    }
                },
                "required": ["agentRole", "transferMessage"]
            }

    def getPortfolioManagerFinalOutput(self) -> Optional[str]:
        if self.portManagerFinalOutput:
            return self.portManagerFinalOutput

        # Fallback, extract from portManager's message history if available
        if self.portManager and self.portManager.messageHistory:
            assistantMessages = [
                msg["content"] for msg in self.portManager.messageHistory 
                if msg.get("role") == "assistant" and msg.get("content")
            ]
            if assistantMessages:
                return "\n\n".join(assistantMessages)

        return None

    # Handles when the spokesperson in the QnA transfers from themselves to a specialist
    def executeSpecialistTransfer(self, agentRole: str, transferMessage: str, config: SingleEquityRatingConfig) -> Dict[str, Any]:
        specialist = self.specialistMap.get(agentRole)
        if not specialist:
            return {"error": f"Specialist agent '{agentRole}' not found."}

        if self.llmClient and not specialist.llmClient:
            specialist.setClient(self.llmClient)

        setCurrentStage("qa")

        dateStr = config.simulatedDateStr if config and config.simulatedDateStr else self.timestamp.strftime("%Y-%m-%d")
        sysPrompt = buildSpecialistQnASysPrompt(
            dateStr=dateStr,
            agentRole=agentRole,
            toolsStr=specialist.getSpecificToolsStr(),
            promptArgs=config.getPromptArgs() if config else None
        )

        pmOutput = self.getPortfolioManagerFinalOutput()
        isPortManager = (specialist is self.portManager) or (agentRole in ["Impartial Portfolio Manager", "One-Shot Analyst"])

        if not isPortManager and pmOutput:
            pmContext = (
                f"Portfolio Manager's Final Executive Decision:\n"
                f"{pmOutput}\n\n"
            )
            taskGuidance = (
                f"Task: Answer the user's question directly from your role as {agentRole}. "
                f"Rely on your previous thinking steps, tool outputs, message history from earlier stages, "
                f"and the Portfolio Manager's final executive decision provided above."
            )
        else:
            pmContext = ""
            taskGuidance = (
                f"Task: Answer the user's question directly from your role as {agentRole}. "
                f"Rely on your previous thinking steps, tool outputs, and message history from earlier stages."
            )

        incomingPrompt = (
            f"The Boardroom Spokesperson has transferred the following user question to you:\n\n"
            f"{pmContext}"
            f"Transfer Request: {transferMessage}\n\n"
            f"{taskGuidance}"
        )

        try:
            rawResponse, _ = specialist.analyseAndReply(
                incomingMessage=incomingPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                subrole="qa",
                requireInitialTools=False,
                summarisationOverride=False,
                sysPromptOverride=sysPrompt
            )
            return {
                "status": "success",
                "specialist": agentRole,
                "response": rawResponse
            }
        except Exception as e:
            return {
                "status": "error",
                "specialist": agentRole,
                "error": str(e)
            }
        finally:
            emitEvent("agentRunEnd", {
                "agentRole": agentRole,
                "phase": "raw",
                "stageNum": "qa"
            })
            if self.spokesperson:
                setCurrentAgent("Boardroom Spokesperson", self.spokesperson.color)
            setCurrentStage("qa")
            setAgentPhase("raw")


    # Handles the entire QnA process
    def processQnAQuery(self, query: str, targetAgent: Optional[str] = None):
        config = self.lastConfig
        activeRoles = getActiveSpecialistRoles(config.boardroomPace)
        self.configureTransferToolSchema(config.boardroomPace)

        setCurrentStage("qa")

        turnRecord = {
            "turnIndex": len(self.qnaTurns),
            "userQuery": query,
            "targetAgent": targetAgent,
            "spokespersonHistoryLen": len(self.spokesperson.messageHistory) if self.spokesperson else 0,
            "specialistHistoryLens": {
                role: len(agent.messageHistory)
                for role, agent in self.specialistMap.items()
                if agent is not None
            }
        }
        self.qnaTurns.append(turnRecord)
        self.qnaHistory.append({"role": "user", "content": query, "targetAgent": targetAgent})

        # Check if direct delegation was requested to a specific active specialist
        isDirectTarget = bool(targetAgent and targetAgent != "auto" and targetAgent in self.specialistMap and targetAgent in activeRoles)

        if isDirectTarget:
            try:
                res = self.executeSpecialistTransfer(targetAgent, query, config)
                if res and "response" in res:
                    self.qnaHistory.append({"role": "assistant", "agentRole": targetAgent, "content": res["response"]})
            except SimulationStoppedException:
                emitEvent("simStopped", {"message": "Q&A stopped by user."})
            except Exception as e:
                if isStopRequested():
                    emitEvent("simStopped", {"message": "Q&A stopped by user."})
                else:
                    traceback.print_exc()
                    emitEvent("error", {"message": f"Q&A Error: {str(e)}"})
            finally:
                emitEvent("qaComplete", {"stageNum": "qa"})
            return

        # Default Auto-Delegate flow through Spokesperson
        transferTool = self.toolRegistry.getTool("transferToAgent") if self.toolRegistry else None
        self.spokesperson.tools = [transferTool] if transferTool else []
        if transferTool:
            transferTool.toolLog.clear()

        contextStr = self.fullConvSummary or "No context found!"
        dateStr = config.simulatedDateStr if config.simulatedDateStr else self.timestamp.strftime("%Y-%m-%d")
        sysPrompt = buildSpokespersonSysPrompt(
            dateStr=dateStr,
            toolsStr="transferToAgent",
            boardroomContextStr=contextStr,
            promptArgs=config.getPromptArgs(),
            activeRoles=activeRoles
        )

        try:
            spokespersonPrompt = (
                f"User Question: {query}\n\n"
                f"Task: Review the question against the completed boardroom discussion context.\n"
                f"- If this is a simple or high-level inquiry, answer directly.\n"
                f"- If this requires specialist financial depth, quantitative valuation, macro analysis, or risk challenge, "
                f"delegate to the specialist(s) using the 'transferToAgent' tool at the end of your response."
            )

            rawSpokespersonResponse, _ = self.spokesperson.analyseAndReply(
                incomingMessage=spokespersonPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                subrole="qa",
                requireInitialTools=False,
                summarisationOverride=False,
                sysPromptOverride=sysPrompt
            )

            self.qnaHistory.append({"role": "assistant", "content": rawSpokespersonResponse})

            # Check if transfers were logged during this turn
            if transferTool and transferTool.toolLog:
                for transferCall in list(transferTool.toolLog):
                    agentRole = transferCall.get("agentRole")
                    transferMessage = transferCall.get("transferMessage")
                    if agentRole and transferMessage:
                        self.executeSpecialistTransfer(agentRole, transferMessage, config)

        except SimulationStoppedException:
            emitEvent("simStopped", {"message": "Q&A stopped by user."})
        except Exception as e:
            if isStopRequested():
                emitEvent("simStopped", {"message": "Q&A stopped by user."})
            else:
                traceback.print_exc()
                emitEvent("error", {"message": f"Q&A Error: {str(e)}"})
        finally:
            emitEvent("agentRunEnd", {
                "agentRole": "Boardroom Spokesperson",
                "phase": "raw",
                "stageNum": "qa"
            })
            emitEvent("qaComplete", {"stageNum": "qa"})
