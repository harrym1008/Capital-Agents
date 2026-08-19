
import pandas as pd

from boardroom.boardroom_engine import BoardroomEngine
from llmtools.tool_registry import ToolRegistry
from llm.agents.agent import FinancialAgent

from cli.ansi import ANSI


def generateBoardroom(toolRegistry: ToolRegistry, timestamp: pd.Timestamp) -> BoardroomEngine:
    timestampStr = timestamp.strftime("%Y-%m-%d")
    toolMap = toolRegistry.getToolMap()

    macroAgent = FinancialAgent(
        agentRole="Macro Analyst",
        tools=[
            toolMap["fetchMacroContext"],
            toolMap["fetchMacroNews"],
            toolMap["fetchMacroSentimentHistory"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.CYAN,
        dateStr=timestampStr
    )

    bullAgent = FinancialAgent(
        agentRole="Bullish Value Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            toolMap["fetchLatest10QSentiment"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.GREEN,
        dateStr=timestampStr
    )

    bearAgent = FinancialAgent(
        agentRole="Bearish Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            toolMap["fetchLatest10QSentiment"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.RED,
        dateStr=timestampStr
    )

    aggRiskAnalystAgent = FinancialAgent(
        agentRole="Aggressive Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            toolMap["fetchLatest10QSentiment"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.YELLOW,
        dateStr=timestampStr
    )

    consRiskAnalystAgent = FinancialAgent(
        agentRole="Conservative Risk Analyst",
        tools=[
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            toolMap["fetchLatest10QSentiment"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.BLUE,
        dateStr=timestampStr
    )

    portfolioManager = FinancialAgent(
        agentRole="Impartial Portfolio Manager",
        tools=[
            toolMap["fetchCompanyProfile"],
            # toolMap["fetchLatest10QSentiment"],
            toolMap["executePythonCalculation"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["confirmBoardroomDecisionLongTerm"]
        ],
        ansiColor=ANSI.MAGENTA,
        dateStr=timestampStr
    )    

    oneShotAnalyst = FinancialAgent(
        agentRole="One-Shot Analyst",
        tools=[
            toolMap["fetchMacroContext"],
            toolMap["fetchMacroNews"],
            toolMap["fetchMacroSentimentHistory"],
            toolMap["fetchCompanyProfile"],
            toolMap["fetchCompanyValuationMetrics"],
            toolMap["fetchIncomeStatement"],
            toolMap["fetchBalanceSheet"],
            toolMap["fetchCashFlowStatement"],
            toolMap["fetchLatest10QSentiment"],
            # toolMap["fetchStatementOfEquity"],
            # toolMap["fetchComprehensiveIncomeStatement"],
            toolMap["fetchStockPricePerformance"],
            toolMap["fetchCompanyRecentNews"],
            toolMap["fetchTickerSentimentHistory"],
            toolMap["fetchSentimentDivergence"],
            toolMap["calculateDistFromCurrPrice"],
            toolMap["executePythonCalculation"]
        ],
        ansiColor=ANSI.CYAN,
        dateStr=timestampStr
    )

    spokespersonAgent = FinancialAgent(
        agentRole="Boardroom Spokesperson",
        tools=[],
        ansiColor=ANSI.CYAN,
        dateStr=timestampStr
    )

    boardroom = BoardroomEngine(
        agents={
            "macroAnalyst": macroAgent,
            "bullAnalyst": bullAgent,
            "bearAnalyst": bearAgent,
            "aggRiskAnalyst": aggRiskAnalystAgent,
            "consRiskAnalyst": consRiskAnalystAgent,
            "portManager": portfolioManager,
            "oneShotAnalyst": oneShotAnalyst,
            "spokesperson": spokespersonAgent
        }, 
        timestamp=timestamp,
        toolRegistry=toolRegistry
    )

    return boardroom
