from typing import Dict, Any, Optional, List, Tuple

from colorama import Fore, Style
import pandas as pd
import numpy as np

from collectors.constants import NEW_YORK
from dataquery.macro_provider import MacroSeries
from llmtools.tool_registry import ToolRegistry, Tool
from llmtools.functions.company import calculateSharpeRatio
from llmtools.functions.confirmation import (
    confirmSectorAllocation,
    confirmPortfolioAllocation,
    decideRebalanceNecessity
)
from llm.agents.agent import FinancialAgent
from boardroom.boardroom_config import (
    AgentPortfolioSimulationConfig, 
    SimulationTimestep, 
    BoardroomPace
)
from boardroom.boardroom_engine import BoardroomEngine
from simulation.market_sim import MarketSimulation, ExecutionTime
from simulation.orders import MarketOrder, OrderSide
from ui.ui_hooks import setCurrentStage,  setCurrentMilestoneId, emitEvent, SimulationStoppedException, isStopRequested


# Formats a datetime as Day DD Mon YYYY
def formatDateFriendly(dt: Any) -> str:
    if dt is None or pd.isna(dt):
        return "--"
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    return f"{dt:%a} {dt.day} {dt:%b} {dt:%Y}"


# Formats a datetime as DD-MM-YYYY
def formatDateDdMmYyyy(dt: Any) -> str:
    """Formats a datetime or timestamp as 'DD-MM-YYYY'."""
    if dt is None or pd.isna(dt):
        return "--"
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    return dt.strftime("%d-%m-%Y")


# Returns S&P 500 nearest closing price in the past for a given timestamp 
def getSp500PriceOnDate(dateTs: pd.Timestamp, sp500Df: pd.DataFrame) -> float:
    if sp500Df is None or sp500Df.empty or "close" not in sp500Df.columns:
        return 1.0
    normDate = dateTs.tz_localize(None).normalize() if dateTs.tzinfo is not None else dateTs.normalize()
    dateCol = "normDate" if "normDate" in sp500Df.columns else "date"
    sub = sp500Df[sp500Df[dateCol] <= normDate]
    if not sub.empty:
        return float(sub.iloc[-1]["close"])
    return float(sp500Df.iloc[0]["close"])


# Calculate the next milestone date
def computeNextMilestoneDate(currentTs: pd.Timestamp, timestep: SimulationTimestep) -> pd.Timestamp:
    match timestep:
        case SimulationTimestep.ONE_WEEK:
            return currentTs + pd.Timedelta(weeks=1)
        case SimulationTimestep.TWO_WEEKS:
            return currentTs + pd.Timedelta(weeks=2)
        case SimulationTimestep.THREE_WEEKS:
            return currentTs + pd.Timedelta(weeks=3)
        case SimulationTimestep.ONE_MONTH:
            return currentTs + pd.DateOffset(months=1)
        case SimulationTimestep.TWO_MONTHS:
            return currentTs + pd.DateOffset(months=2)
        case SimulationTimestep.THREE_MONTHS:
            return currentTs + pd.DateOffset(months=3)
        case _:
            return currentTs + pd.DateOffset(months=1)


# Truncates a string to a maximum number of words
def truncateToWords(text: str, maxWords: int = 500) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= maxWords:
        return text
    return " ".join(words[:maxWords]) + " ... [truncated]"


# The agent portfolio simulation engine, manages multi-agent portfolio creation, rebalancing and market simulation management
# It is jerry-rigged to work with the existing UI hooks so it is not a clean bit of code! 
# If I had more time, it would be heavily, heavily refactored/rewritten 
class AgentPortfolioSimulationEngine(BoardroomEngine):
    def __init__(self, toolRegistry: ToolRegistry, timestamp: pd.Timestamp):
        super().__init__(toolRegistry, timestamp)
        self.marketSim: Optional[MarketSimulation] = None
        self.confirmedSectorAllocation: Optional[Dict[str, Any]] = None
        self.confirmedPortfolioAllocation: Optional[Dict[str, Any]] = None
        self.lastConfig: Optional[AgentPortfolioSimulationConfig] = None

        # Instance-bound simulation journal
        self.journal: List[Dict[str, Any]] = []

        # Historical Stage 6 executive decisions for Performance Evaluation
        self.stage6Decisions: List[Dict[str, Any]] = []

        # Simulation history tracking
        self.portfolioHistoryPoints: List[Dict[str, Any]] = []
        self.sp500HistoryPoints: List[Dict[str, Any]] = []
        self.rebalanceHistory: List[Dict[str, Any]] = []
        self.milestonesList: List[Dict[str, Any]] = []
        self.consecutiveSkippedSteps: int = 0
        self.currentMilestoneId: str = "milestone_0"
        self.currentMilestoneLabel: str = "Inception"
        self.currentSimStatus: str = "Idle"
        self.sp500InitialPrice: float = 1.0

    # Override emitNewPhase to include milestone ID and agent list
    def emitNewPhase(self, phaseNumber: int, phaseName: str, pace: BoardroomPace = BoardroomPace.COMPLETE, 
                     customAgents: Optional[List[Dict[str, str]]] = None):
        setCurrentStage(phaseNumber)
        setCurrentMilestoneId(self.currentMilestoneId)
        
        if phaseNumber == 0:
            tempHeader = f"{phaseName}"
        else:
            tempHeader = f"Phase {phaseNumber}: {phaseName}"
        tempHeader = f"{'|'*5} {tempHeader} {'|'*5}"
        headerLength = len(tempHeader)
        print(f"\n{Style.BRIGHT}{'-'*headerLength}\n{tempHeader}\n{'-'*headerLength}{Style.RESET_ALL}\n")

        if customAgents is not None:
            emitEvent("stageStart", {
                "stageNum": phaseNumber,
                "stageName": phaseName,
                "agents": customAgents,
                "milestoneId": self.currentMilestoneId
            })
            return

        agents = self.getDefaultPhaseAgents(phaseNumber, pace)
        emitEvent("stageStart", {
            "stageNum": phaseNumber,
            "stageName": phaseName,
            "agents": agents,
            "milestoneId": self.currentMilestoneId
        })

    # Read and record into the journal (to bring information from one milestone to the next)        
    def readJournal(self, tool: Tool, data: Any, timestamp: pd.Timestamp, limit: int = 10) -> Dict[str, Any]:
        currentDateStr = timestamp.strftime("%Y-%m-%d")
        entries = [e for e in self.journal if e.get("date", "") <= currentDateStr]
        if limit and limit > 0:
            entries = entries[-limit:]

        if not entries:
            return {
                "status": "success",
                "entryCount": 0,
                "asOfDate": currentDateStr,
                "message": "The journal is currently empty.",
                "entries": []
            }

        summaryStr = f"Retrieved {len(entries)} historical journal entries up to {currentDateStr}."
        tool.toolLog.append({"summary": summaryStr, "count": len(entries)})
        return {
            "status": "success",
            "entryCount": len(entries),
            "asOfDate": currentDateStr,
            "message": summaryStr,
            "entries": entries
        }

    def recordJournalEntry(self, tool: Tool, data: Any, timestamp: pd.Timestamp, entryText: str, 
                           strategicOutlook: str = "", actionTaken: str = "") -> Dict[str, Any]:
        currentDateStr = timestamp.strftime("%Y-%m-%d")
        entryIndex = len(self.journal) + 1
        entry = {
            "entryIndex": entryIndex,
            "date": currentDateStr,
            "milestone": self.currentMilestoneLabel or f"Milestone {entryIndex}",
            "summary": str(entryText).strip(),
            "strategicOutlook": str(strategicOutlook).strip(),
            "actionTaken": str(actionTaken).strip()
        }
        self.journal.append(entry)
        summaryStr = f"Journal entry #{entryIndex} recorded for {currentDateStr}."
        tool.toolLog.append({"summary": summaryStr, "entry": entry})
        return {
            "status": "success",
            "message": summaryStr,
            "recordedEntry": entry
        }


    # Jerry rigged wrappers for the confirmation tools to emit events to the UI when sector or portfolio allocations are confirmed
    def _wrappedConfirmSectorAllocation(self, tool: Tool, data: Any, timestamp: pd.Timestamp, 
                                        sectorAllocations: Dict[str, float], rationale: str) -> Dict[str, Any]:
        res = confirmSectorAllocation(tool, data, timestamp, sectorAllocations, rationale)
        if isinstance(res, dict) and (res.get("status") == "success" or "confirmedAllocation" in res):
            if tool.toolLog:
                self.confirmedSectorAllocation = tool.toolLog[-1]
            emitEvent("sectorAllocationConfirmed", {
                "stageNum": 3,
                "confirmedAllocation": self.confirmedSectorAllocation,
                "milestoneId": self.currentMilestoneId
            })
        return res

    def _wrappedConfirmPortfolioAllocation(self, tool: Tool, data: Any, timestamp: pd.Timestamp, sectorAllocations: Any,
                                           portfolioRationale: str, initialCapital: float = 100000.0) -> Dict[str, Any]:
        res = confirmPortfolioAllocation(tool, data, timestamp, sectorAllocations, portfolioRationale, initialCapital)
        if isinstance(res, dict) and (res.get("status") == "success" or "confirmedPortfolio" in res):
            if tool.toolLog:
                self.confirmedPortfolioAllocation = tool.toolLog[-1]
            evtName = "portfolioCreated" if self.currentMilestoneId == "milestone_0" else "portfolioRebalanced"
            emitEvent(evtName, {
                "stageNum": 6,
                "confirmedPortfolio": self.confirmedPortfolioAllocation,
                "milestoneId": self.currentMilestoneId
            })
        return res

    def _wrappedDecideRebalanceNecessity(self, tool: Tool, data: Any, timestamp: pd.Timestamp, decision: str,
                                         reasoning: str, macroShiftDetected: bool = False, urgency: str = "none") -> Dict[str, Any]:
        res = decideRebalanceNecessity(tool, data, timestamp, decision, reasoning, macroShiftDetected, urgency)
        if isinstance(res, dict) and (res.get("status") == "success" or "decisionRecord" in res):
            record = res.get("decisionRecord") or (tool.toolLog[-1] if tool.toolLog else {})
            emitEvent("rebalanceDecisionConfirmed", {
                "milestoneId": self.currentMilestoneId,
                "decision": record.get("decision", decision),
                "reasoning": record.get("reasoning", reasoning),
                "urgency": record.get("urgency", urgency)
            })
        return res


    def generate(self) -> None:
        timestampStr = self.timestamp.strftime("%Y-%m-%d")
        toolMap = self.toolRegistry.getToolMap()

        readJournalTool = Tool(
            toolFunction=self.readJournal,
            toolName="readJournal",
            toolDescription="Reads previous executive journal entries from past milestones.",
            parameterSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of previous journal entries to retrieve (1 to 20). Defaults to 10.",
                        "default": 10
                    }
                },
                "required": []
            },
            storeIntoSources=True
        )

        readPortfolioJournalAliasTool = Tool(
            toolFunction=self.readJournal,
            toolName="readPortfolioJournal",
            toolDescription="Reads previous executive journal entries from past milestones.",
            parameterSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of previous journal entries to retrieve (1 to 20). Defaults to 10.",
                        "default": 10
                    }
                },
                "required": []
            },
            storeIntoSources=True
        )

        recordJournalEntryTool = Tool(
            toolFunction=self.recordJournalEntry,
            toolName="recordJournalEntry",
            toolDescription="Records an executive summary and strategic outlook into the journal for this milestone.",
            parameterSchema={
                "type": "object",
                "properties": {
                    "entryText": {
                        "type": "string",
                        "description": "A concise 30-50 word executive summary of the portfolio state, key decisions made, and risk observations for this milestone."
                    },
                    "strategicOutlook": {
                        "type": "string",
                        "description": "Strategic macroeconomic and sector outlook perspective for the upcoming period."
                    },
                    "actionTaken": {
                        "type": "string",
                        "description": "Brief description of the action taken (e.g., 'Maintained Holdings', 'Rebalanced 4 Sectors', 'Initiated Tech Expansion')."
                    }
                },
                "required": ["entryText"]
            },
            storeIntoSources=True
        )

        origConfirmSector = toolMap.get("confirmSectorAllocation")
        wrappedConfirmSectorTool = Tool(
            toolFunction=self._wrappedConfirmSectorAllocation,
            toolName="confirmSectorAllocation",
            toolDescription=origConfirmSector.description if origConfirmSector else "Confirms sector allocations.",
            parameterSchema=origConfirmSector.paramSchema if origConfirmSector else {},
            storeIntoSources=True
        )
        wrappedConfirmSectorTool.registry = self.toolRegistry

        origConfirmPort = toolMap.get("confirmPortfolioAllocation")
        wrappedConfirmPortTool = Tool(
            toolFunction=self._wrappedConfirmPortfolioAllocation,
            toolName="confirmPortfolioAllocation",
            toolDescription=origConfirmPort.description if origConfirmPort else "Confirms portfolio stock positions.",
            parameterSchema=origConfirmPort.paramSchema if origConfirmPort else {},
            storeIntoSources=True
        )
        wrappedConfirmPortTool.registry = self.toolRegistry

        origDecide = toolMap.get("decideRebalanceNecessity")
        wrappedDecideTool = Tool(
            toolFunction=self._wrappedDecideRebalanceNecessity,
            toolName="decideRebalanceNecessity",
            toolDescription=origDecide.description if origDecide else "Decides whether portfolio rebalance is necessary.",
            parameterSchema=origDecide.paramSchema if origDecide else {},
            storeIntoSources=True
        )
        wrappedDecideTool.registry = self.toolRegistry

        self.toolRegistry.registerTool(readJournalTool)
        self.toolRegistry.registerTool(readPortfolioJournalAliasTool)
        self.toolRegistry.registerTool(recordJournalEntryTool)
        self.toolRegistry.registerTool(wrappedConfirmSectorTool)
        self.toolRegistry.registerTool(wrappedConfirmPortTool)
        self.toolRegistry.registerTool(wrappedDecideTool)

        macroTools = [
            toolMap["fetchMacroContext"],
            toolMap["fetchMacroNews"],
            toolMap["fetchMacroSentimentHistory"],
            toolMap["fetchAllSectorRankings"],
            readJournalTool,
            readPortfolioJournalAliasTool,
            wrappedDecideTool,
            toolMap["executePythonCalculation"]
        ]

        self.macroAnalyst = FinancialAgent(
            agentRole="Macro Analyst",
            tools=macroTools,
            color="cyan",
            dateStr=timestampStr
        )

        self.bullAnalyst = FinancialAgent(
            agentRole="Bullish Value Analyst",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                readJournalTool,
                readPortfolioJournalAliasTool,
                toolMap["executePythonCalculation"]
            ],
            color="green",
            dateStr=timestampStr
        )

        self.bearAnalyst = FinancialAgent(
            agentRole="Bearish Risk Analyst",
            tools=[
                toolMap["fetchAllSectorsAnalysis"],
                readJournalTool,
                readPortfolioJournalAliasTool,
                toolMap["executePythonCalculation"]
            ],
            color="red",
            dateStr=timestampStr
        )

        pmTools = [
            toolMap["fetchAllSectorsAnalysis"],
            wrappedConfirmSectorTool,
            wrappedConfirmPortTool,
            readJournalTool,
            readPortfolioJournalAliasTool,
            recordJournalEntryTool,
            toolMap["executePythonCalculation"]
        ]

        self.portManager = FinancialAgent(
            agentRole="Impartial Portfolio Manager",
            tools=pmTools,
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
                toolMap["calculateDistFromCurrPrice"],
                readJournalTool,
                readPortfolioJournalAliasTool,
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
                toolMap["calculateDistFromCurrPrice"],
                readJournalTool,
                readPortfolioJournalAliasTool,
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
                toolMap["calculateDistFromCurrPrice"],
                readJournalTool,
                readPortfolioJournalAliasTool,
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
                toolMap["calculateDistFromCurrPrice"],
                readJournalTool,
                readPortfolioJournalAliasTool,
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

    def updateAgentsTimestamp(self, timestamp: pd.Timestamp):
        self.timestamp = timestamp
        dateStr = timestamp.strftime("%Y-%m-%d")
        for agent in self.agentsList:
            agent.dateStr = dateStr
            agent.messageHistory.clear()


    # Get lots of metric data for the current simulation state, including portfolio value, returns, positions, and performance metrics
    def getSimulationMetrics(self, config: AgentPortfolioSimulationConfig, sp500Df: pd.DataFrame) -> Dict[str, Any]:
        if not self.marketSim:
            return {}
        username = "AgentPortfolio" if "AgentPortfolio" in self.marketSim.userPortfolios else "AgentFund"
        pfDict = self.marketSim.getPortfolioValueAtCurrentDate(username)
        if not pfDict:
            return {}

        currentDateTs = self.marketSim.currentDate
        currentDateStr = currentDateTs.strftime("%Y-%m-%d")
        totalVal = float(pfDict["totalValue"])
        cashVal = float(pfDict["cash"])
        stockVal = totalVal - cashVal
        capital = config.initialCapital

        totalReturnDollar = totalVal - capital
        totalReturnPct = ((totalVal / capital) - 1.0) * 100.0 if capital > 0 else 0.0

        spCurrentPrice = getSp500PriceOnDate(currentDateTs, sp500Df)
        sp500Val = capital * (spCurrentPrice / self.sp500InitialPrice) if self.sp500InitialPrice > 0 else capital
        sp500ReturnDollar = sp500Val - capital
        sp500ReturnPct = ((sp500Val / capital) - 1.0) * 100.0 if capital > 0 else 0.0

        alphaPct = totalReturnPct - sp500ReturnPct

        # Calculate Sharpe and Max Drawdown from history
        sharpeRatio = 0.0
        maxDrawdownPct = 0.0
        if len(self.portfolioHistoryPoints) >= 2:
            vals = [p["y"] for p in self.portfolioHistoryPoints]
            series = pd.Series(vals)
            runningMax = series.cummax()
            drawdown = (series / runningMax) - 1.0
            maxDrawdownPct = float(abs(drawdown.min())) * 100.0

            # Calculate sharpe ratio based on excess returns over the 2-year treasury yield
            if len(self.portfolioHistoryPoints) >= 3:
                try:
                    portDf = pd.DataFrame({
                        "date": pd.to_datetime([p["x"] for p in self.portfolioHistoryPoints]),
                        "close": [float(p["y"]) for p in self.portfolioHistoryPoints],
                    })
                    currentDateTs = self.marketSim.currentDate if self.marketSim else None
                    treasDf = self.toolRegistry.dataProviders.macro.getSeries(
                        MacroSeries.TREAS_2Y,
                        startDate=pd.to_datetime(self.portfolioHistoryPoints[0]["x"]),
                        endDate=currentDateTs,
                    )
                    sharpeRatio = float(calculateSharpeRatio(portDf, treasDf))
                except Exception:
                    sharpeRatio = 0.0

        # Build active positions list
        positionsList = []
        portfolioObj = self.marketSim.userPortfolios.get(username)
        if portfolioObj:
            for ticker, pos in portfolioObj.positions.items():
                curPrice = self.marketSim.getCurrentPrice(ticker)
                if pd.isna(curPrice) or curPrice <= 0:
                    curPrice = pos.averagePrice
                posVal = pos.quantity * curPrice
                posWeight = (posVal / totalVal * 100.0) if totalVal > 0 else 0.0
                posPnlDollar = (curPrice - pos.averagePrice) * pos.quantity
                posPnlPct = ((curPrice / pos.averagePrice) - 1.0) * 100.0 if pos.averagePrice > 0 else 0.0

                profile = self.toolRegistry.dataProviders.tickers.getTickerProfile(ticker)
                companyName = profile.name if profile and profile.name else ticker
                sector = profile.sector if profile and profile.sector else "General Equities"

                positionsList.append({
                    "ticker": ticker,
                    "companyName": companyName,
                    "sector": sector,
                    "shares": round(float(pos.quantity), 4),
                    "avgPrice": round(float(pos.averagePrice), 2),
                    "currentPrice": round(float(curPrice), 2),
                    "currentValue": round(float(posVal), 2),
                    "weightPct": round(float(posWeight), 2),
                    "pnlDollar": round(float(posPnlDollar), 2),
                    "pnlPct": round(float(posPnlPct), 2)
                })

        positionsList.sort(key=lambda x: x["currentValue"], reverse=True)

        systemLogs = []
        if portfolioObj and hasattr(portfolioObj, "mainLog"):
            systemLogs = [entry.toDict() if hasattr(entry, "toDict") else entry for entry in portfolioObj.mainLog]

        # Build clean milestones list to ensure no nested circular references
        cleanMilestones = []
        for m in self.milestonesList:
            mCopy = dict(m)
            if "sessionMetrics" in mCopy and isinstance(mCopy["sessionMetrics"], dict):
                cleanMetrics = dict(mCopy["sessionMetrics"])
                cleanMetrics.pop("milestones", None)
                mCopy["sessionMetrics"] = cleanMetrics
            cleanMilestones.append(mCopy)

        friendlyDate = formatDateFriendly(currentDateTs)
        return {
            "simStatus": self.currentSimStatus,
            "currentMilestoneId": self.currentMilestoneId,
            "currentMilestoneLabel": self.currentMilestoneLabel,
            "currentDate": currentDateStr,
            "currentDateFormatted": friendlyDate,
            "currentDateFriendly": friendlyDate,
            "totalValue": round(totalVal, 2),
            "cashValue": round(cashVal, 2),
            "stockValue": round(stockVal, 2),
            "totalReturnDollar": round(totalReturnDollar, 2),
            "totalReturnPct": round(totalReturnPct, 2),
            "sp500Value": round(sp500Val, 2),
            "sp500ReturnDollar": round(sp500ReturnDollar, 2),
            "sp500ReturnPct": round(sp500ReturnPct, 2),
            "alphaPct": round(alphaPct, 2),
            "sharpeRatio": round(sharpeRatio, 2),
            "maxDrawdownPct": round(maxDrawdownPct, 2),
            "positions": positionsList,
            "portfolioPoints": list(self.portfolioHistoryPoints),
            "sp500Points": list(self.sp500HistoryPoints),
            "rebalanceHistory": list(self.rebalanceHistory),
            "milestones": cleanMilestones,
            "systemLogs": systemLogs
        }


    def emitSimulationState(self, config: AgentPortfolioSimulationConfig, sp500Df: pd.DataFrame):
        state = self.getSimulationMetrics(config, sp500Df)
        emitEvent("agentSimStateUpdate", state)


    def capturePostSessionMetrics(self, config: AgentPortfolioSimulationConfig, sp500Df: pd.DataFrame) -> Dict[str, Any]:
        """Captures complete portfolio state, holdings, and sliced history as of session completion."""
        metrics = self.getSimulationMetrics(config, sp500Df)
        metrics.pop("milestones", None)
        currentDateStr = self.marketSim.currentDate.strftime("%Y-%m-%d") if self.marketSim else config.startDateStr
        metrics["portfolioPoints"] = [p for p in self.portfolioHistoryPoints if p.get("x", "") <= currentDateStr]
        metrics["sp500Points"] = [p for p in self.sp500HistoryPoints if p.get("x", "") <= currentDateStr]
        metrics["rebalanceHistory"] = list(self.rebalanceHistory)
        return metrics


    # Formats the current holdings into a human-readable string for display in the UI
    def formatCurrentHoldingsPrompt(self) -> str:
        if not self.marketSim:
            return "No holdings currently active."
        username = "AgentPortfolio" if "AgentPortfolio" in self.marketSim.userPortfolios else "AgentFund"
        portfolioObj = self.marketSim.userPortfolios.get(username)
        if not portfolioObj or not portfolioObj.positions:
            return "Portfolio is currently 100% Cash."

        totalVal = float(self.marketSim.getPortfolioValueAtCurrentDate(username)["totalValue"])
        lines = []
        for ticker, pos in portfolioObj.positions.items():
            curPrice = self.marketSim.getCurrentPrice(ticker)
            if pd.isna(curPrice) or curPrice <= 0:
                curPrice = pos.averagePrice
            posVal = pos.quantity * curPrice
            weightPct = (posVal / totalVal * 100.0) if totalVal > 0 else 0.0
            pnlPct = ((curPrice / pos.averagePrice) - 1.0) * 100.0 if pos.averagePrice > 0 else 0.0

            profile = self.toolRegistry.dataProviders.tickers.getTickerProfile(ticker)
            sector = profile.sector if profile and profile.sector else "General"
            lines.append(
                f"- {ticker} ({profile.name if profile else ticker}) | Sector: {sector} | "
                f"Shares: {pos.quantity:,.2f} | Avg Cost: ${pos.averagePrice:,.2f} | "
                f"Current Price: ${curPrice:,.2f} | Value: ${posVal:,.2f} ({weightPct:.1f}%) | P&L: {pnlPct:+.1f}%"
            )
        return "\n".join(lines)

    # Inception = the portfolio creation process, onetime
    def runInceptionBoardroom(self, config: AgentPortfolioSimulationConfig, sp500Df: pd.DataFrame, startFormatted: str, promptArgs: Dict[str, Any]) -> None:
        # Run the inception boardroom process (not in a FSM)

        # Phase 1: Macro Environment Analysis
        self.currentSimStatus = f"{startFormatted}: Stage 1"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(1, "Macro Environment Analysis", BoardroomPace.COMPLETE)
        macroPrompt = (
            f"Task: Conduct top-down macroeconomic analysis to guide initial portfolio inception for a {promptArgs['initialCapital']} portfolio.\n"
            f"Judge regimes on forward fundamentals, not yesterday's price noise.\n"
            f"Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n\n"
            f"1. Use 'fetchAllSectorRankings', 'fetchMacroContext', and 'fetchMacroNews' to analyse market regime, rates, and leading sectors.\n"
            f"2. Output your economic indicator table, macro narrative, and market regime classification."
        )
        macroRaw, _ = self.macroAnalyst.analyseAndReply(
            incomingMessage=macroPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=True,
            modeOverride="PortfolioCreation"
        )

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 2: Sector Allocation Analysis (Bull & Bear concurrently)
        self.currentSimStatus = f"{startFormatted}: Stage 2"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(2, "Sector Allocation Analysis", BoardroomPace.COMPLETE)
        bullPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: Propose an aggressive, growth-oriented sector allocation for a {promptArgs['initialCapital']} portfolio.\n"
            f"Champion durable compounding over fleeting momentum, judging each sector across upside, sideways and stress paths.\n"
            f"Mandatory Constraints:\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. Use 'fetchAllSectorsAnalysis' to evaluate sector momentum.\n"
            f"2. Propose sector percentage weights totaling 100%."
        )
        bearPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Task: Propose a defensive, risk-mitigated sector allocation for a {promptArgs['initialCapital']} portfolio.\n"
            f"Demand evidence of resilience through the cycle rather than reacting to recent price softness alone.\n"
            f"Mandatory Constraints:\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"1. Use 'fetchAllSectorsAnalysis' to evaluate sector stability.\n"
            f"2. Propose sector percentage weights totaling 100%."
        )
        (bullSectorRaw, _), (bearSectorRaw, _) = self.runAgentsConcurrently(
            lambda: self.bullAnalyst.analyseAndReply(bullPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioCreation"),
            lambda: self.bearAnalyst.analyseAndReply(bearPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioCreation")
        )

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 3: Sector Allocation Decision
        self.currentSimStatus = f"{startFormatted}: Stage 3"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(3, "Sector Allocation Decision", BoardroomPace.COMPLETE)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")
        pmSectorPrompt = (
            f"Macro Context:\n{macroRaw}\n\n"
            f"Bullish Sector Proposal:\n{bullSectorRaw}\n\n"
            f"Bearish Sector Proposal:\n{bearSectorRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile Bullish and Bearish proposals to determine "
            f"the initial executive sector allocation for this {promptArgs['initialCapital']} portfolio.\n"
            f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
            f"Mandatory Constraints:\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n"
            f"- Allocations must sum to approximately 100.0%.\n\n"
            f"1. You may call 'fetchAllSectorsAnalysis' to inspect sector metrics.\n"
            f"2. Execute the 'confirmSectorAllocation' tool with your 'sectorAllocations' dictionary and executive 'rationale'."
        )
        _, _ = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=10,
            requireInitialTools=True,
            modeOverride="PortfolioCreation"
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation,
            "milestoneId": self.currentMilestoneId
        })

        confirmed = self.confirmedSectorAllocation.get("confirmedAllocation", self.confirmedSectorAllocation) if self.confirmedSectorAllocation else {}
        sectorsDict = confirmed.get("sectorAllocations", {})
        confirmedSectorsText = "\n".join([f"- {secInfo.get('sector', secKey)} ({secKey}): {int(round(float(secInfo.get('allocationPct', 0))))}%" for secKey, secInfo in sectorsDict.items()])

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 4: Stock Scouting (Growth & Value Hunters concurrently)
        self.currentSimStatus = f"{startFormatted}: Stage 4"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(4, "Stock Scouting", BoardroomPace.COMPLETE)
        growthPrompt = (
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Growth Stock Hunter, scout high-conviction growth equities within confirmed sectors.\n"
            f"Favour forward earnings power and catalyst runway over a single soft quarter, weighing recovery and acceleration scenarios.\n"
            f"Constraints: Target stock count: {promptArgs['targetStockCount']}. Max stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"1. Use 'fetchStocksInSector' (style='growth') for confirmed sectors.\n"
            f"2. Use 'fetchBatchStockOverviews' on top conviction candidates."
        )
        valuePrompt = (
            f"Confirmed Portfolio Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Task: As the Defensive Stock Hunter, scout high-conviction defensive and dividend equities within confirmed sectors.\n"
            f"Require proven solvency and margin of safety through the cycle, tolerating modest underperformance where the balance sheet endures.\n"
            f"Constraints: Target stock count: {promptArgs['targetStockCount']}. Max stock allocation: {promptArgs['maxStockAllocation']}.\n"
            f"1. Use 'fetchStocksInSector' (style='defensive') for confirmed sectors.\n"
            f"2. Use 'fetchBatchStockOverviews' on top conviction candidates."
        )
        (growthRaw, _), (valueRaw, _) = self.runAgentsConcurrently(
            lambda: self.growthHunter.analyseAndReply(growthPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioCreation"),
            lambda: self.valueHunter.analyseAndReply(valuePrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioCreation")
        )

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 5: Stock Allocation Proposals (Aggressive & Conservative Risk Analysts concurrently)
        self.currentSimStatus = f"{startFormatted}: Stage 5"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(5, "Stock Allocation Proposals", BoardroomPace.COMPLETE)
        aggProposalPrompt = (
            f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates:\n{growthRaw}\n\n"
            f"Defensive Candidates:\n{valueRaw}\n\n"
            f"Task: Review candidates from both hunters and construct an aggressive stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Press for genuine upside optionality while respecting evidence; do not churn holdings on noise.\n"
            f"Group stock proposals under confirmed sectors, assigning whole integer weights summing to 100% per sector."
        )
        consProposalPrompt = (
            f"Confirmed Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"Growth Candidates:\n{growthRaw}\n\n"
            f"Defensive Candidates:\n{valueRaw}\n\n"
            f"Task: Review candidates from both hunters and construct a conservative stock allocation proposal for {promptArgs['initialCapital']}.\n"
            f"Protect compounding first: prefer seasoned cash generators unless a challenger clearly improves safety.\n"
            f"Group stock proposals under confirmed sectors, assigning whole integer weights summing to 100% per sector."
        )
        (aggProposalRaw, _), (consProposalRaw, _) = self.runAgentsConcurrently(
            lambda: self.aggRiskAnalyst.analyseAndReply(aggProposalPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=False, modeOverride="PortfolioCreation"),
            lambda: self.consRiskAnalyst.analyseAndReply(consProposalPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=False, modeOverride="PortfolioCreation")
        )

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 6: Final Executive Decision
        self.currentSimStatus = f"{startFormatted}: Stage 6"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(6, "Final Executive Decision", BoardroomPace.COMPLETE)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")

        pmFinalPrompt = (
            f"Initial Capital: {promptArgs['initialCapital']}\n"
            f"Target Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"Growth Hunter Candidates:\n{growthRaw}\n\n"
            f"Value Hunter Candidates:\n{valueRaw}\n\n"
            f"Aggressive Risk Proposal:\n{aggProposalRaw}\n\n"
            f"Conservative Risk Proposal:\n{consProposalRaw}\n\n"
            f"Task: As the Impartial Portfolio Manager, reconcile proposals to construct the definitive inception portfolio.\n"
            f"Balance conviction with humility: size positions for bull, base and bear paths rather than a single forecast.\n"
            f"Mandatory: Call 'confirmPortfolioAllocation' with your allocated stock positions across "
            f"confirmed sectors, 'portfolioRationale', and 'initialCapital'={config.initialCapital}.\n"
            f"You may also call 'recordJournalEntry' with your 30-50 word executive rationale summarizing portfolio inception."
        )
        pmResponse, _ = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=10,
            requireInitialTools=True,
            modeOverride="PortfolioCreation"
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        self.stage6Decisions.append({
            "milestoneId": self.currentMilestoneId,
            "milestoneLabel": self.currentMilestoneLabel,
            "date": config.startDateStr,
            "type": "inception",
            "pmResponse": pmResponse or ""
        })

        emitEvent("portfolioCreated", {
            "stageNum": 6,
            "confirmedPortfolio": self.confirmedPortfolioAllocation,
            "milestoneId": self.currentMilestoneId
        })


    # Rebalance at milestone boardroom process, for periodic portfolio reviews and rebalancing
    def runRebalanceAtMilestoneBoardroom(self, config: AgentPortfolioSimulationConfig, sp500Df: pd.DataFrame, currentDateStr: str, currentDateFormatted: str,
        friendlyDate: str, promptArgs: Dict[str, Any], isExtended: bool, currentHoldingsStr: str, macroRaw: str, currentPortfolioTotalVal: float) -> None:

        pace = BoardroomPace.COMPLETE if isExtended else BoardroomPace.FAST

        if isExtended:
            # Phase 2: Sector Allocation Analysis (Bull & Bear Analysts)
            self.currentSimStatus = f"{friendlyDate}: Stage 2"
            self.emitSimulationState(config, sp500Df)
            self.emitNewPhase(2, "Sector Allocation Analysis", pace)
            bullSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\nTask: As the Bullish Analyst, identify leading growth and expansion sectors. "
                f"Weigh whether incumbents can recover and compound before urging rotation on recent softness."
            )
            bearSectorPrompt = (
                f"Macro Context:\n{macroRaw}\n\nTask: As the Bearish Analyst, identify vulnerable sectors facing headwinds. "
                f"Distinguish temporary price weakness from genuine solvency deterioration before urging exits."
            )
            (bullRaw, _), (bearRaw, _) = self.runAgentsConcurrently(
                lambda: self.bullAnalyst.analyseAndReply(bullSectorPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioRebalancing"),
                lambda: self.bearAnalyst.analyseAndReply(bearSectorPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioRebalancing")
            )
            sectorContextForPM = f"Bullish Analysis:\n{bullRaw}\n\nBearish Analysis:\n{bearRaw}"
        else:
            sectorContextForPM = f"Macro Context:\n{macroRaw}"

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 3: Sector Rebalancing Decision
        self.currentSimStatus = f"{friendlyDate}: Stage 3"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(3, "Sector Rebalancing Decision", pace)
        confirmSectorTool = self.toolRegistry.getTool("confirmSectorAllocation")
        pmSectorPrompt = (
            f"{sectorContextForPM}\n\n"
            f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
            f"Task: As the Impartial Portfolio Manager, determine the target rebalanced sector allocations.\n"
            f"Balance conviction with humility, sizing for bull, base and bear paths rather than a single forecast.\n"
            f"Mandatory Constraints:\n"
            f"- Rebalance Amount Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Strategic Allocation Bias: {promptArgs['allocationBiasGuidance']}\n"
            f"- {promptArgs['sectorDiversityRule']}\n"
            f"- Max single sector allocation: {promptArgs['maxSectorAllocation']}\n\n"
            f"Mandatory: Call 'confirmSectorAllocation' with your target sector allocation dictionary."
        )
        _, _ = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmSectorPrompt,
            mandatedToolName="confirmSectorAllocation",
            config=config,
            subrole="sector",
            maxRetries=10,
            requireInitialTools=True,
            modeOverride="PortfolioRebalancing"
        )

        if confirmSectorTool and confirmSectorTool.toolLog:
            self.confirmedSectorAllocation = confirmSectorTool.toolLog[-1]

        emitEvent("sectorAllocationConfirmed", {
            "stageNum": 3,
            "confirmedAllocation": self.confirmedSectorAllocation,
            "milestoneId": self.currentMilestoneId
        })

        confirmed = self.confirmedSectorAllocation.get("confirmedAllocation", self.confirmedSectorAllocation) if self.confirmedSectorAllocation else {}
        sectorsDict = confirmed.get("sectorAllocations", {})
        confirmedSectorsText = "\n".join([f"- {secInfo.get('sector', secKey)} ({secKey}): {int(round(float(secInfo.get('allocationPct', 0))))}%" for secKey, secInfo in sectorsDict.items()])

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 4: Stock Scouting
        self.currentSimStatus = f"{friendlyDate}: Stage 4"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(4, "Stock Holdings Audit & Scouting", pace)
        growthPrompt = (
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
            f"Task: As the Growth Stock Hunter, evaluate existing growth holdings and scout high-conviction momentum/growth replacements.\n"
            f"Think as a growth owner: do not recommend replacing an incumbent on a modest dip alone; require a clearly superior forward path.\n"
            f"PORTFOLIO CONTINUITY MANDATE: Your shortlisted candidate tickers fed into 'fetchBatchStockOverviews' MUST include ALL "
            f"of the companies that exist right now inside CURRENT HOLDINGS above. Extract every ticker dynamically from that list at run time.\n"
            f"Constraints: Target stock count: {promptArgs['targetStockCount']}. Max stock allocation: {promptArgs['maxStockAllocation']}."
        )
        valuePrompt = (
            f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
            f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
            f"Task: As the Defensive Stock Hunter, evaluate defensive holdings and scout margin-of-safety replacements.\n"
            f"Think as a defensive steward: retain sound incumbents through soft patches unless overview metrics show the cushion is gone.\n"
            f"PORTFOLIO CONTINUITY MANDATE: Your shortlisted candidate tickers fed into 'fetchBatchStockOverviews' MUST include ALL "
            f"of the companies that exist right now inside CURRENT HOLDINGS above. Extract every ticker dynamically from that list at run time.\n"
            f"Constraints: Target stock count: {promptArgs['targetStockCount']}. Max stock allocation: {promptArgs['maxStockAllocation']}."
        )
        (growthRaw, _), (valueRaw, _) = self.runAgentsConcurrently(
            lambda: self.growthHunter.analyseAndReply(growthPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioRebalancing"),
            lambda: self.valueHunter.analyseAndReply(valuePrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=True, modeOverride="PortfolioRebalancing")
        )

        if isExtended:
            # Phase 5: Stock Allocation Proposals (Risk Analysts)
            self.currentSimStatus = f"{friendlyDate}: Stage 5"
            self.emitSimulationState(config, sp500Df)
            self.emitNewPhase(5, "Stock Allocation Proposals", pace)
            aggPrompt = (
                f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
                f"Growth Candidates from Hunter:\n{growthRaw}\n\n"
                f"Defensive Candidates from Hunter:\n{valueRaw}\n\n"
                f"Task: Review current holdings and scouted candidates from both hunters, then construct an assertive alpha-maximising stock allocation proposal.\n"
                f"Seek alpha through disciplined upgrades, not restless trading; model recovery as well as momentum for each incumbent.\n"
                f"Group stock proposals under confirmed sectors, assigning whole integer weights summing to 100% per sector."
            )
            consPrompt = (
                f"Target Rebalanced Sector Allocations (LOCKED):\n{confirmedSectorsText}\n\n"
                f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
                f"Growth Candidates from Hunter:\n{growthRaw}\n\n"
                f"Defensive Candidates from Hunter:\n{valueRaw}\n\n"
                f"Task: Review current holdings and scouted candidates from both hunters, then construct a risk-controlled defensive stock allocation proposal.\n"
                f"Default to keeping proven incumbents; demand clear evidence of impaired safety before endorsing turnover.\n"
                f"Group stock proposals under confirmed sectors, assigning whole integer weights summing to 100% per sector."
            )
            (aggRaw, _), (consRaw, _) = self.runAgentsConcurrently(
                lambda: self.aggRiskAnalyst.analyseAndReply(aggPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=False, modeOverride="PortfolioRebalancing"),
                lambda: self.consRiskAnalyst.analyseAndReply(consPrompt, self.toolRegistry, self.timestamp, config, requireInitialTools=False, modeOverride="PortfolioRebalancing")
            )
            scoutContextForPM = f"Growth Scouting:\n{growthRaw}\n\nValue Scouting:\n{valueRaw}\n\nAggressive Proposal:\n{aggRaw}\n\nConservative Proposal:\n{consRaw}"
        else:
            scoutContextForPM = f"Growth Scouting:\n{growthRaw}\n\nValue Scouting:\n{valueRaw}"

        if isStopRequested(): raise SimulationStoppedException()

        # Phase 6: Final Executive Rebalancing Decision
        self.currentSimStatus = f"{friendlyDate}: Stage 6"
        self.emitSimulationState(config, sp500Df)
        self.emitNewPhase(6, "Final Executive Decision", pace)
        confirmPortTool = self.toolRegistry.getTool("confirmPortfolioAllocation")

        pmFinalPrompt = (
            f"Total Rebalance Capital: ${currentPortfolioTotalVal:,.2f}\n"
            f"Target Sector Allocations:\n{confirmedSectorsText}\n\n"
            f"CURRENT HOLDINGS:\n{currentHoldingsStr}\n\n"
            f"Mandatory Constraints:\n"
            f"- Rebalance Amount Mandate: {promptArgs['rebalanceAmountGuidance']}\n"
            f"- Target stock count: {promptArgs['targetStockCount']}\n"
            f"- Max stock allocation: {promptArgs['maxStockAllocation']}\n\n"
            f"{scoutContextForPM}\n\n"
            f"Task: Construct the rebalanced portfolio. Execute 'confirmPortfolioAllocation' with your 'sectorAllocations' "
            f"dictionary, 'portfolioRationale', and 'initialCapital'={currentPortfolioTotalVal:.2f}.\n"
            f"Minimise needless turnover: retain incumbents whose overview metrics remain sound, justifying every exit against the mandate.\n"
            f"You may also call 'recordJournalEntry' with your 30-50 word rationale detailing portfolio shifts."
        )
        pmResponse, _ = self.executeMandatedToolStage(
            agent=self.portManager,
            initialPrompt=pmFinalPrompt,
            mandatedToolName="confirmPortfolioAllocation",
            config=config,
            subrole="decision",
            maxRetries=10,
            requireInitialTools=True,
            modeOverride="PortfolioRebalancing"
        )

        if confirmPortTool and confirmPortTool.toolLog:
            self.confirmedPortfolioAllocation = confirmPortTool.toolLog[-1]

        self.stage6Decisions.append({
            "milestoneId": self.currentMilestoneId,
            "milestoneLabel": self.currentMilestoneLabel,
            "date": currentDateStr,
            "type": "review",
            "pmResponse": pmResponse or ""
        })


    def execute(self, config: AgentPortfolioSimulationConfig):
        if isStopRequested():
            raise SimulationStoppedException("Simulation stop requested before start.")

        config.generateSummaries = False
        self.lastConfig = config
        startDateTs = pd.Timestamp(config.startDateStr, tz=NEW_YORK).normalize()
        endDateTs = pd.Timestamp(config.endDateStr, tz=NEW_YORK).normalize()
        username = "AgentPortfolio"

        # Initialise Market Simulation
        self.marketSim = MarketSimulation(
            startDate=config.startDateStr,
            endDate=config.endDateStr,
            tickerDataProvider=self.toolRegistry.dataProviders.tickers,
            dailyPriceProvider=self.toolRegistry.dataProviders.ohlcv
        )
        self.marketSim.initialiseUser(username, initialCash=config.initialCapital)

        # Initialise S&P 500 Benchmark series
        sp500Df = self.toolRegistry.dataProviders.macro.getSeries(MacroSeries.SP500, startDateTs, endDateTs)
        if sp500Df.empty or "close" not in sp500Df.columns:
            sp500Df = self.toolRegistry.dataProviders.macro.loadSeries(MacroSeries.SP500)
            if not sp500Df.empty and "date" in sp500Df.columns:
                spDateCol = pd.to_datetime(sp500Df["date"]).dt.tz_localize(None).dt.normalize()
                simNorm = startDateTs.tz_localize(None).normalize()
                sp500Df = sp500Df[spDateCol >= simNorm].reset_index(drop=True)

        if not sp500Df.empty and "close" in sp500Df.columns:
            sp500Df = sp500Df.sort_values("date").reset_index(drop=True)
            sp500Df["normDate"] = pd.to_datetime(sp500Df["date"]).dt.tz_localize(None).dt.normalize()
            sp500Df = sp500Df.drop_duplicates(subset=["normDate"]).sort_values("normDate").reset_index(drop=True)
            self.sp500InitialPrice = float(sp500Df.iloc[0]["close"])
        else:
            self.sp500InitialPrice = 1.0

        self.journal.clear()
        self.portfolioHistoryPoints.clear()
        self.sp500HistoryPoints.clear()
        self.rebalanceHistory.clear()
        self.milestonesList.clear()
        self.stage6Decisions.clear()
        self.consecutiveSkippedSteps = 0

        # Record inception chart point
        startFormatted = formatDateFriendly(startDateTs)
        self.portfolioHistoryPoints.append({"x": config.startDateStr, "y": float(config.initialCapital)})
        self.sp500HistoryPoints.append({"x": config.startDateStr, "y": float(config.initialCapital)})


        # MILESTONE 0: Inception
        self.currentMilestoneId = "milestone_0"
        self.currentMilestoneLabel = f"{startFormatted} (Inception)"
        setCurrentMilestoneId(self.currentMilestoneId)
        inceptionStages = [
            {"num": 1, "name": "Macro Environment Analysis"},
            {"num": 2, "name": "Sector Allocation Analysis"},
            {"num": 3, "name": "Sector Allocation Decision"},
            {"num": 4, "name": "Stock Scouting"},
            {"num": 5, "name": "Stock Allocation Proposals"},
            {"num": 6, "name": "Final Executive Decision"}
        ]
        self.milestonesList.append({
            "milestoneId": self.currentMilestoneId,
            "label": self.currentMilestoneLabel,
            "date": config.startDateStr,
            "dateFormatted": startFormatted,
            "type": "inception",
            "pace": "complete",
            "stages": inceptionStages
        })

        emitEvent("milestoneStarted", {
            "milestoneId": self.currentMilestoneId,
            "label": self.currentMilestoneLabel,
            "date": config.startDateStr,
            "dateFormatted": startFormatted,
            "type": "inception",
            "pace": "complete",
            "stages": inceptionStages
        })

        self.updateAgentsTimestamp(startDateTs)
        promptArgs = config.getPromptArgs()
        self.runInceptionBoardroom(config, sp500Df, startFormatted, promptArgs)


        # Execute the initial portfolio orders to populate the simulation portfolio with the confirmed inception allocation
        confirmedPort = self.confirmedPortfolioAllocation.get("confirmedPortfolio", self.confirmedPortfolioAllocation) or {}
        positions = confirmedPort.get("positions", [])
        portfolioObj = self.marketSim.userPortfolios.get(username)
        availableCash = float(portfolioObj.cash) if portfolioObj else float(config.initialCapital)

        rawWeights = {}
        for pos in positions:
            ticker = pos.get("ticker")
            if not ticker:
                continue
            weight = float(pos.get("weightPct", 0.0))
            if weight <= 0:
                weight = float(pos.get("dollarAllocation", 0.0))
            rawWeights[ticker] = max(0.0, weight)

        sumWeights = sum(rawWeights.values())
        normWeights = {t: (w / sumWeights) for t, w in rawWeights.items()} if sumWeights > 0 else {}
        tickersList = [t for t, w in normWeights.items() if w > 0]
        allocatedSoFar = 0.0

        for i, ticker in enumerate(tickersList):
            if i == len(tickersList) - 1:
                dollarAlloc = max(0.0, round(availableCash - allocatedSoFar, 2))
            else:
                dollarAlloc = round(availableCash * normWeights[ticker], 2)
                allocatedSoFar += dollarAlloc

            if dollarAlloc > 0:
                order = MarketOrder(ticker, OrderSide.BUY, cashValue=dollarAlloc)
                self.marketSim.addOrder(order, username, executionTime=ExecutionTime.CLOSE)

        # Run day trades to fill inception orders at start date close
        self.marketSim.processDaysTrades(startDateTs)
        if portfolioObj and portfolioObj.cash < 0.01:
            portfolioObj.cash = 0.0

        # Record Journal Entry if not called explicitly
        journalEntries = [e for e in self.journal if e.get("date", "") <= config.startDateStr]
        if not journalEntries:
            summaryText = confirmedPort.get("portfolioRationale") or f"Portfolio Inception successfully launched with {len(positions)} equities across confirmed sectors."
            self.journal.append({
                "entryIndex": len(self.journal) + 1,
                "date": config.startDateStr,
                "milestone": self.currentMilestoneLabel,
                "summary": summaryText[:200],
                "strategicOutlook": config.getAllocationBiasLabel(),
                "actionTaken": f"Inception: Initial deployment of {promptArgs['initialCapital']}"
            })

        self.rebalanceHistory.append({
            "milestoneId": self.currentMilestoneId,
            "date": config.startDateStr,
            "dateFormatted": startFormatted,
            "type": "inception",
            "summary": f"Initial portfolio deployed across confirmed sectors with {len(positions)} equities.",
            "stockCount": len(positions)
        })

        # MAIN SIMULATION LOOP - per milestone review and rebalance
        milestoneIndex = 1
        currentSimDateTs = startDateTs

        while True:
            if isStopRequested():
                raise SimulationStoppedException()

            nextMilestoneDateTs = computeNextMilestoneDate(currentSimDateTs, config.timestep)
            if nextMilestoneDateTs >= endDateTs:
                # The remaining interval to endDate is less than or equal to a full timestep.
                # Stop the review loop so the market simulation advances directly to endDate for Performance Evaluation.
                break

            if currentSimDateTs >= endDateTs:
                break

            print(f"\nAdvancing Market Simulation from {currentSimDateTs.strftime('%Y-%m-%d')} to {nextMilestoneDateTs.strftime('%Y-%m-%d')}...")

            # Advance Market Simulator day-by-day to next milestone date
            stepCount = 0
            while self.marketSim.currentDate < nextMilestoneDateTs:
                if isStopRequested(): 
                    raise SimulationStoppedException()
                
                success = self.marketSim.runNextDay()
                if not success:
                    break

                stepCount += 1
                currDateStr = self.marketSim.currentDate.strftime("%Y-%m-%d")
                currVal = float(self.marketSim.getPortfolioValueAtCurrentDate(username)["totalValue"])
                currSpPrice = getSp500PriceOnDate(self.marketSim.currentDate, sp500Df)
                currSpVal = config.initialCapital * (currSpPrice / self.sp500InitialPrice) if self.sp500InitialPrice > 0 else config.initialCapital

                self.portfolioHistoryPoints.append({"x": currDateStr, "y": round(currVal, 2)})
                self.sp500HistoryPoints.append({"x": currDateStr, "y": round(currSpVal, 2)})

                # Emit progress every few days
                if stepCount % 5 == 0:
                    self.emitSimulationState(config, sp500Df)

            # Update prior milestone's post-session metrics over the advanced interval
            priorMilestoneId = f"milestone_{milestoneIndex - 1}"
            priorMetrics = self.capturePostSessionMetrics(config, sp500Df)
            for m in self.milestonesList:
                if m.get("milestoneId") == priorMilestoneId:
                    m["sessionMetrics"] = priorMetrics
                    emitEvent("milestoneCompleted", {
                        "milestoneId": priorMilestoneId,
                        "label": m.get("label"),
                        "date": m.get("date"),
                        "type": m.get("type"),
                        "sessionMetrics": priorMetrics
                    })
                    break

            # Check if simulation completed or market simulator cannot advance further
            if stepCount == 0 or self.marketSim.currentDate >= endDateTs or (hasattr(self.marketSim, "endDate") and self.marketSim.currentDate >= self.marketSim.endDate):
                self.emitSimulationState(config, sp500Df)
                break

            currentSimDateTs = self.marketSim.currentDate
            currentDateStr = currentSimDateTs.strftime("%Y-%m-%d")
            currentDateFormatted = formatDateFriendly(currentSimDateTs)
            friendlyDate = currentDateFormatted


            # Milestone boardroom process
            self.currentMilestoneId = f"milestone_{milestoneIndex}"
            self.currentMilestoneLabel = f"{friendlyDate} (Review)"
            setCurrentMilestoneId(self.currentMilestoneId)
            self.milestonesList.append({
                "milestoneId": self.currentMilestoneId,
                "label": self.currentMilestoneLabel,
                "date": currentDateStr,
                "dateFormatted": currentDateFormatted,
                "type": "review",
                "pace": "fast",
                "stages": [
                    {"num": 1, "name": "Macro Environment Analysis"},
                    {"num": 3, "name": "Sector Rebalancing Decision"},
                    {"num": 4, "name": "Stock Holdings Audit & Scouting"},
                    {"num": 6, "name": "Final Executive Decision"}
                ]
            })

            # Emit milestone started event with current holdings
            portfolioObj = self.marketSim.userPortfolios.get(username)
            perfState = self.getSimulationMetrics(config, sp500Df)
            currentHoldingsList = []
            if portfolioObj and portfolioObj.positions:
                for ticker, pos in portfolioObj.positions.items():
                    profile = self.toolRegistry.dataProviders.tickers.getTickerProfile(ticker)
                    curPrice = self.marketSim.getCurrentPrice(ticker)
                    if pd.isna(curPrice) or curPrice <= 0:
                        curPrice = pos.averagePrice
                    curVal = float(pos.quantity) * float(curPrice)
                    currentHoldingsList.append({
                        "ticker": ticker,
                        "companyName": profile.name if profile else ticker,
                        "sector": profile.sector if profile else "General",
                        "industry": profile.industry if profile else "General",
                        "quantity": pos.quantity,
                        "currentPrice": curPrice,
                        "value": curVal,
                        "weightPct": (curVal / perfState.get("totalValue", config.initialCapital) * 100.0) if perfState.get("totalValue", 0) > 0 else 0.0
                    })

            emitEvent("milestoneStarted", {
                "milestoneId": self.currentMilestoneId,
                "label": self.currentMilestoneLabel,
                "date": currentDateStr,
                "dateFormatted": currentDateFormatted,
                "type": "review",
                "baselinePositions": currentHoldingsList
            })

            self.updateAgentsTimestamp(currentSimDateTs)
            currentHoldingsStr = self.formatCurrentHoldingsPrompt()

            # Check 3-consecutive-skipped mandate rule
            isMandatedFullRebalance = self.consecutiveSkippedSteps >= 3
            decision = "extendedBalanceRequired" if isMandatedFullRebalance else None

            # Phase 1: Macro Environment Analysis & Portfolio Audit
            self.currentSimStatus = f"{friendlyDate}: Stage 1"
            self.emitSimulationState(config, sp500Df)
            self.emitNewPhase(1, "Macro Environment Analysis", config.boardroomPace)
            macroAuditPrompt = (
                f"Milestone Date: {currentDateStr} ({currentDateFormatted})\n"
                f"Portfolio Capital: ${perfState.get('totalValue', config.initialCapital):,.2f} | "
                f"Return Since Inception: {perfState.get('totalReturnPct', 0.0):+.2f}% vs S&P 500: "
                f"{perfState.get('sp500ReturnPct', 0.0):+.2f}% (Alpha: {perfState.get('alphaPct', 0.0):+.2f}%)\n"
                f"Consecutive Skipped Rebalance Reviews: {self.consecutiveSkippedSteps} / 3\n\n"
                f"CURRENT PORTFOLIO HOLDINGS:\n{currentHoldingsStr}\n\n"
                f"Task: As the Macro Analyst, audit current portfolio holdings against prevailing macroeconomic regime, inflation, yields, and sector leadership.\n"
                f"Judge holdings on forward regime exposure rather than punishing brief price softness.\n"
                f"1. Use 'fetchMacroContext', 'fetchMacroNews', and 'fetchAllSectorRankings' to evaluate the macro environment.\n"
                f"2. You may use 'readJournal' to inspect previous journal entries.\n"
                f"3. Deliver your economic indicator table, macro narrative, and portfolio vulnerability audit."
            )
            macroRaw, _ = self.macroAnalyst.analyseAndReply(
                incomingMessage=macroAuditPrompt,
                toolRegistry=self.toolRegistry,
                timestamp=self.timestamp,
                config=config,
                requireInitialTools=True,
                modeOverride="PortfolioRebalancing"
            )

            if isStopRequested(): raise SimulationStoppedException()

            if isMandatedFullRebalance:
                print(f"3 consecutive timesteps skipped rebalance. Mandating full 6-stage rebalance for {currentDateStr}.")
                rebalanceDecisionReasoning = "Mandatory full 6-stage rebalance triggered due to 3 consecutive skipped review periods."
            else:
                # Phase 1.5: Macro Rebalance Necessity Decision
                self.currentSimStatus = f"{friendlyDate}: Stage 1.5"
                self.emitSimulationState(config, sp500Df)
                self.emitNewPhase(1, "Macro Rebalance Checkpoint", config.boardroomPace)
                decideTool = self.toolRegistry.getTool("decideRebalanceNecessity")
                decidePrompt = (
                    f"Macro Context & Audit:\n{macroRaw}\n\n"
                    f"Task: Execute the 'decideRebalanceNecessity' tool to formally determine whether the portfolio should rebalance at this milestone.\n"
                    f"Valid decision options:\n"
                    f"- 'noBalanceRequired': Existing holdings are performing soundly, macro regime remains stable, "
                    f"and no rebalancing is needed (advances directly to next timestep).\n"
                    f"- 'balanceRequired': Standard 4-stage fast rebalance to adjust sector tilts or replace lagging names.\n"
                    f"- 'extendedBalanceRequired': Full 6-stage comprehensive overhaul due to major macro/cyclical regime rotation."
                )
                _, _ = self.executeMandatedToolStage(
                    agent=self.macroAnalyst,
                    initialPrompt=decidePrompt,
                    mandatedToolName="decideRebalanceNecessity",
                    config=config,
                    subrole="decision",
                    maxRetries=10,
                    requireInitialTools=True,
                    modeOverride="PortfolioRebalancing"
                )

                if decideTool and decideTool.toolLog:
                    lastDecisionRecord = decideTool.toolLog[-1]
                    decision = lastDecisionRecord.get("decision", "balanceRequired")
                    rebalanceDecisionReasoning = lastDecisionRecord.get("reasoning", "")
                else:
                    decision = "balanceRequired"
                    rebalanceDecisionReasoning = "Standard rebalance review performed."

            # Process Rebalance Decision
            if decision == "noBalanceRequired" and not isMandatedFullRebalance:
                self.consecutiveSkippedSteps += 1
                # self.currentSimStatus = f"{friendlyDate}: Rebalance Skipped"
                self.rebalanceHistory.append({
                    "milestoneId": self.currentMilestoneId,
                    "date": currentDateStr,
                    "dateFormatted": currentDateFormatted,
                    "type": "skipped",
                    "summary": f"No rebalance required. {rebalanceDecisionReasoning}",
                    "consecutiveSkipped": self.consecutiveSkippedSteps
                })
                skippedStages = [{"num": 1, "name": "Macro Environment Analysis"}]
                sessionMetrics = self.capturePostSessionMetrics(config, sp500Df)
                for m in self.milestonesList:
                    if m.get("milestoneId") == self.currentMilestoneId:
                        m["sessionMetrics"] = sessionMetrics
                        m["stages"] = skippedStages
                        m["pace"] = "skipped"
                        break

                emitEvent("milestoneCompleted", {
                    "milestoneId": self.currentMilestoneId,
                    "label": self.currentMilestoneLabel,
                    "date": currentDateStr,
                    "type": "skipped",
                    "pace": "skipped",
                    "reasoning": rebalanceDecisionReasoning,
                    "stages": skippedStages,
                    "sessionMetrics": sessionMetrics
                })
                self.emitSimulationState(config, sp500Df)
                milestoneIndex += 1
                continue

            # Full Rebalance execution
            self.consecutiveSkippedSteps = 0
            isExtended = (decision == "extendedBalanceRequired")
            pace = BoardroomPace.COMPLETE if isExtended else BoardroomPace.FAST

            completeStages = [
                {"num": 1, "name": "Macro Environment Analysis"},
                {"num": 2, "name": "Sector Allocation Analysis"},
                {"num": 3, "name": "Sector Rebalancing Decision"},
                {"num": 4, "name": "Stock Holdings Audit & Scouting"},
                {"num": 5, "name": "Stock Allocation Proposals"},
                {"num": 6, "name": "Final Executive Decision"}
            ]
            fastStages = [
                {"num": 1, "name": "Macro Environment Analysis"},
                {"num": 3, "name": "Sector Rebalancing Decision"},
                {"num": 4, "name": "Stock Holdings Audit & Scouting"},
                {"num": 6, "name": "Final Executive Decision"}
            ]
            activeStages = completeStages if isExtended else fastStages

            for m in self.milestonesList:
                if m.get("milestoneId") == self.currentMilestoneId:
                    m["pace"] = "complete" if isExtended else "fast"
                    m["stages"] = activeStages
                    break

            emitEvent("milestoneUpdated", {
                "milestoneId": self.currentMilestoneId,
                "label": self.currentMilestoneLabel,
                "date": currentDateStr,
                "dateFormatted": currentDateFormatted,
                "type": "review",
                "pace": "complete" if isExtended else "fast",
                "decision": decision,
                "stages": activeStages
            })

            # Milestone Boardroom process (Phases 2-6)
            currentPortfolioTotalVal = float(self.marketSim.getPortfolioValueAtCurrentDate(username)["totalValue"])
            self.runRebalanceAtMilestoneBoardroom(
                config=config,
                sp500Df=sp500Df,
                currentDateStr=currentDateStr,
                currentDateFormatted=currentDateFormatted,
                friendlyDate=friendlyDate,
                promptArgs=promptArgs,
                isExtended=isExtended,
                currentHoldingsStr=currentHoldingsStr,
                macroRaw=macroRaw,
                currentPortfolioTotalVal=currentPortfolioTotalVal
            )

            portfolioObj = self.marketSim.userPortfolios.get(username)
            preRebalanceHoldingsList = []
            if portfolioObj and portfolioObj.positions:
                for ticker, pos in portfolioObj.positions.items():
                    profile = self.toolRegistry.dataProviders.tickers.getTickerProfile(ticker)
                    curPrice = self.marketSim.getCurrentPrice(ticker)
                    if pd.isna(curPrice) or curPrice <= 0:
                        curPrice = pos.averagePrice
                    curVal = float(pos.quantity) * float(curPrice)
                    preRebalanceHoldingsList.append({
                        "ticker": ticker,
                        "companyName": profile.name if profile else ticker,
                        "sector": profile.sector if profile else "General",
                        "industry": profile.industry if profile else "General",
                        "quantity": pos.quantity,
                        "currentPrice": curPrice,
                        "value": curVal,
                        "weightPct": (curVal / currentPortfolioTotalVal * 100.0) if currentPortfolioTotalVal > 0 else 0.0
                    })

            emitEvent("portfolioRebalanced", {
                "stageNum": 6,
                "confirmedPortfolio": self.confirmedPortfolioAllocation,
                "milestoneId": self.currentMilestoneId,
                "baselinePositions": preRebalanceHoldingsList
            })

            # Trade execution ... sells first then buys using all cash proceeds (which may have increased from dividends or mergers)
            self.currentSimStatus = f"{friendlyDate}: Executing Trades"
            self.emitSimulationState(config, sp500Df)

            confirmedPort = self.confirmedPortfolioAllocation.get("confirmedPortfolio", self.confirmedPortfolioAllocation) or {}
            targetPositions = confirmedPort.get("positions", [])
            portfolioObj = self.marketSim.userPortfolios.get(username)
            currentHoldings = dict(portfolioObj.positions) if portfolioObj else {}
            cashBeforeSale = float(portfolioObj.cash) if portfolioObj else 0.0

            heldValues = {}
            for ticker, pos in currentHoldings.items():
                curPrice = self.marketSim.getCurrentPrice(ticker)
                if pd.isna(curPrice) or curPrice <= 0:
                    curPrice = pos.averagePrice
                heldValues[ticker] = float(pos.quantity) * float(curPrice)

            preRebalanceTotalVal = sum(heldValues.values()) + cashBeforeSale

            rawWeights = {}
            for pos in targetPositions:
                ticker = pos.get("ticker")
                if not ticker:
                    continue
                weight = float(pos.get("weightPct", 0.0))
                if weight <= 0:
                    weight = float(pos.get("dollarAllocation", 0.0))
                rawWeights[ticker] = max(0.0, weight)

            # Calculate the normalised target weights including 
            sumWeights = sum(rawWeights.values())
            targetWeightDict = {t: (w / sumWeights) for t, w in rawWeights.items()} if sumWeights > 0 else {}
            sellsExecuted = []
            buysExecuted = []

            # 1. Execute Sells
            for ticker, pos in list(currentHoldings.items()):
                curVal = heldValues.get(ticker, 0.0)
                curPrice = self.marketSim.getCurrentPrice(ticker)
                if pd.isna(curPrice) or curPrice <= 0:
                    curPrice = pos.averagePrice

                targetWeight = targetWeightDict.get(ticker, 0.0)
                idealTargetVal = targetWeight * preRebalanceTotalVal

                if targetWeight <= 0.0:
                    # Liquidate entire holding
                    order = MarketOrder(ticker, OrderSide.SELL, quantity=-1)
                    self.marketSim.addOrder(order, username, executionTime=ExecutionTime.CLOSE)
                    sellsExecuted.append(f"Sold 100% of {ticker} (${curVal:,.2f})")
                elif curVal > idealTargetVal:
                    # Trim holding
                    dollarToSell = curVal - idealTargetVal
                    qtyToSell = dollarToSell / curPrice if curPrice > 0 else 0
                    if qtyToSell > 0:
                        order = MarketOrder(ticker, OrderSide.SELL, quantity=qtyToSell)
                        self.marketSim.addOrder(order, username, executionTime=ExecutionTime.CLOSE)
                        sellsExecuted.append(f"Trimmed {ticker} by ${dollarToSell:,.2f}")

            # Execute all SELL orders first to realise cash proceeds
            self.marketSim.processDaysTrades(currentSimDateTs)

            # 2. Execute Buys (Deploy 100% of available cash including pre-existing dividends + sell proceeds)
            availableCash = float(portfolioObj.cash) if portfolioObj else 0.0
            postSellHeldValues = {}
            if portfolioObj:
                for ticker, pos in portfolioObj.positions.items():
                    curPrice = self.marketSim.getCurrentPrice(ticker)
                    if pd.isna(curPrice) or curPrice <= 0:
                        curPrice = pos.averagePrice
                    postSellHeldValues[ticker] = float(pos.quantity) * float(curPrice)

            postSellTotalVal = sum(postSellHeldValues.values()) + availableCash
            buyDeficits = {}
            for ticker, targetWeight in targetWeightDict.items():
                idealDollar = targetWeight * postSellTotalVal
                currentHeld = postSellHeldValues.get(ticker, 0.0)
                deficit = max(0.0, idealDollar - currentHeld)
                if deficit > 0:
                    buyDeficits[ticker] = deficit

            totalDeficit = sum(buyDeficits.values())
            if totalDeficit > 0 and availableCash > 0:
                scale = availableCash / totalDeficit
                allocatedSoFar = 0.0
                eligibleTickers = [t for t, d in buyDeficits.items() if d > 0]

                for i, ticker in enumerate(eligibleTickers):
                    if i == len(eligibleTickers) - 1:
                        dollarToBuy = max(0.0, round(availableCash - allocatedSoFar, 2))
                    else:
                        dollarToBuy = round(buyDeficits[ticker] * scale, 2)
                        allocatedSoFar += dollarToBuy

                    if dollarToBuy > 0:
                        order = MarketOrder(ticker, OrderSide.BUY, cashValue=dollarToBuy)
                        self.marketSim.addOrder(order, username, executionTime=ExecutionTime.CLOSE)
                        if ticker in currentHoldings and currentHoldings[ticker].quantity > 0:
                            buysExecuted.append(f"Expanded {ticker} by +${dollarToBuy:,.2f}")
                        else:
                            buysExecuted.append(f"Initiated {ticker} with ${dollarToBuy:,.2f}")

            # Execute all BUY orders second using collected cash
            self.marketSim.processDaysTrades(currentSimDateTs)
            if portfolioObj and portfolioObj.cash < 0.01:
                portfolioObj.cash = 0.0

            # Record Journal Entry
            summaryText = confirmedPort.get("portfolioRationale") or f"Portfolio rebalanced into {len(targetPositions)} stocks. Sells: {len(sellsExecuted)}, Buys: {len(buysExecuted)}."
            self.journal.append({
                "entryIndex": len(self.journal) + 1,
                "date": currentDateStr,
                "milestone": self.currentMilestoneLabel,
                "summary": summaryText[:200],
                "strategicOutlook": f"Rebalance Mandate Level {config.rebalanceAmount} executed.",
                "actionTaken": f"Rebalanced {len(targetPositions)} equities ({len(sellsExecuted)} trims/exits, {len(buysExecuted)} additions)"
            })

            self.rebalanceHistory.append({
                "milestoneId": self.currentMilestoneId,
                "date": currentDateStr,
                "dateFormatted": currentDateFormatted,
                "type": "rebalanced",
                "summary": f"Rebalanced into {len(targetPositions)} equities ({len(sellsExecuted)} trims/exits, {len(buysExecuted)} additions).",
                "sells": sellsExecuted,
                "buys": buysExecuted,
                "stockCount": len(targetPositions)
            })

            sessionMetrics = self.capturePostSessionMetrics(config, sp500Df)
            for m in self.milestonesList:
                if m.get("milestoneId") == self.currentMilestoneId:
                    m["sessionMetrics"] = sessionMetrics
                    m["stages"] = activeStages
                    m["pace"] = "complete" if isExtended else "fast"
                    break

            emitEvent("milestoneCompleted", {
                "milestoneId": self.currentMilestoneId,
                "label": self.currentMilestoneLabel,
                "date": currentDateStr,
                "type": "rebalanced",
                "pace": "complete" if isExtended else "fast",
                "stages": activeStages,
                "sellsCount": len(sellsExecuted),
                "buysCount": len(buysExecuted),
                "sessionMetrics": sessionMetrics
            })

            self.emitSimulationState(config, sp500Df)
            milestoneIndex += 1

        # Ensure market simulation has fully advanced to endDateTs before Performance Evaluation
        stepCount = 0
        while self.marketSim.currentDate < endDateTs:
            if isStopRequested():
                raise SimulationStoppedException()
            success = self.marketSim.runNextDay()
            if not success:
                break
            stepCount += 1
            currDateStr = self.marketSim.currentDate.strftime("%Y-%m-%d")
            currVal = float(self.marketSim.getPortfolioValueAtCurrentDate(username)["totalValue"])
            currSpPrice = getSp500PriceOnDate(self.marketSim.currentDate, sp500Df)
            currSpVal = config.initialCapital * (currSpPrice / self.sp500InitialPrice) if self.sp500InitialPrice > 0 else config.initialCapital
            self.portfolioHistoryPoints.append({"x": currDateStr, "y": round(currVal, 2)})
            self.sp500HistoryPoints.append({"x": currDateStr, "y": round(currSpVal, 2)})

        if stepCount > 0:
            self.emitSimulationState(config, sp500Df)

        # Update the latest completed milestone's metrics with performance up to the end date
        if self.milestonesList:
            lastMilestone = self.milestonesList[-1]
            lastMilestoneId = lastMilestone.get("milestoneId")
            interimMetrics = self.capturePostSessionMetrics(config, sp500Df)
            lastMilestone["sessionMetrics"] = interimMetrics
            emitEvent("milestoneCompleted", {
                "milestoneId": lastMilestoneId,
                "label": lastMilestone.get("label"),
                "date": lastMilestone.get("date"),
                "type": lastMilestone.get("type"),
                "sessionMetrics": interimMetrics
            })

        # Dedicated Performance Evaluation Step on final date
        self.runPerformanceEvaluationBoardroom(config, sp500Df, self.marketSim.currentDate)

        # Final completion emission
        self.currentSimStatus = "Simulation Complete"
        finalMetrics = self.getSimulationMetrics(config, sp500Df)
        emitEvent("agentSimCompleted", finalMetrics)
        print(f"\n{Style.BRIGHT}Agent-Driven Portfolio Simulation completed successfully!")


    def runPerformanceEvaluationBoardroom(
        self,
        config: AgentPortfolioSimulationConfig,
        sp500Df: pd.DataFrame,
        evalDateTs: pd.Timestamp
    ) -> None:
        """Executes the final Performance Evaluation milestone where the Portfolio Manager reviews all Stage 6 decisions and full results."""
        if isStopRequested():
            raise SimulationStoppedException()

        evalDateStr = evalDateTs.strftime("%Y-%m-%d")
        evalDateFormatted = formatDateFriendly(evalDateTs)
        milestoneIndex = len(self.milestonesList)
        self.currentMilestoneId = f"milestone_{milestoneIndex}"
        self.currentMilestoneLabel = f"{evalDateFormatted} (Performance Evaluation)"
        setCurrentMilestoneId(self.currentMilestoneId)

        evalStages = [
            {"num": 1, "name": "Comprehensive Performance Evaluation"}
        ]
        self.milestonesList.append({
            "milestoneId": self.currentMilestoneId,
            "label": self.currentMilestoneLabel,
            "date": evalDateStr,
            "dateFormatted": evalDateFormatted,
            "type": "evaluation",
            "pace": "one_shot",
            "stages": evalStages
        })

        emitEvent("milestoneStarted", {
            "milestoneId": self.currentMilestoneId,
            "label": self.currentMilestoneLabel,
            "date": evalDateStr,
            "dateFormatted": evalDateFormatted,
            "type": "evaluation",
            "pace": "one_shot",
            "stages": evalStages
        })

        self.updateAgentsTimestamp(evalDateTs)
        self.currentSimStatus = f"{evalDateFormatted} (Evaluation): Executive Performance Review"
        self.emitSimulationState(config, sp500Df)

        customAgent = [{"role": "Impartial Portfolio Manager", "color": "purple", "name": "Portfolio Manager"}]
        self.emitNewPhase(1, "Comprehensive Performance Evaluation", BoardroomPace.ONE_SHOT, customAgents=customAgent)

        # Collect full performance metrics
        perfMetrics = self.getSimulationMetrics(config, sp500Df)
        startFormatted = formatDateFriendly(pd.Timestamp(config.startDateStr))

        # Format historical Stage 6 decisions (truncated to max 500 words each)
        decisionsTextList = []
        for d in self.stage6Decisions:
            truncatedText = truncateToWords(d.get("pmResponse", ""), 500)
            decisionsTextList.append(
                f"### Milestone: {d.get('milestoneLabel', d.get('date'))} ({d.get('date')})\n"
                f"{truncatedText}"
            )
        decisionsHistoryStr = "\n\n".join(decisionsTextList) if decisionsTextList else "No prior Stage 6 logs available."

        # Format final holdings
        finalHoldingsStr = self.formatCurrentHoldingsPrompt()

        pmEvalPrompt = (
            f"=== COMPREHENSIVE PERFORMANCE EVALUATION ===\n"
            f"Simulation Period: {config.startDateStr} ({startFormatted}) to {evalDateStr} ({evalDateFormatted})\n"
            f"Initial Capital: ${config.initialCapital:,.2f}\n"
            f"Final Portfolio Assets: ${perfMetrics.get('totalValue', config.initialCapital):,.2f} "
            f"(Cash: ${perfMetrics.get('cashValue', 0.0):,.2f}, Equities: ${perfMetrics.get('stockValue', 0.0):,.2f})\n"
            f"Total Return: {perfMetrics.get('totalReturnPct', 0.0):+.2f}% (${perfMetrics.get('totalReturnDollar', 0.0):+,.2f}) vs S&P 500: "
            f"{perfMetrics.get('sp500ReturnPct', 0.0):+.2f}% (${perfMetrics.get('sp500ReturnDollar', 0.0):+,.2f})\n"
            f"Alpha Generated: {perfMetrics.get('alphaPct', 0.0):+.2f}%\n"
            f"Portfolio Sharpe Ratio: {perfMetrics.get('sharpeRatio', 0.0):.2f}\n"
            f"Portfolio Max Drawdown: {perfMetrics.get('maxDrawdownPct', 0.0):.2f}%\n\n"
            f"=== FINAL PORTFOLIO HOLDINGS ===\n"
            f"{finalHoldingsStr}\n\n"
            f"=== HISTORICAL STAGE 6 EXECUTIVE REBALANCING DECISIONS ===\n"
            f"{decisionsHistoryStr}\n\n"
            f"Task: As the Impartial Portfolio Manager, conduct an exhaustive, rigorous post-mortem performance evaluation of the portfolio strategy across time itself.\n"
            f"Structure your response with clear, professional sections:\n"
            f"1. Executive Summary: Overarching assessment of performance vs S&P 500 benchmark and initial investment mandate.\n"
            f"2. Winning Themes & Successful Bets: Highlight top-contributing stock picks, sector allocations, and well-timed entries/expansions.\n"
            f"3. Strategic Shortcomings & Underperformers: Frank analysis of lagging equities, sector drags, mistimed trims, and severe drawdown episodes.\n"
            f"4. Mandate & Risk Discipline Audit: Evaluation of how consistently the portfolio adhered "
            f"to target risk tolerances, diversification constraints, and market regime shifts.\n"
            f"5. Definitive Conclusion: Final verdict on fund performance and lessons learned across the entire simulated lifecycle."
        )

        evalRaw, _ = self.portManager.analyseAndReply(
            incomingMessage=pmEvalPrompt,
            toolRegistry=self.toolRegistry,
            timestamp=self.timestamp,
            config=config,
            requireInitialTools=False,
            modeOverride="PortfolioCreation"
        )

        # Capture post-session metrics for this evaluation milestone
        postMetrics = self.capturePostSessionMetrics(config, sp500Df)
        for m in self.milestonesList:
            if m.get("milestoneId") == self.currentMilestoneId:
                m["sessionMetrics"] = postMetrics
                m["stages"] = evalStages
                break

        emitEvent("milestoneCompleted", {
            "milestoneId": self.currentMilestoneId,
            "label": self.currentMilestoneLabel,
            "date": evalDateStr,
            "type": "evaluation",
            "pace": "one_shot",
            "stages": evalStages,
            "sessionMetrics": postMetrics
        })
        self.emitSimulationState(config, sp500Df)
