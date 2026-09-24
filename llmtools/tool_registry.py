from dataquery import LRUCache, MacroDataProvider, NewsDataProvider, DailyPriceProvider, \
                      TickerDataProvider, ShortDataProvider, EdgarDataProvider, ForexDataProvider, SectorDataProvider, SectorLeadersProvider, FinnhubDataProvider
from collectors.constants import START_DATE, END_DATE
from collectors.rate_limiter import GlobalRateLimiters

import time
import re
from threading import RLock
from typing import Any, Dict, List, Callable, Optional, Union
import pandas as pd
from ui.ui_hooks import emitEvent, setCurrentCallId, getCurrentCallId
from llmtools.sources_manager import SourcesManager


# Shared data providers container for tool execution
class DataProviders:
    def __init__(self, allowOnlineDownloads: bool = True):
        # In-memory cache allocated to 512 MB
        self.cache = LRUCache(512 * 1024 ** 2, main=True)
        self.rateLimiters = GlobalRateLimiters()

        self.tickers = TickerDataProvider()
        self.macro = MacroDataProvider(self.cache, self.rateLimiters)
        self.news = NewsDataProvider(self.cache, self.rateLimiters)
        self.ohlcv = DailyPriceProvider(self.tickers, self.cache, self.rateLimiters, allowOnlineDownloads=allowOnlineDownloads)
        self.short = ShortDataProvider(self.cache, self.rateLimiters)
        self.edgar = EdgarDataProvider(self.tickers, self.cache, self.rateLimiters.edgarLimiter)
        self.forex = ForexDataProvider(START_DATE, END_DATE, self.cache, self.rateLimiters)
        self.sectors = SectorDataProvider(self.cache, self.rateLimiters, self.macro, self.tickers)
        self.sectorLeaders = SectorLeadersProvider(self.cache)
        self.finnhub = FinnhubDataProvider(self.cache, self.rateLimiters.finnhubLimiter)

        # In-memory sentiment cache allocated to 8 MB
        self.sentimentCache = LRUCache(8 * 1024 ** 2, main=False)
        self.sentimentLock = RLock()


# Encapsulates single callable agent tool with schema validation and progress tracking
class Tool:
    def __init__(self, toolFunction: Callable, toolName: str, toolDescription: str, parameterSchema: Dict[str, Any], storeIntoSources: bool = False):
        self.function = toolFunction
        self.name = toolName
        self.description = toolDescription
        self.paramSchema = parameterSchema
        self.storeIntoSources = storeIntoSources
        self.registry: Optional['ToolRegistry'] = None
        self.toolLog = []
        self.progressLock = RLock()
        self.lastProgressVal: Dict[str, Any] = {}
        self.lastProgressTime: Dict[str, float] = {}

    # Emit throttled execution progress event to UI websocket
    def updateProgress(self, progress: Union[float, int, str], message: Optional[str] = None, callId: Optional[str] = None):
        with self.progressLock:
            effectiveCallId = callId or getCurrentCallId()
            if not effectiveCallId:
                return

            now = time.time()
            lastVal = self.lastProgressVal.get(effectiveCallId)
            lastTime = self.lastProgressTime.get(effectiveCallId, 0.0)

            if isinstance(progress, (int, float)):
                clamped = max(0.0, min(100.0, float(progress)))
                isTerminal = (clamped >= 100.0 or clamped <= 0.0)
                if not isTerminal and lastVal is not None and isinstance(lastVal, (int, float)):
                    if abs(clamped - lastVal) < 2.0 and (now - lastTime) < 0.1:
                        return
                self.lastProgressVal[effectiveCallId] = clamped
                self.lastProgressTime[effectiveCallId] = now
                emitProgress = clamped
            else:
                strProgress = str(progress).strip()
                if lastVal == strProgress:
                    return

                stageMatch = re.match(r"^(Stage \d+/\d+):\s*([\d\.]+)%", strProgress)
                lastStageMatch = re.match(r"^(Stage \d+/\d+):\s*([\d\.]+)%", str(lastVal)) if lastVal else None

                if stageMatch and lastStageMatch and stageMatch.group(1) == lastStageMatch.group(1):
                    currentPct = float(stageMatch.group(2))
                    lastPct = float(lastStageMatch.group(2))
                    isTerminal = (currentPct >= 100.0 or currentPct <= 0.0)
                    if not isTerminal and abs(currentPct - lastPct) < 2.0 and (now - lastTime) < 0.1:
                        return

                self.lastProgressVal[effectiveCallId] = strProgress
                self.lastProgressTime[effectiveCallId] = now
                emitProgress = strProgress

            emitEvent("toolCallProgress", {
                "toolName": self.name,
                "callId": effectiveCallId,
                "progress": emitProgress,
                "message": message,
                "status": "running"
            })

    # Return OpenAI-compatible function schema definition
    def getToolSchema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.paramSchema
            }
        }
    
    # Execute tool function with error handling and optional source citation tracking
    def executeTool(self, data: DataProviders, timestamp: pd.Timestamp, args: Dict[str, Any], callId: Optional[str] = None, sourcesManager: Optional[SourcesManager] = None, skipSources: bool = False):
        if callId:
            setCurrentCallId(callId)
        try:
            toolOutput = self.function(self, data, timestamp, **args)
            if toolOutput is None:
                return {"error": f"Tool '{self.name}' returned None."}
            elif isinstance(toolOutput, str):
                return {"error": toolOutput}
            elif not isinstance(toolOutput, dict):
                toolOutput = {"result": toolOutput}

            isError = isinstance(toolOutput, dict) and "error" in toolOutput

            if not skipSources and self.storeIntoSources and not isError:
                effectiveSourcesManager = sourcesManager
                if effectiveSourcesManager is None and getattr(self, "registry", None) is not None:
                    effectiveSourcesManager = self.registry.sourcesManager

                if effectiveSourcesManager is not None:
                    citationNumber = effectiveSourcesManager.addSource(self.name, args, toolOutput)
                    if citationNumber is not None:
                        if isinstance(toolOutput, dict):
                            reordered = {
                                "toolCitationNumber": citationNumber,
                                "citationNumber": citationNumber
                            }
                            reordered.update(toolOutput)
                            toolOutput = reordered
                        else:
                            toolOutput["toolCitationNumber"] = citationNumber
                            toolOutput["citationNumber"] = citationNumber
            
            return toolOutput
        
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error = {
                "error": f"Uncaught error occurred while executing tool '{self.name}': {str(e)}",
                "traceback": tb
            }
            return error
        finally:
            if callId:
                setCurrentCallId(None)


# Central registry for agent tools, execution routing, and citation management
class ToolRegistry:
    # Initialise empty tool registry and source manager
    def __init__(self):
        self.dataProviders = DataProviders()
        self.tools: Dict[str, Tool] = {}
        self.sourcesManager = SourcesManager()

    # Register single tool instance in registry map
    def registerTool(self, tool: Tool):
        tool.registry = self
        self.tools[tool.name] = tool

    # Register multiple tool instances sequentially
    def registerTools(self, tools: List[Tool]):
        for tool in tools:
            self.registerTool(tool)

    # Lookup registered tool by name
    def getTool(self, toolName: str):
        return self.tools.get(toolName)

    # Return dictionary of all registered tool handlers
    def getToolMap(self):
        return self.tools

    # Clear execution logs across all registered tools
    def clearToolLogs(self):
        for tool in self.tools.values():
            tool.toolLog.clear()

    # Dispatch tool execution by name with parameter dictionary
    def executeTool(self, toolName: str, timestamp: pd.Timestamp, arguments: Dict[str, Any] = {}, callId: Optional[str] = None, skipSources: bool = False):
        tool = self.getTool(toolName)
        if tool:
            effectiveSourcesManager = None if skipSources else self.sourcesManager
            return tool.executeTool(self.dataProviders, timestamp, arguments, callId=callId, sourcesManager=effectiveSourcesManager, skipSources=skipSources)
        else:
            raise ValueError(f"Tool '{toolName}' not found in registry.")