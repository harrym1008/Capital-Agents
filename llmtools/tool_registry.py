from dataquery import LRUCache, MacroDataProvider, NewsDataProvider, DailyPriceProvider, \
                      TickerDataProvider, ShortDataProvider, EdgarDataProvider, ForexDataProvider, SectorDataProvider
from collectors.constants import START_DATE, END_DATE
from collectors.rate_limiter import GlobalRateLimiters

import time
import re
from threading import RLock
from typing import Any, Dict, List, Callable, Optional, Union
import pandas as pd
from ui.ui_hooks import emitEvent, setCurrentCallId, getCurrentCallId


class DataProviders:
    def __init__(self, allowOnlineDownloads: bool = True):
        self.cache = LRUCache(512 * 1024 ** 2)  # 512 MB max size of cache in RAM
        self.rateLimiters = GlobalRateLimiters()

        self.tickers = TickerDataProvider()
        self.macro = MacroDataProvider(self.cache, self.rateLimiters)
        self.news = NewsDataProvider(self.cache, self.rateLimiters)
        self.ohlcv = DailyPriceProvider(self.tickers, self.cache, self.rateLimiters, allowOnlineDownloads=allowOnlineDownloads)
        self.short = ShortDataProvider(self.cache, self.rateLimiters)
        self.edgar = EdgarDataProvider(self.tickers, self.cache, self.rateLimiters.edgarLimiter)
        self.forex = ForexDataProvider(START_DATE, END_DATE, self.cache, self.rateLimiters)
        self.sectors = SectorDataProvider(self.cache, self.rateLimiters, self.macro, self.tickers)

        self.sentimentCache = LRUCache(8 * 1024 ** 2)  # 8 MB max size
        self.sentimentLock = RLock()


class Tool:
    def __init__(self, toolFunction: Callable, toolName: str, toolDescription: str, parameterSchema: Dict[str, Any]):
        self.function = toolFunction
        self.name = toolName
        self.description = toolDescription
        self.paramSchema = parameterSchema
        self.toolLog = []
        self.progressLock = RLock()
        self.lastProgressVal: Dict[str, Any] = {}
        self.lastProgressTime: Dict[str, float] = {}

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

    def getToolSchema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.paramSchema
            }
        }
    
    def executeTool(self, data: DataProviders, timestamp: pd.Timestamp, args: Dict[str, Any], callId: Optional[str] = None):
        if callId:
            setCurrentCallId(callId)
        try:
            toolOutput = self.function(self, data, timestamp, **args)
            if toolOutput is None:
                return {"error": f"Tool '{self.name}' returned None."}
            elif isinstance(toolOutput, str):
                return {"error": toolOutput}        # Assume sole string return values are error messages            
            elif not isinstance(toolOutput, dict):
                return {"result": toolOutput}
            
            return toolOutput
        
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error = {
                "error": f"Uncaught error occurred while executing tool '{self.name}': {str(e)}",
                "traceback": tb
            }
            # raise e
            return error
        finally:
            if callId:
                setCurrentCallId(None)


class ToolRegistry:
    def __init__(self):
        self.dataProviders = DataProviders()
        self.tools: Dict[str, Tool] = {}

    def registerTool(self, tool: Tool):
        self.tools[tool.name] = tool

    def registerTools(self, tools: List[Tool]):
        for tool in tools:
            self.registerTool(tool)

    def getTool(self, toolName: str):
        return self.tools.get(toolName)

    def getToolMap(self):
        return self.tools

    def executeTool(self, toolName: str, timestamp: pd.Timestamp, arguments: Dict[str, Any] = {}, callId: Optional[str] = None):
        tool = self.getTool(toolName)
        if tool:
            return tool.executeTool(self.dataProviders, timestamp, arguments, callId=callId)
        else:
            raise ValueError(f"Tool '{toolName}' not found in registry.")