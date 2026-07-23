from dataquery import LRUCache, MacroDataProvider, NewsDataProvider, DailyPriceProvider, \
                      TickerDataProvider, ShortDataProvider, EdgarDataProvider, ForexDataProvider
from collectors.constants import START_DATE, END_DATE

from typing import Any, Dict, List, Callable
import pandas as pd


class DataProviders:
    def __init__(self):
        self.cache = LRUCache(512 * 1024 ** 2)  # 512 MB max size of cache in RAM

        self.tickers = TickerDataProvider()
        self.macro = MacroDataProvider(self.cache)
        self.news = NewsDataProvider(self.cache)
        self.ohlcv = DailyPriceProvider(self.tickers, self.cache)
        self.short = ShortDataProvider(self.cache)
        self.edgar = EdgarDataProvider(self.tickers, self.cache)
        self.forex = ForexDataProvider(START_DATE, END_DATE, self.cache)


class Tool:
    def __init__(self, toolFunction: Callable, toolName: str, toolDescription: str, parameterSchema: Dict[str, Any]):
        self.toolFunction = toolFunction
        self.toolName = toolName
        self.toolDescription = toolDescription
        self.parameterSchema = parameterSchema
        self.toolLog = []

    def getToolSchema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.toolName,
                "description": self.toolDescription,
                "parameters": self.parameterSchema
            }
        }
    
    def executeTool(self, data: DataProviders, timestamp: pd.Timestamp, args: Dict[str, Any]):
        try:
            toolCall = self.toolFunction(self, data, timestamp, **args)
            return toolCall
        except Exception as e:
            error = {"error": f"Uncaught error occurred while executing tool '{self.toolName}': {str(e)}"}
            import traceback
            traceback.print_exc()
            return error


class ToolRegistry:
    def __init__(self):
        self.dataProviders = DataProviders()
        self.tools: Dict[str, Tool] = {}

    def registerTool(self, tool: Tool):
        self.tools[tool.toolName] = tool

    def registerTools(self, tools: List[Tool]):
        for tool in tools:
            self.registerTool(tool)

    def getTool(self, toolName: str):
        return self.tools.get(toolName)
    
    def executeTool(self, toolName: str, timestamp: pd.Timestamp, arguments: Dict[str, Any] = {}):
        tool = self.getTool(toolName)
        if tool:
            return tool.executeTool(self.dataProviders, timestamp, arguments)
        else:
            raise ValueError(f"Tool '{toolName}' not found in registry.")