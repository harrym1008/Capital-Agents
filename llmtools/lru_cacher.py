import pandas as pd

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from llmtools.tool_registry import ToolRegistry


class CachedToolCall:
    def __init__(self, toolName, timestamp, args={}):
        self.toolName = toolName
        self.timestamp = timestamp
        self.args = args


def startPrecacheThread(toolRegistry: ToolRegistry, timestamp: pd.Timestamp, macroTools: bool, ticker: str = None):
    if macroTools:
        precacheThread = threading.Thread(
            target=precacheMacroToolCalls,
            args=(toolRegistry, timestamp)
        )
    else:
        precacheThread = threading.Thread(
            target=precacheTickerSpecificToolCalls,
            args=(toolRegistry, timestamp, ticker)
        )
    precacheThread.start()
    return precacheThread



def precacheMacroToolCalls(toolRegistry: ToolRegistry, timestamp: pd.Timestamp):
    print(f"Starting precache of macro tool calls at timestamp '{timestamp}'...")
    toolCalls = [
        CachedToolCall("fetchMacroContext", timestamp,),
        CachedToolCall("fetchMacroNews", timestamp, {"limit": 12}),
        CachedToolCall("fetchMacroSentimentHistory", timestamp),
        CachedToolCall("fetchAllSectorRankings", timestamp, {"lookback": "1mo"}),
    ]


    with ThreadPoolExecutor(max_workers=len(toolCalls)) as executor:
        futureToTool = {
            executor.submit(
                toolRegistry.executeTool,
                toolCall.toolName,
                toolCall.timestamp,
                toolCall.args
            ): toolCall
            for toolCall in toolCalls
        }

        for future in as_completed(futureToTool):
            toolCall = futureToTool[future]
            try:
                result = future.result()
                strResult = str(result)
                truncatedResult = strResult[:98] + "..." if len(strResult) > 100 else strResult
                print(f"[{toolCall.toolName}] Completed: {truncatedResult}")
            except Exception as exc:
                print(f"[{toolCall.toolName}] Generated an exception: {exc}")


    print(f"Finished precache of tool calls. LRU cache size: {toolRegistry.dataProviders.cache.getCacheUsagePrettyString()}")




def precacheTickerSpecificToolCalls(toolRegistry: ToolRegistry, timestamp: pd.Timestamp, ticker: str):
    print(f"Starting precache of tool calls for ticker '{ticker}' at timestamp '{timestamp}'...")

    toolCalls = [
        CachedToolCall("fetchCompanyProfile", timestamp, {"ticker": ticker}),
        CachedToolCall("fetchCompanyRecentNews", timestamp, {"ticker": ticker, "limit": 12}),
        CachedToolCall("fetchStockPricePerformance", timestamp, {"ticker": ticker}),
        CachedToolCall("calculateDistFromCurrPrice", timestamp, {"ticker": ticker, "targetPrice": 60.0}),

        CachedToolCall("fetchCompanyValuationMetrics", timestamp, {"ticker": ticker}),
        CachedToolCall("fetchIncomeStatement", timestamp, {"ticker": ticker, "periodType": "annual"}),
        CachedToolCall("fetchBalanceSheet", timestamp, {"ticker": ticker, "periodType": "quarterly"}),
        CachedToolCall("fetchCashFlowStatement", timestamp, {"ticker": ticker, "periodType": "annual"}),
        # CachedToolCall("fetchLatest10QSentiment", timestamp, {"ticker": ticker}),
    ]

    with ThreadPoolExecutor(max_workers=len(toolCalls)) as executor:
        futureToTool = {
            executor.submit(
                toolRegistry.executeTool,
                toolCall.toolName,
                toolCall.timestamp,
                toolCall.args
            ): toolCall
            for toolCall in toolCalls
        }

        for future in as_completed(futureToTool):
            toolCall = futureToTool[future]
            try:
                result = future.result()
                strResult = str(result)
                truncatedResult = strResult[:98] + "..." if len(strResult) > 100 else strResult
                print(f"[{toolCall.toolName}] Completed: {truncatedResult}")
            except Exception as exc:
                print(f"[{toolCall.toolName}] Generated an exception: {exc}")


    print(f"Finished precache of tool calls. LRU cache size: {toolRegistry.dataProviders.cache.getCacheUsagePrettyString()}")
