import os
import sys
import json
import time
from typing import Dict, Any
import pandas as pd

# Add repository root to path
repoRoot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repoRoot not in sys.path:
    sys.path.insert(0, repoRoot)

from collectors.constants import UTC, NEW_YORK
from llmtools.registry_builder import SCHEMAS, buildToolRegistry


# Default input parameter configurations for every single schema in SCHEMAS
defaultInputsBySchema: Dict[str, Dict[str, Any]] = {
    "empty": {},
    "decideRebalanceNecessity": {
        "decision": "balanceRequired",
        "reasoning": "Macro shift detected with inflation concerns warranting targeted reallocation.",
        "macroShiftDetected": True,
        "urgency": "medium",
    },
    "confirmSectorAllocation": {
        "sectorAllocations": {
            "information_technology": 100.0,
        },
        "rationale": "High-conviction allocation focusing on information technology leadership.",
    },
    "confirmPortfolioAllocation": {
        "sectorAllocations": {
            "information_technology": [
                {
                    "ticker": "AAPL",
                    "perSectorWeight": 100.0,
                    "rationale": "Market leader with resilient services growth and robust balance sheet.",
                }
            ]
        },
        "portfolioRationale": "Executive portfolio construction rationale focusing on industry-leading technological moat, disciplined risk management, capital preservation, solid balance sheets with substantial net cash, robust shareholder returns, and durable long-term free cash flow generation across business cycles.",
    },
    "fetchStocksInSector": {
        "sector": "information_technology",
        "style": "all",
        "limit": 12,
    },
    "sectorQuery": {
        "sectorOrTicker": "information_technology",
    },
    "sectorRanking": {
        "lookback": "1mo",
    },
    "macroNews": {
        "limit": 12,
    },
    "justTicker": {
        "ticker": "AAPL",
    },
    "batchTickers": {
        "tickers": ["AAPL", "GOOGL", "MSFT"],
    },
    "shortInterestHistory": {
        "ticker": "AAPL",
        "months": 6,
    },
    "tickerNews": {
        "ticker": "AAPL",
        "limit": 12,
    },
    "finStatement": {
        "ticker": "AAPL",
        "periodType": "annual",
    },
    "confirmDecisionImmediate": {
        "ticker": "AAPL",
        "rating": "BUY",
        "weighting": "OVERWEIGHT",
        "threeDayTarget": 315.0,
        "twoWeekTarget": 325.0,
    },
    "confirmDecisionShortTerm": {
        "ticker": "AAPL",
        "rating": "BUY",
        "weighting": "OVERWEIGHT",
        "oneMonthTarget": 320.0,
        "threeMonthTarget": 340.0,
    },
    "confirmDecisionMediumTerm": {
        "ticker": "AAPL",
        "rating": "BUY",
        "weighting": "OVERWEIGHT",
        "threeMonthTarget": 340.0,
        "twelveMonthTarget": 380.0,
    },
    "confirmDecisionLongTerm": {
        "ticker": "AAPL",
        "rating": "BUY",
        "weighting": "OVERWEIGHT",
        "twelveMonthTarget": 380.0,
        "threeYearTarget": 450.0,
    },
    "confirmDecisionDistantTerm": {
        "ticker": "AAPL",
        "rating": "BUY",
        "weighting": "OVERWEIGHT",
        "threeYearTarget": 450.0,
        "tenYearTarget": 600.0,
    },
    "stockPriceChange": {
        "ticker": "AAPL",
        "targetPrice": 320.0,
    },
    "pythonCode": {
        "code": "result = 15000 * (1 + 0.055)**10",
    },
    "transferToAgent": {
        "agentRole": "Macro Analyst",
        "transferMessage": "Please analyze recent inflation dynamics and their impact on equity markets.",
    },
}


def findSchemaKey(toolParamSchema: Dict[str, Any]) -> str:
    for schemaKey, schemaVal in SCHEMAS.items():
        if toolParamSchema == schemaVal:
            return schemaKey
    return "empty"


def runAllToolTests():
    toolRegistry = buildToolRegistry()

    testTimestamps = [
        ("2026-08-10", pd.Timestamp("2026-08-10 09:30").tz_localize(NEW_YORK).tz_convert(UTC)),
        ("right now", pd.Timestamp.now(tz=NEW_YORK).tz_convert(UTC)),
    ]

    overallResults = {}

    for timestampLabel, testTimestamp in testTimestamps:
        print(f"\n{'=' * 80}")
        print(f"Executing tools for timestamp: {timestampLabel} ({testTimestamp})")
        print(f"{'=' * 80}\n")

        timestampResults = {"passed": [], "failed": []}

        for toolName, toolObj in toolRegistry.tools.items():
            schemaKey = findSchemaKey(toolObj.paramSchema)
            defaultArgs = defaultInputsBySchema.get(schemaKey, {})

            print(f"--- Running Tool: {toolName} (Schema: {schemaKey}) at {timestampLabel} ---")
            startTime = time.time()
            toolOutput = toolRegistry.executeTool(
                toolName=toolName,
                timestamp=testTimestamp,
                arguments=defaultArgs,
            )
            elapsedTime = time.time() - startTime

            # Print output formatted as json
            print(f"Output for {toolName} [{elapsedTime:.2f}s]:")
            print(json.dumps(toolOutput, indent=2, default=str))
            print()

            isPassed = isinstance(toolOutput, dict) and "error" not in toolOutput

            if isPassed:
                timestampResults["passed"].append(toolName)
                print(f"[STATUS] {toolName}: PASSED")
            else:
                timestampResults["failed"].append((toolName, toolOutput.get("error", "Unknown error")))
                print(f"[STATUS] {toolName}: FAILED - {toolOutput.get('error')}")
            print("-" * 60)

        overallResults[timestampLabel] = timestampResults

    print("\n" + "=" * 80)
    print("TEST SUMMARY ACROSS ALL TIMESTAMPS")
    print("=" * 80)
    for timestampLabel, res in overallResults.items():
        totalCount = len(res["passed"]) + len(res["failed"])
        print(f"\nTimestamp '{timestampLabel}': {len(res['passed'])}/{totalCount} PASSED")
        if res["failed"]:
            print(f"  Failed tools ({len(res['failed'])}):")
            for failedTool, errMsg in res["failed"]:
                print(f"    - {failedTool}: {errMsg}")
        else:
            print("  All tools passed successfully!")

    return overallResults


if __name__ == "__main__":
    runAllToolTests()
