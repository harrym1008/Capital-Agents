import os
import sys
import io
import math
import traceback

from typing import Dict, Any
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, isLocalDataAvailable, NumberType 


def executePythonCalculation(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, code: str) -> Any:
    oldStdout = sys.stdout
    redirectedOutput = io.StringIO()
    sys.stdout = redirectedOutput

    # NOTE: __import__ is included so that LLM-generated code can use
    # standard "import math" / "import numpy as np" statements inside
    # the sandbox. Without it, any import statement raises ImportError.
    safeGlobals = {
        "__builtins__": {
            "__import__": __import__,
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "hash": hash,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        },
        "math": math,
        "numpy": np,
        "np": np,
        "random": __import__("random"),
        "datetime": __import__("datetime"),
    }

    try:
        strippedCode = code.strip()
        localScope = None     # Defined here so it's accessible in the outer except block

        try:
            resultValue = eval(strippedCode, safeGlobals)
            capturedStdout = redirectedOutput.getvalue()
            output = {
                "success": True,
                "result": cleanData(resultValue),
                "stdout": capturedStdout
            }
            tool.toolLog.append(output)
            return output
        
        except SyntaxError:     # The code contains statements, it is not just a pure expression
            localScope = {}
            exec(strippedCode, safeGlobals, localScope)
            capturedStdout = redirectedOutput.getvalue()

            preloadedModules = {"math", "numpy", "np", "random", "datetime"}
            cleanedLocalScope = {
                k: cleanData(v)
                for k, v in localScope.items()
                if not k.startswith("_") and k not in preloadedModules
            }
            output = {
                "success": True,
                "variables": cleanedLocalScope,
                "stdout": capturedStdout
            }
            tool.toolLog.append(output)
            return output
        
    except Exception as e:
        tb = traceback.format_exc()
        failedLine = None

        if isinstance(e, SyntaxError):
            failedLine = {
                "line": e.lineno,
                "column": e.offset,
                "code": e.text.rstrip() if e.text else None
            }
        else:
            _, _, excTraceback = sys.exc_info()
            if excTraceback is not None:
                try:
                    frames = traceback.extract_tb(excTraceback)
                    if frames:
                        lastFrame = frames[-1]
                        failedLine = {
                            "line": lastFrame.lineno,
                            "code": lastFrame.line
                        }
                except Exception:
                    pass

        lineCount = len(strippedCode.splitlines())
        hintStr = "Keep Python calculations simple (1-5 lines max). Do NOT write functions, loops, or complex scripts."
        if lineCount > 15:
            hintStr = f"Your code was {lineCount} lines long, which caused errors. Simplify your code to a short 1-3 line direct arithmetic calculation."

        errorResult = {
            "success": False,
            "error": f"{e.__class__.__name__}: {str(e)}",
            "hint": hintStr,
            "traceback": tb,
            "failedLine": failedLine
        }

        # If exec() was attempted, include the partially-built variable state
        if localScope is not None:
            preloadedModules = {"math", "numpy", "np", "random", "datetime"}
            try:
                cleanedVars = {
                    k: cleanData(v)
                    for k, v in localScope.items()
                    if not k.startswith("_") and k not in preloadedModules
                }
            except Exception:
                cleanedVars = {"note": "Could not serialize variable data"}
            errorResult["variables"] = cleanedVars

        tool.toolLog.append(errorResult)
        return errorResult
    finally:
        sys.stdout = oldStdout



def confirmBoardroomDecisionBase(
    tool: Tool, 
    ticker: str, 
    rating: str, 
    weighting: str, 
    targetKey1: str, 
    targetVal1: float, 
    targetKey2: str, 
    targetVal2: float
) -> Dict[str, Any]:
    try:
        rating = rating.upper()
        weighting = weighting.upper()

        if rating not in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            return f"Invalid rating value: {rating}. Must be one of STRONG BUY, BUY, HOLD, SELL, STRONG SELL."
        if weighting not in ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"]:
            return f"Invalid weighting value: {weighting}. Must be one of UNDERWEIGHT, EQUAL-WEIGHT, OVERWEIGHT."

        cleanedVal1 = cleanNumber(targetVal1, NumberType.STOCK_PRICE)[1:]
        cleanedVal2 = cleanNumber(targetVal2, NumberType.STOCK_PRICE)[1:]

        summaryLogStr = f"Boardroom decision confirmed for {ticker}: Rating: {rating}, Weighting: {weighting}, {targetKey1}: ${cleanedVal1}, {targetKey2}: ${cleanedVal2}."

        result = {
            "ticker": ticker,
            "rating": rating,
            "weighting": weighting,
            targetKey1: cleanedVal1,
            targetKey2: cleanedVal2
        }
        tool.toolLog.append(result | {"summary": summaryLogStr, "targets": [targetVal1, targetVal2]})
        tool.toolLog.append(summaryLogStr)

        return cleanData(result)
    except Exception as e:
        return f"An error occurred while confirming boardroom decision: {str(e)}"

def confirmBoardroomDecisionImmediateTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp,
                                      ticker: str, rating: str, weighting: str, threeDayTarget: float, twoWeekTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeDayTarget", threeDayTarget, "twoWeekTarget", twoWeekTarget)


def confirmBoardroomDecisionShortTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                      ticker: str, rating: str, weighting: str, oneMonthTarget: float, threeMonthTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "oneMonthTarget", oneMonthTarget, "threeMonthTarget", threeMonthTarget)


def confirmBoardroomDecisionMediumTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                       ticker: str, rating: str, weighting: str, threeMonthTarget: float, twelveMonthTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeMonthTarget", threeMonthTarget, "twelveMonthTarget", twelveMonthTarget)


def confirmBoardroomDecisionLongTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                     ticker: str, rating: str, weighting: str, twelveMonthTarget: float, threeYearTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "twelveMonthTarget", twelveMonthTarget, "threeYearTarget", threeYearTarget)


def confirmBoardroomDecisionDistantTerm(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, 
                                        ticker: str, rating: str, weighting: str, threeYearTarget: float, tenYearTarget: float) -> Dict[str, Any]:
    return confirmBoardroomDecisionBase(tool, ticker, rating, weighting, "threeYearTarget", threeYearTarget, "tenYearTarget", tenYearTarget)
