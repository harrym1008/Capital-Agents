from enum import Enum
import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from collectors.constants import END_DATE

FINANCIAL_SERVICES_WARNING = "`financial_services` is the incorrect key for that sector, you must, from now onwards, use `financials`."


# Recursively scans and normalises sector identifiers from 'financial_services' to 'financials'
def normaliseSectorInput(val: Any) -> Tuple[Any, bool]:
    wasUpdated = False

    if isinstance(val, str):
        cleaned = val.strip()
        lower = cleaned.lower()
        if lower in ["financial_services", "financial services"]:
            return "financials", True
        return val, False

    elif isinstance(val, dict):
        newDict = {}
        for k, v in val.items():
            newKey = k
            if isinstance(k, str):
                cleanedK = k.strip().lower()
                if cleanedK in ["financial_services", "financial services"]:
                    newKey = "financials" if k == k.lower() else "Financials"
                    wasUpdated = True

            newV, valUpdated = normaliseSectorInput(v)
            if valUpdated:
                wasUpdated = True
            newDict[newKey] = newV
        return newDict, wasUpdated

    elif isinstance(val, list):
        newList = []
        for item in val:
            newItem, itemUpdated = normaliseSectorInput(item)
            if itemUpdated:
                wasUpdated = True
            newList.append(newItem)
        return newList, wasUpdated

    elif isinstance(val, tuple):
        items = []
        for item in val:
            newItem, itemUpdated = normaliseSectorInput(item)
            if itemUpdated:
                wasUpdated = True
            items.append(newItem)
        return tuple(items), wasUpdated

    return val, False


# Attaches top-level 'important' warning if 'financial_services' was used as input
def attachSectorWarning(output: Dict[str, Any], wasUpdated: bool) -> Dict[str, Any]:
    if wasUpdated and isinstance(output, dict):
        reordered = {"important": FINANCIAL_SERVICES_WARNING}
        reordered.update(output)
        return reordered
    return output


# Format dictionary key as date string or stripped text
def cleanKey(key):
    if hasattr(key, "strftime"):
        return key.strftime("%Y-%m-%d")
    return str(key).strip()


# Recursively sanitise nested structures converting NaN to None
def cleanData(value):
    if isinstance(value, dict):
        return {cleanKey(k): cleanData(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [cleanData(item) for item in value]
    elif hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    elif isinstance(value, (int, np.integer, float, np.floating)):
        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value 
    elif isinstance(value, pd.Series):
        return {cleanKey(k): cleanData(v) for k, v in value.items()}
    elif isinstance(value, pd.DataFrame):
        outputDict = {}
        for colName in value.columns:
            cleanedColName = str(colName)
            outputDict[cleanedColName] = {cleanKey(idx): cleanData(val) for idx, val in value[colName].items()}
        return outputDict
    return value


# Enumeration of numerical formatting profiles
class NumberType(Enum):
    DOLLARS = 1
    DOLLARS_CHANGE = 2
    DECIMAL = 3
    DECIMAL_CHANGE = 4
    LARGE_DOLLARS = 5
    LARGE_DOLLARS_CHANGE = 6
    SMALL_DOLLARS = 7
    SMALL_DOLLARS_CHANGE = 8
    PERCENTAGE = 9
    PERCENTAGE_CHANGE = 10
    STOCK_PRICE = 11
    STOCK_PRICE_CHANGE = 12
    LARGE_NUMBER = 13
    UNSCALED_PERCENTAGE = 14
    EXCHANGE_RATE = 15
    CHANGE_BP = 16 


# Format numeric value according to specified financial display profile
def cleanNumber(value, numType: NumberType):
    if value is None:
        return "N/A"
    if pd.isna(value):
        return "null"
    
    def stringifyNumber(value, sf=4, minDp=1, plusSign=False, leading="", trailing=""):
        if value == 0:
            return ("0" + "0" * (minDp - 1)) if minDp > 1 else "0"
        
        sign = "-" if value < 0 else ("+" if plusSign and value > 0 else "")
        value = abs(value)
        
        magnitude = math.floor(math.log10(value))
        decimals = max(sf - magnitude - 1, minDp)
        formatted = f"{value:.{decimals}f}"

        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return sign + leading + formatted + trailing
        
    def formatLargeDollars(value):
        absValue = abs(value)
        if absValue >= 1_000_000_000_000:
            return stringifyNumber(value / 1_000_000_000_000, sf=4, minDp=1, leading="$") + "tn"
        elif absValue >= 1_000_000_000:
            return stringifyNumber(value / 1_000_000_000, sf=4, minDp=1, leading="$") + "bn"
        elif absValue >= 1_000_000:
            return stringifyNumber(value / 1_000_000, sf=4, minDp=1, leading="$") + "mn"
        elif absValue >= 1_000:
            return stringifyNumber(value / 1_000, sf=4, minDp=1, leading="$") + "k"
        else:
            return stringifyNumber(value, sf=4, minDp=1, leading="$")

    if isinstance(value, (int, float, np.integer, np.floating)):
        match numType:
            case NumberType.DOLLARS:
                return stringifyNumber(value, sf=4, minDp=1, leading="$")
            case NumberType.DOLLARS_CHANGE:
                return stringifyNumber(value, sf=4, minDp=1, plusSign=True, leading="$")
            case NumberType.DECIMAL:
                return stringifyNumber(value, sf=3, minDp=1)
            case NumberType.DECIMAL_CHANGE:
                return stringifyNumber(value, sf=3, minDp=1, plusSign=True)
            case NumberType.SMALL_DOLLARS:
                return stringifyNumber(value, sf=4, minDp=2, leading="$")
            case NumberType.SMALL_DOLLARS_CHANGE:
                return stringifyNumber(value, sf=4, minDp=2, plusSign=True, leading="$")
            case NumberType.PERCENTAGE: 
                return stringifyNumber(value, sf=3, minDp=1, trailing="%")
            case NumberType.PERCENTAGE_CHANGE:
                if value == 0:
                    return "0% (no change)"
                return stringifyNumber(value, sf=3, minDp=1, plusSign=True, trailing="%")
            
            case NumberType.LARGE_DOLLARS:
                return formatLargeDollars(value)
            case NumberType.LARGE_DOLLARS_CHANGE:
                return formatLargeDollars(value) if value < 0 else "+" + formatLargeDollars(value)
            case NumberType.STOCK_PRICE:
                numStr = stringifyNumber(value, sf=4, minDp=2, leading="$")
                if "." not in numStr:
                    numStr += ".00"
                elif len(numStr.split(".")[1]) == 1:
                    numStr += "0"
                return numStr
                
            case NumberType.STOCK_PRICE_CHANGE:
                numStr = stringifyNumber(value, sf=4, minDp=2, plusSign=True, leading="$")
                if "." not in numStr:
                    numStr += ".00"
                elif len(numStr.split(".")[1]) == 1:
                    numStr += "0"
                return numStr

            case NumberType.LARGE_NUMBER:
                # Strip dollar sign for large unit values
                return formatLargeDollars(value).replace("$", "")

            case NumberType.UNSCALED_PERCENTAGE:
                return stringifyNumber(value*100, sf=3, minDp=1, trailing="%")

            case NumberType.EXCHANGE_RATE:
                return stringifyNumber(value, sf=5, minDp=2)

            case NumberType.CHANGE_BP:
                return stringifyNumber(value, sf=3, minDp=1, plusSign=True) + " bps"

            case _:
                return str(value)
    else:
        return str(value)


# Strip HTML tags and standardise unicode quotation and punctuation characters
def cleanHtmlContent(htmlContent):
    if not htmlContent:
        return ""
    soup = BeautifulSoup(htmlContent, "html.parser")
    rawText = soup.get_text(separator=" ")
    replacements = [
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", "'"),
        ("\u201d", "'"),
        ("\u2013", "-"),
        ("\u2014", " - "),
        ("\u2022", "*"),
        ("\u2026", "..."),
        ("\xa0", " "),
        ("\u00a0", " "),
        ('"', "'")
    ]
    cleanText = " ".join(rawText.split())
    for old, new in replacements:
        cleanText = cleanText.replace(old, new)

    removeRightwards = [
        "Read More:",
        "Read Next:",
        "Photo Courtesy:",
        "Photo courtesy:",
        "Image: "
    ]
    for phrase in removeRightwards:
        if phrase in cleanText:
            cleanText = cleanText.split(phrase)[0].strip()
    return cleanText


# Convert timestamp to timezone-naive UTC timestamp
def tsToUtcNaive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_localize(None)
    return ts.tz_convert("UTC").tz_localize(None)


# Verify whether timestamp falls within local historical data boundary
def isLocalDataAvailable(timestamp: pd.Timestamp) -> bool:  
    ts = tsToUtcNaive(timestamp)
    endDate = tsToUtcNaive(END_DATE)
    return ts <= endDate


# Compute human-readable relative age string for article timestamp
def formatArticleAge(dateVal, timestamp: pd.Timestamp) -> str:
    if dateVal is None or pd.isna(dateVal) or dateVal == "":
        return "unknown"
    try:
        if isinstance(dateVal, pd.Timestamp):
            articleTimestamp = dateVal
        else:
            articleTimestamp = pd.Timestamp(dateVal)

        if pd.isna(articleTimestamp):
            return "unknown"

        if articleTimestamp.tzinfo is not None:
            if timestamp.tzinfo is not None:
                articleTimestamp = articleTimestamp.tz_convert(timestamp.tzinfo)
            else:
                articleTimestamp = articleTimestamp.tz_localize(None)
        else:
            if timestamp.tzinfo is not None:
                articleTimestamp = articleTimestamp.tz_localize(timestamp.tzinfo)

        age = timestamp - articleTimestamp
        if pd.isna(age) or not isinstance(age, pd.Timedelta):
            return "unknown"

        if age < pd.Timedelta(0):
            return "0m old"
        elif age < pd.Timedelta(hours=1):
            return f"{age.components.minutes}m old"
        elif age < pd.Timedelta(days=1):
            return f"{age.components.hours}h {age.components.minutes}m old"
        elif age < pd.Timedelta(days=7):
            return f"{age.components.days}d {age.components.hours}h old"
        else:
            return f"{age.components.days}d old"
    except Exception:
        return "unknown"
