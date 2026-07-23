from enum import Enum
import math

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from collectors.constants import END_DATE


def cleanKey(key):
    if hasattr(key, "strftime"):
        return key.strftime("%Y-%m-%d")
    return str(key).strip()


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


def cleanNumber(value, numType: NumberType):
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
                return formatLargeDollars(value).replace("$", "")   # Remove the dollar sign for large numbers

            case NumberType.UNSCALED_PERCENTAGE:
                return stringifyNumber(value*100, sf=3, minDp=1, trailing="%")

            case NumberType.EXCHANGE_RATE:
                return stringifyNumber(value, sf=5, minDp=2)

            case _:
                return str(value)
    else:
        return str(value)

    
def cleanHtmlContent(htmlContent):  # Convert HTML and convert unicode chars
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



def tsToUtcNaive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").tz_localize(None)
    return ts.tz_convert("UTC").tz_localize(None)


def isLocalDataAvailable(timestamp: pd.Timestamp) -> bool:  
    ts = tsToUtcNaive(timestamp)
    endDate = tsToUtcNaive(END_DATE)

    return ts <= endDate

