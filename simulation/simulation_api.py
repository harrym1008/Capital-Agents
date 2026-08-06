import os
import time
import math
import random
import pandas as pd
from datetime import datetime, timedelta
from flask import request, jsonify, session

from collectors.constants import END_DATE_STR, ALL_TICKERS_FILE, NYSE_DIRECTORY, NASDAQ_DIRECTORY, NEW_YORK
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache
from dataquery.price_provider import DailyPriceProvider
from dataquery.ticker_provider import TickerDataProvider
from simulation.market_sim import MarketSimulation
from simulation.orders import MarketOrder, LimitOrder, StopOrder, StopLimitOrder, OrderSide, OrderStatus
from ui.ws_api import generateOhlcvChartData


class SimulationManager:
    def __init__(self, cacheSizeBytes=256 * 1024 ** 2):
        self.simCache = LRUCache(cacheSizeBytes)
        self.tickerProvider = TickerDataProvider()
        self.rateLimiters = GlobalRateLimiters()
        self.priceProvider = DailyPriceProvider(self.tickerProvider, self.simCache, self.rateLimiters)

        self.userSimulations = {}
        self.userHistory = {}

        # Build available tickers index
        self.availableTickers = []
        self.allTickersCount = 0
        try:
            if os.path.exists(ALL_TICKERS_FILE):
                allTickersDf = pd.read_parquet(ALL_TICKERS_FILE)
                self.allTickersCount = len(allTickersDf)
                for row in allTickersDf.itertuples():
                    ticker = row.ticker
                    exchange = row.exchange
                    if exchange == "XNYS":
                        path = os.path.join(NYSE_DIRECTORY, f"{ticker}.parquet")
                    else:
                        path = os.path.join(NASDAQ_DIRECTORY, f"{ticker}.parquet")
                    if os.path.exists(path):
                        self.availableTickers.append(ticker)
        except Exception as e:
            raise RuntimeError(f"Failed to load available tickers: {str(e)}")
            

    def cleanNans(self, obj):
        if isinstance(obj, dict):
            return {k: self.cleanNans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.cleanNans(x) for x in obj]
        elif isinstance(obj, (float, int)) or hasattr(obj, "dtype"):
            try:
                val = float(obj)
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            except (ValueError, TypeError):
                return str(obj)
        return obj

    def getSessionId(self):
        reqSessionId = request.headers.get("X-Session-ID") or request.args.get("sessionId")
        if reqSessionId:
            return reqSessionId
        if "sim_session_id" not in session:
            session["sim_session_id"] = f"session_{time.perf_counter_ns()}"
        return session["sim_session_id"]

    def getSessionSimulation(self, sessionId):
        return self.userSimulations.get(sessionId)

    def createSimulation(self, sessionId, portfolioName, startDate, initialCash):
        activeSim = MarketSimulation(startDate, END_DATE_STR, 
                                     tickerDataProvider=self.tickerProvider, dailyPriceProvider=self.priceProvider)
        activeSim.initialiseUser(portfolioName, initialCash=initialCash)

        self.userSimulations[sessionId] = {
            "sim": activeSim,
            "portfolioName": portfolioName,
            "inspectedTicker": "NVDA",
            "inspectedTimeframe": "3M"
        }
        
        # Pre-populate 2 preceding days at starting balance so Chart.js always draws a line at start
        startTs = pd.Timestamp(startDate)
        dayMinus2 = (startTs - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        dayMinus1 = (startTs - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.userHistory[sessionId] = [
            {
                "date": dayMinus2,
                "totalValue": float(initialCash),
                "cash": float(initialCash),
                "stockValue": 0.0,
                "absReturn": 0.0,
                "pctReturn": 0.0
            },
            {
                "date": dayMinus1,
                "totalValue": float(initialCash),
                "cash": float(initialCash),
                "stockValue": 0.0,
                "absReturn": 0.0,
                "pctReturn": 0.0
            }
        ]
        self.updateSimulationHistory(sessionId)
        return self.getSimulationState(sessionId)

    def updateSimulationHistory(self, sessionId):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            return
        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        if sessionId not in self.userHistory:
            self.userHistory[sessionId] = []

        currentDateStr = activeSim.currentDate.strftime("%Y-%m-%d")
        pfDict = activeSim.getPortfolioValueAtCurrentDate(portfolioName)
        if pfDict:
            totalValue = pfDict["cash"]
            stockValue = 0.0
            for ticker, pos in pfDict["positions"].items():
                curPrice = pos.get("currentPrice")
                if curPrice is None or math.isnan(curPrice) or math.isinf(curPrice) or curPrice <= 0:
                    curPrice = pos.get("averagePrice", 0.0)
                if curPrice is not None and not math.isnan(curPrice) and not math.isinf(curPrice):
                    posVal = pos["quantity"] * curPrice
                    totalValue += posVal
                    stockValue += posVal

            snapshot = {
                "date": currentDateStr,
                "totalValue": totalValue,
                "cash": pfDict["cash"],
                "stockValue": stockValue,
                "absReturn": totalValue - pfDict["startingCash"],
                "pctReturn": ((totalValue / pfDict["startingCash"]) - 1) * 100 if pfDict["startingCash"] != 0 else 0.0
            }

            if self.userHistory[sessionId] and self.userHistory[sessionId][-1]["date"] == currentDateStr:
                self.userHistory[sessionId][-1] = snapshot
            else:
                self.userHistory[sessionId].append(snapshot)

    def getSimulationState(self, sessionId):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            return {"active": False}

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        d = activeSim.currentDate
        currentDateStr = d.strftime("%Y-%m-%d")
        currentDateFriendly = f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}"

        pfDict = activeSim.getPortfolioValueAtCurrentDate(portfolioName)
        if pfDict is None:
            return {"active": False}

        totalValue = pfDict["cash"]
        stockValue = 0.0
        for ticker, pos in pfDict["positions"].items():
            curPrice = pos["currentPrice"]
            if curPrice is not None and not math.isnan(curPrice) and not math.isinf(curPrice):
                posVal = pos["quantity"] * curPrice
                totalValue += posVal
                stockValue += posVal

        pfDict["totalValue"] = totalValue
        pfDict["stockValue"] = stockValue
        pfDict["absReturn"] = totalValue - pfDict["startingCash"]
        if pfDict["startingCash"] != 0:
            pfDict["pctReturn"] = (totalValue / pfDict["startingCash"] - 1) * 100
        else:
            pfDict["pctReturn"] = 0.0

        portfolioObj = activeSim.userPortfolios.get(portfolioName)
        pfDict["log"] = list(portfolioObj.log) if portfolioObj else []

        pendingOrdersList = []
        for idx, userOrder in enumerate(activeSim.pendingOrders):
            o = userOrder.order
            orderTypeStr = "Market"
            if isinstance(o, LimitOrder):
                orderTypeStr = "Limit"
            elif isinstance(o, StopOrder):
                orderTypeStr = "Stop"
            elif isinstance(o, StopLimitOrder):
                orderTypeStr = "Stop Limit"

            pendingOrdersList.append({
                "index": idx,
                "ticker": o.ticker,
                "side": o.side.value,
                "type": orderTypeStr,
                "isQuantityBased": o.isQuantityBased,
                "quantity": o.quantity,
                "cashValue": o.cashValue,
                "limitPrice": getattr(o, "limitPrice", None),
                "stopPrice": getattr(o, "stopPrice", None),
                "status": o.status.value
            })

        isEnded = activeSim.currentDate >= activeSim.endDate
        historyList = self.userHistory.get(sessionId, [])

        inspectedTicker = simData.get("inspectedTicker", "NVDA")
        inspectedTimeframe = simData.get("inspectedTimeframe", "3M")
        inspectedInfo = self.getTickerInfo(sessionId, inspectedTicker, timeframe=inspectedTimeframe)

        rawState = {
            "active": True,
            "portfolioName": portfolioName,
            "currentDate": currentDateStr,
            "currentDateFriendly": currentDateFriendly,
            "isEnded": isEnded,
            "portfolio": pfDict,
            "pendingOrders": pendingOrdersList,
            "history": historyList,
            "inspectedInfo": inspectedInfo
        }
        return self.cleanNans(rawState)

    def submitOrder(self, sessionId, ticker, side, orderType, amountType, amountValue, limitPrice=None, stopPrice=None):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session found. Please start a simulation.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        if ticker not in activeSim.dailyPriceProvider.tickersPaths or activeSim.dailyPriceProvider.tickersPaths[ticker] is None:
            raise ValueError(f"Ticker '{ticker}' is not available in the price database.")

        portfolioObj = activeSim.userPortfolios.get(portfolioName)
        if not portfolioObj:
            raise ValueError("Portfolio not found.")

        normalizedAmountType = str(amountType).lower().strip()
        if normalizedAmountType in ["shares", "quantity", "qty"]:
            amountType = "quantity"
        elif normalizedAmountType in ["usd", "cash", "dollars"]:
            amountType = "cash"

        if side == "SELL":
            if ticker not in portfolioObj.positions:
                raise ValueError(f"Cannot sell '{ticker}' because you do not hold any active positions in it.")
            heldQty = portfolioObj.positions[ticker].quantity
            if amountType == "quantity":
                if amountValue > heldQty:
                    if amountValue - heldQty <= 0.0001:
                        amountValue = heldQty
                    else:
                        raise ValueError(f"Insufficient shares to sell. You hold {heldQty:.4f} shares of '{ticker}'.")
                elif heldQty - amountValue <= 0.0001:
                    amountValue = heldQty
        elif side == "BUY":
            if amountType == "cash":
                if amountValue > portfolioObj.cash:
                    raise ValueError(f"Insufficient cash balance. Available: ${portfolioObj.cash:,.2f}.")

        sideEnum = OrderSide.BUY if side == "BUY" else OrderSide.SELL

        if orderType == "limit":
            limitVal = float(limitPrice) if limitPrice else 0.0
            order = LimitOrder(ticker, sideEnum, limitPrice=limitVal, quantity=amountValue) if amountType == "quantity" else LimitOrder(ticker, sideEnum, limitPrice=limitVal, cashValue=amountValue)
        elif orderType == "stop":
            stopVal = float(stopPrice) if stopPrice else 0.0
            order = StopOrder(ticker, sideEnum, stopPrice=stopVal, quantity=amountValue) if amountType == "quantity" else StopOrder(ticker, sideEnum, stopPrice=stopVal, cashValue=amountValue)
        elif orderType == "stop_limit":
            stopVal = float(stopPrice) if stopPrice else 0.0
            limitVal = float(limitPrice) if limitPrice else 0.0
            order = StopLimitOrder(ticker, sideEnum, stopPrice=stopVal, limitPrice=limitVal, quantity=amountValue) if amountType == "quantity" else StopLimitOrder(ticker, sideEnum, stopPrice=stopVal, limitPrice=limitVal, cashValue=amountValue)
        else:  # market
            order = MarketOrder(ticker, sideEnum, quantity=amountValue) if amountType == "quantity" else MarketOrder(ticker, sideEnum, cashValue=amountValue)

        activeSim.addOrder(order, portfolioName)
        return self.getSimulationState(sessionId)

    def cancelOrder(self, sessionId, orderIndex):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        idx = int(orderIndex)
        if 0 <= idx < len(activeSim.pendingOrders):
            activeSim.pendingOrders.pop(idx)
            return self.getSimulationState(sessionId)
        else:
            raise ValueError("Order index out of range.")

    def getCurrentlyListedTickers(self, sessionId=None):
        simData = self.userSimulations.get(sessionId) if sessionId else None
        if simData and simData["sim"]:
            simDate = simData["sim"].currentDate
        else:
            simDate = pd.Timestamp.now(tz=NEW_YORK).normalize()

        listedTickers = [
            t for t in self.availableTickers
            if self.tickerProvider.isTickerListed(t, simDate)
        ]
        return listedTickers if len(listedTickers) > 0 else self.availableTickers

    def getRandomListedTicker(self, sessionId=None):
        listed = self.getCurrentlyListedTickers(sessionId)
        return random.choice(listed)

    def queueRandomBasket(self, sessionId, count):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        listedTickers = self.getCurrentlyListedTickers(sessionId)
        if len(listedTickers) == 0:
            raise ValueError("No available listed tickers in database.")

        count = min(int(count), len(listedTickers))
        portfolioObj = activeSim.userPortfolios.get(portfolioName)
        if not portfolioObj or portfolioObj.cash <= 0:
            raise ValueError("Insufficient cash balance to queue random tickers.")

        cashPerTicker = portfolioObj.cash / count
        selectedTickers = random.sample(listedTickers, count)

        for ticker in selectedTickers:
            order = MarketOrder(ticker, OrderSide.BUY, cashValue=cashPerTicker)
            activeSim.addOrder(order, portfolioName)

        return self.getSimulationState(sessionId)

    def advanceSimulation(self, sessionId, days=None, targetDate=None, targetDateCutoff=None):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        if targetDateCutoff:
            cutoffTs = pd.Timestamp(targetDateCutoff).tz_localize(NEW_YORK)
            if activeSim.currentDate >= cutoffTs:
                state = self.getSimulationState(sessionId)
                return 0, state

            nextTradingTs = activeSim.currentDate + timedelta(days=1)
            while not activeSim.isTradingDay(nextTradingTs) and nextTradingTs <= activeSim.endDate:
                nextTradingTs += timedelta(days=1)

            if nextTradingTs > cutoffTs:
                state = self.getSimulationState(sessionId)
                return 0, state

        untilEnd = False
        parsedTargetDate = None
        daysToAdvance = 0

        if targetDate:
            parsedTargetDate = pd.Timestamp(targetDate).tz_localize(NEW_YORK)
            if activeSim.currentDate >= parsedTargetDate:
                raise ValueError(f"Target date '{targetDate}' must be after current simulated date '{activeSim.currentDate.strftime('%Y-%m-%d')}'.")
            daysToAdvance = (parsedTargetDate - activeSim.currentDate).days
        elif days == "untilEnd":
            untilEnd = True
            daysToAdvance = 5000
        else:
            daysToAdvance = int(days)
            if daysToAdvance <= 0:
                raise ValueError("Days to advance must be positive.")

        i = 0
        executedDays = 0
        while True:
            if parsedTargetDate is not None:
                if activeSim.currentDate >= parsedTargetDate:
                    break
            elif not untilEnd and i >= daysToAdvance:
                break

            try:
                success = activeSim.runNextDay()
                if not success:
                    break

                i += 1
                executedDays += 1
                self.updateSimulationHistory(sessionId)

            except ValueError as e:
                currentDate = activeSim.currentDate
                portfolioObj = activeSim.userPortfolios.get(portfolioName)
                if portfolioObj:
                    portfolioObj.addToLog(currentDate, f"OrderExecutionFailed: {str(e)}")
                if activeSim.pendingOrders:
                    failedUserOrder = activeSim.pendingOrders.pop(0)
                    failedUserOrder.order.setOrderStatus(OrderStatus.FAILED)
                    activeSim.ordersArchive.append(failedUserOrder)
                break

        state = self.getSimulationState(sessionId)
        return executedDays, state

    def resetSimulation(self, sessionId):
        if sessionId in self.userSimulations:
            del self.userSimulations[sessionId]
        if sessionId in self.userHistory:
            del self.userHistory[sessionId]
        return True

    def getTickerInfo(self, sessionId, ticker, timeframe="3M"):
        ticker = ticker.strip().upper()
        timeframeStr = str(timeframe).strip().lower()
        simData = self.userSimulations.get(sessionId)

        if simData:
            simData["inspectedTicker"] = ticker
            simData["inspectedTimeframe"] = timeframeStr.upper()

        if simData and simData["sim"]:
            simDateTs = simData["sim"].currentDate
            activeSim = simData["sim"]
        else:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
            activeSim = None

        tickerExists = False
        lastPrice = None
        if activeSim and hasattr(activeSim, "dailyPriceProvider"):
            tickerExists = (ticker in activeSim.dailyPriceProvider.tickersPaths and activeSim.dailyPriceProvider.tickersPaths[ticker] is not None)
            if tickerExists:
                row = activeSim.dailyPriceProvider.getSingleDayTickerData(ticker, simDateTs)
                if row is not None and "close" in row and pd.notna(row["close"]):
                    lastPrice = float(row["close"])

        chartData = generateOhlcvChartData(ticker, simDateTs, targets=[], horizon=timeframeStr)
        if lastPrice is None and chartData and chartData.get("simAnchor"):
            lastPrice = chartData["simAnchor"].get("y")

        profile = self.tickerProvider.getTickerProfile(ticker)
        profileData = {}
        if profile:
            profileData = {
                "ticker": profile.ticker,
                "name": profile.name,
                "exchange": profile.exchange,
                "sector": profile.sector,
                "industry": profile.industry,
                "website": profile.website,
                "summary": profile.summary
            }
            tickerExists = True

        return {
            "ok": True,
            "exists": tickerExists,
            "ticker": ticker,
            "lastPrice": lastPrice,
            "chart": chartData,
            "profile": profileData
        }

    def getAvailableTickers(self):
        return {
            "tickers": self.availableTickers,
            "totalCount": self.allTickersCount,
            "availableCount": len(self.availableTickers)
        }


# Global SimulationManager instance
simulationManager = SimulationManager()


def registerSimulationApiRoutes(app):

    @app.route("/api/sim/available-tickers", methods=["GET"])
    def getAvailableTickersRoute():
        return jsonify(simulationManager.getAvailableTickers())

    @app.route("/api/sim/random-ticker", methods=["GET"])
    def getRandomTickerRoute():
        sessionId = simulationManager.getSessionId()
        ticker = simulationManager.getRandomListedTicker(sessionId)
        return jsonify({"ok": True, "ticker": ticker})

    @app.route("/api/sim/start", methods=["POST"])
    def startSimulationRoute():
        data = request.get_json(silent=True) or {}
        portfolioName = data.get("portfolioName", "My Growth Portfolio").strip() or "My Growth Portfolio"
        startDate = data.get("startDate", "2016-06-01").strip()
        dollarAmountStr = str(data.get("dollarAmount", "1000000"))

        try:
            dollarAmount = float(dollarAmountStr)
            if dollarAmount <= 0:
                return jsonify({"ok": False, "error": "Starting dollar amount must be positive."}), 400
        except ValueError:
            return jsonify({"ok": False, "error": "Invalid starting dollar balance."}), 400

        try:
            sessionId = simulationManager.getSessionId()
            state = simulationManager.createSimulation(sessionId, portfolioName, startDate, dollarAmount)
            return jsonify({"ok": True, "state": state, "sessionId": sessionId})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/sim/state", methods=["GET"])
    def getSimulationStateRoute():
        sessionId = simulationManager.getSessionId()
        state = simulationManager.getSimulationState(sessionId)
        return jsonify({"ok": True, "state": state})

    @app.route("/api/sim/order", methods=["POST"])
    def submitOrderRoute():
        sessionId = simulationManager.getSessionId()
        data = request.get_json(silent=True) or {}
        ticker = data.get("ticker", "").upper().strip()
        side = data.get("side", "BUY").upper().strip()
        orderType = data.get("type", "market").lower().strip()
        amountType = data.get("amountType", "cash").lower().strip()
        amountValueStr = str(data.get("amountValue", "0"))
        limitPriceStr = data.get("limitPrice")
        stopPriceStr = data.get("stopPrice")

        if not ticker:
            return jsonify({"ok": False, "error": "Ticker symbol cannot be empty."}), 400

        try:
            amountValue = float(amountValueStr)
            if amountValue <= 0:
                return jsonify({"ok": False, "error": "Amount must be greater than zero."}), 400
        except ValueError:
            return jsonify({"ok": False, "error": "Amount must be a valid number."}), 400

        try:
            state = simulationManager.submitOrder(
                sessionId=sessionId,
                ticker=ticker,
                side=side,
                orderType=orderType,
                amountType=amountType,
                amountValue=amountValue,
                limitPrice=limitPriceStr,
                stopPrice=stopPriceStr
            )
            return jsonify({"ok": True, "state": state})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sim/cancel-order", methods=["POST"])
    def cancelOrderRoute():
        sessionId = simulationManager.getSessionId()
        data = request.get_json(silent=True) or {}
        orderIndex = data.get("index")
        if orderIndex is None:
            return jsonify({"ok": False, "error": "Missing order index."}), 400

        try:
            state = simulationManager.cancelOrder(sessionId, orderIndex)
            return jsonify({"ok": True, "state": state})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sim/random-basket", methods=["POST"])
    def queueRandomBasketRoute():
        sessionId = simulationManager.getSessionId()
        data = request.get_json(silent=True) or {}
        countStr = str(data.get("count", "5"))

        try:
            state = simulationManager.queueRandomBasket(sessionId, countStr)
            return jsonify({"ok": True, "state": state})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sim/advance", methods=["POST"])
    def advanceSimulationRoute():
        sessionId = simulationManager.getSessionId()
        data = request.get_json(silent=True) or {}
        daysToAdvanceStr = data.get("days")
        targetDateStr = data.get("targetDate")
        targetDateCutoffStr = data.get("targetDateCutoff")

        try:
            executedDays, state = simulationManager.advanceSimulation(
                sessionId=sessionId,
                days=daysToAdvanceStr,
                targetDate=targetDateStr,
                targetDateCutoff=targetDateCutoffStr
            )
            return jsonify({"ok": True, "executedDays": executedDays, "state": state})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sim/reset", methods=["POST"])
    def resetSimulationRoute():
        sessionId = simulationManager.getSessionId()
        simulationManager.resetSimulation(sessionId)
        return jsonify({"ok": True})

    @app.route("/api/sim/ticker-info", methods=["GET"])
    def getTickerInfoRoute():
        ticker = request.args.get("ticker", "NVDA").strip().upper()
        timeframeStr = request.args.get("timeframe", "3M").strip()
        sessionId = simulationManager.getSessionId()
        try:
            info = simulationManager.getTickerInfo(sessionId, ticker, timeframe=timeframeStr)
            return jsonify({
                "ok": True,
                "exists": info["exists"],
                "lastPrice": info["lastPrice"],
                "chart": info["chart"],
                "profile": info["profile"]
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
