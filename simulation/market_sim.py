from enum import Enum
from datetime import timedelta

import pandas as pd
import math

from dataquery.price_provider import DailyPriceProvider
from collectors.market_calendar import MarketCalendar 
from collectors.constants import NEW_YORK
from dataquery.ticker_provider import TickerDataProvider
from simulation.orders import Order, MarketOrder, LimitOrder, StopOrder, StopLimitOrder, OrderSide, OrderStatus
from simulation.portfolio import Position, Dividend, Portfolio


class ExecutionTime(Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"


class UserOrder:
    def __init__(self, order: Order, user, executionTime=ExecutionTime.OPEN):
        self.order = order
        self.user = user
        self.executionTime = executionTime


class SegmentExecutionResult:
    def __init__(self, filled, executionPrice=None, stopTriggered=False):
        self.filled = filled
        self.executionPrice = executionPrice
        self.stopTriggered = stopTriggered      # Only relevant for StopLimitOrder



class MarketSimulation:
    def __init__(self, startDate, endDate, tickerDataProvider=None, dailyPriceProvider=None):
        self.startDate = pd.Timestamp(startDate, tz=NEW_YORK).normalize()
        self.endDate = pd.Timestamp(endDate, tz=NEW_YORK).normalize()    
        self.currentDate = self.startDate
        self.started = False

        self.marketCalendar = MarketCalendar(startDate, endDate)
        self.tickerDataProvider = TickerDataProvider() if tickerDataProvider is None else tickerDataProvider
        self.dailyPriceProvider = DailyPriceProvider(self.tickerDataProvider) if dailyPriceProvider is None else dailyPriceProvider

        self.users: list[str] = []                          # Set of all users in the simulation
        self.userPortfolios: dict[str, Portfolio] = {}      # Maps user to their portfolios. The portfolio itself is a dict mapping ticker to Position
        
        self.pendingOrders: list[UserOrder] = []     # Pending UserOrders that have not been filled yet
        self.ordersArchive: list[UserOrder] = []     # All UserOrders that have been processed (filled or failed)


    def initialiseUser(self, username, initialCash=1_000_000.0):
        if username in self.users:
            return

        self.users.append(username)
        self.userPortfolios[username] = Portfolio(initialCash)



    def addOrder(self, order, username):
        userOrder = UserOrder(order, username)
        self.pendingOrders.append(userOrder)
        order.setSubmittedTimestamp(self.currentDate)


    def _buildIntradayPath(self, ohlcData):        
        openPrice = ohlcData["open"]
        highPrice = ohlcData["high"]
        lowPrice = ohlcData["low"]
        closePrice = ohlcData["close"]

        if abs(highPrice - openPrice) <= abs(lowPrice - openPrice):
            closer = highPrice
            farther = lowPrice
        else:
            closer = lowPrice
            farther = highPrice

        segments = [
            (openPrice, closer),
            (closer, farther),
            (farther, closePrice)
        ]

        return segments
    

    def checkOrderInSegment(self, segmentStart, segmentEnd, userOrder: UserOrder):
        order = userOrder.order
        isIncreasing = segmentEnd > segmentStart

        # Process MarketOrder
        if isinstance(order, MarketOrder):
            execTime = userOrder.executionTime
            if execTime == ExecutionTime.OPEN:
                return SegmentExecutionResult(True, segmentStart)
            elif execTime == ExecutionTime.CLOSE:
                return SegmentExecutionResult(True, segmentEnd)
            
        # Process LimitOrder
        elif isinstance(order, LimitOrder):
            limit = order.limitPrice

            if order.side == OrderSide.BUY:
                # Buy limit fills when price drops to/below the limit price
                if isIncreasing:
                    # Price is increasing, cannot fill a buy limit order
                    return SegmentExecutionResult(False)
                else:
                    # Check if the price crosses the limit during this segment
                    if segmentEnd <= limit <= segmentStart:
                        return SegmentExecutionResult(True, limit)      # Fill at the limit price
                    return SegmentExecutionResult(False)
                
            else:
                # Sell limit fills when price rises to/above the limit price
                if isIncreasing:
                    # Check if the price crosses the limit during this segment
                    if segmentStart <= limit <= segmentEnd:
                        return SegmentExecutionResult(True, limit)      # Fill at the limit price
                    return SegmentExecutionResult(False)
                else:
                    # Price is decreasing, cannot fill a sell limit order
                    return SegmentExecutionResult(False)
                
        # Process StopOrder
        elif isinstance(order, StopOrder):
            stop = order.stopPrice

            if order.side == OrderSide.BUY:
                # Buy stop triggers when price rises to/above the stop price
                if isIncreasing:
                    # Check if the price crosses the stop during this segment
                    if segmentStart <= stop <= segmentEnd:
                        return SegmentExecutionResult(True, stop)      # Fill at the stop price
                    return SegmentExecutionResult(False)
                else:
                    # Price is decreasing, cannot trigger a buy stop order
                    return SegmentExecutionResult(False)
                
            else:
                # Sell stop triggers when price drops to/below the stop price
                if isIncreasing:
                    # Price is increasing, cannot trigger a sell stop order
                    return SegmentExecutionResult(False)
                else:
                    # Check if the price crosses the stop during this segment
                    if segmentEnd <= stop <= segmentStart:
                        return SegmentExecutionResult(True, stop)      # Fill at the stop price
                    return SegmentExecutionResult(False)
                

        # Process StopLimitOrder
        elif isinstance(order, StopLimitOrder):
            stop = order.stopPrice
            limit = order.limitPrice

            if order.side == OrderSide.BUY:
                # For a buy stop limit order, stop triggers when price rises >= stop price
                # Limit fills when price (after stop triggers) is <= limit price
                if isIncreasing:
                    # Check if the price crosses the stop during this segment
                    if segmentStart <= stop <= segmentEnd:
                        # Stop triggers here
                        if limit >= stop:
                            # Fill immediately at stop price if limit is above stop
                            return SegmentExecutionResult(True, stop)
                        else:
                            # Stop was triggered but price is already above limit
                            return SegmentExecutionResult(False, stopTriggered=True)
                        
                    # Stop price was not crossed
                    return SegmentExecutionResult(False)
                else:
                    # Price is decreasing, cannot trigger a buy stop order
                    return SegmentExecutionResult(False)
                
            else:
                # For a sell stop limit order, stop triggers when price drops <= stop price
                # Limit fills when price (after stop triggers) is >= limit price
                if isIncreasing:
                    # Price is increasing, cannot trigger a sell stop order
                    return SegmentExecutionResult(False)
                else:
                    # Check if the price crosses the stop during this segment
                    if segmentEnd <= stop <= segmentStart:
                        # Stop triggers here
                        if limit <= stop:
                            # Fill immediately at stop price if limit is below stop
                            return SegmentExecutionResult(True, stop)
                        else:
                            # Stop was triggered but price is already below limit
                            return SegmentExecutionResult(False, stopTriggered=True)
                        
                    # Stop price was not crossed
                    return SegmentExecutionResult(False)


    def processSegment(self, segmentStart, segmentEnd, pendingOrders, ticker):
        executedOrders = []
        remainingOrders = []

        for userOrder in pendingOrders:
            order = userOrder.order

            if order.ticker != ticker:
                remainingOrders.append(userOrder)
                continue

            result = self.checkOrderInSegment(segmentStart, segmentEnd, userOrder)

            if result.filled:
                executedOrders.append((userOrder, result.executionPrice))
            else:
                # Check StopLimitOrder stop trigger without fill
                if isinstance(order, StopLimitOrder):
                    # Re-check the stop trigger condition
                    isIncreasing = segmentEnd > segmentStart
                    stopTriggered = False
                    stop = order.stopPrice
                    limit = order.limitPrice

                    if order.side == OrderSide.BUY:
                        if isIncreasing and segmentStart <= stop <= segmentEnd:
                            stopTriggered = True
                    else:
                        if not isIncreasing and segmentEnd <= stop <= segmentStart:
                            stopTriggered = True

                    if stopTriggered:
                        # Stop triggered - check if the limit is reachable in a future segment
                        if order.side == OrderSide.BUY:
                            if not isIncreasing and segmentEnd <= limit <= segmentStart:
                                # Limit is reachable in the current segment, so we can fill immediately
                                executedOrders.append((userOrder, stop))
                                continue
                        else:
                            if isIncreasing and segmentStart <= limit <= segmentEnd:
                                # Limit is reachable in the current segment, so we can fill immediately
                                executedOrders.append((userOrder, stop))
                                continue

                remainingOrders.append(userOrder)

        return executedOrders, remainingOrders
    

    def processDaysTrades(self, date):
        requiredTickers: set[str] = set()
        tickersToOrders: dict[str, list[UserOrder]] = {}
        for userOrder in self.pendingOrders:
            requiredTickers.add(userOrder.order.ticker)
            if userOrder.order.ticker not in tickersToOrders:
                tickersToOrders[userOrder.order.ticker] = []
            tickersToOrders[userOrder.order.ticker].append(userOrder)

        # Process each pending order
        executedThisDay = []
        failedThisDay = []
        remainingOrders = list(self.pendingOrders)

        # Get the OHLC price data for this date
        for ticker in requiredTickers:
            ohlc = self.dailyPriceProvider.getSingleDayTickerData(ticker, date)

            if ohlc is None:
                # No price data, fail the order
                for userOrder in tickersToOrders[ticker]:
                    self.ordersArchive.append(userOrder)
                    remainingOrders.remove(userOrder)
                    failedThisDay.append(userOrder) 
                continue

            segments = self._buildIntradayPath(ohlc)

            for segmentStart, segmentEnd in segments:
                executed, remaining = self.processSegment(segmentStart, segmentEnd, remainingOrders, ticker)
                executedThisDay.extend(executed)
                remainingOrders = remaining

        
        for userOrder in failedThisDay:
            userOrder.order.setFillParams(None, date)
            userOrder.order.setOrderStatus(OrderStatus.FAILED)
            self.executeTrade(userOrder)
            self.ordersArchive.append(userOrder)
            
        # Execute and archive processed orders (SELL orders first to realise cash before BUY orders)
        executedThisDay.sort(key=lambda item: 0 if item[0].order.side == OrderSide.SELL else 1)
        for userOrder, fillPrice in executedThisDay:
            userOrder.order.setFillParams(fillPrice, date)
            userOrder.order.setOrderStatus(OrderStatus.FILLED)
            self.executeTrade(userOrder)
            self.ordersArchive.append(userOrder)


        self.pendingOrders = remainingOrders


    def executeTrade(self, userOrder):
        username = userOrder.user
        portfolio = self.userPortfolios.get(username, None)
        if portfolio is None:
            return
        
        order = userOrder.order
        portfolio.executeTrade(order)


    def isTradingDay(self, date):
        dateString = date.strftime("%Y-%m-%d")
        return self.marketCalendar.isOpenDay(dateString)
    


    def processDaysCorporateActions(self, date):
        todaysActions = self.dailyPriceProvider.getSingleDayCorpActions(date)
        dateNyTs = self.dailyPriceProvider.timestampToNyDay(date)
        
        def formatFloat(value):
            if pd.isna(value):
                return "N/A"
            absValue = abs(value)

            # Check if e notation
            if "e" in str(absValue).lower():
                return f"{value:.3e}"
            
            if absValue < 1000:
                dp = 3
            else:
                dp = max(0, 3 - int(math.log10(absValue // 1000 + 1)))                
            return f"{value:.{dp}f}".rstrip("0").rstrip(".")

        for portfolio in self.userPortfolios.values():
            for _, action in todaysActions.iterrows():
                ticker = action["ticker"]
                if ticker not in portfolio.positions:
                    # Check "oldTicker" for name changes
                    if action["actionType"] == "name_change" and action["oldTicker"] in portfolio.positions:
                        ticker = action["oldTicker"]
                    else:
                        continue

                position = portfolio.positions[ticker]
                actionType = action["actionType"]

                if actionType in ["forward_split", "reverse_split"]:
                    sharesMult = action["newRate"] / action["oldRate"]
                    priceMult = action["oldRate"] / action["newRate"]
                    position.quantity *= sharesMult
                    position.averagePrice *= priceMult

                    portfolio.addToLog(date, ticker, "Stock Split", f"Conducted a {formatFloat(action['newRate'])}:{formatFloat(action['oldRate'])} stock split. Holding now: {formatFloat(position.quantity)} shares @ ${formatFloat(position.averagePrice)} average.")

                elif actionType in ["cash_dividend", "stock_dividend"]:
                    amount = action["rate"]
                    payDate = action["payDate"]
                    nicePayDate = pd.Timestamp(payDate).strftime("%d %b %Y")

                    dividendType = Dividend.Type.CASH if actionType == "cash_dividend" else Dividend.Type.STOCK

                    if amount < 0.005:
                        portfolio.addToLog(date, ticker, "Dividend", f"Ex-dividend date reached - ${formatFloat(amount)} per share. Too few shares are owned (payout would be less than half a penny).")
                        continue

                    dividend = Dividend(ticker, amount, position.quantity, payDate, dividendType)
                    portfolio.potentialDividends.append(dividend)

                    if dividend.dividendType == Dividend.Type.CASH:
                        portfolio.addToLog(date, ticker, "Dividend", f"Ex-dividend date reached - ${formatFloat(amount)} per share. Will be paid on {nicePayDate}.")
                    else:
                        portfolio.addToLog(date, ticker, "Dividend", f"Ex-dividend date reached - {formatFloat(amount)} per share. Will be paid on {nicePayDate}.")

                elif actionType == "spin_off":
                    sourceTicker = action["sourceTicker"]
                    newTicker = action["newTicker"]
                    if ticker == newTicker or ticker != sourceTicker:
                        continue    # There are two copies of every spin-off action, only use the one where the new ticker is different from the old one

                    spinoffRatio = action["newRate"] / action["sourceRate"]
                    newShares = position.quantity * spinoffRatio

                    if newShares > 0:
                        newTickerPrice = self.getCurrentPrice(newTicker)
                        if pd.isna(newTickerPrice):
                            newTickerPrice = 0.0

                        if newTicker not in portfolio.positions:
                            portfolio.positions[newTicker] = Position(newTicker, newShares, newTickerPrice)
                        else:
                            portfolio.positions[newTicker].increasePosition(newShares, newTickerPrice)

                    portfolio.addToLog(date, ticker, "Spin-off", f"Spin-off of {newTicker} executed (ratio {formatFloat(action['newRate'])}:{formatFloat(action['sourceRate'])}). Received {formatFloat(newShares)} shares of {newTicker}.")

                elif actionType in ["cash_merger", "stock_merger", "stock_and_cash_merger"]:
                    acquireeTicker = action["acquireeTicker"]
                    if ticker != acquireeTicker:
                        continue    # Only process the acquiree side of the merger action

                    acquirerTicker = action["acquirerTicker"]
                    if acquirerTicker is None:
                        continue

                    mergeQty = position.quantity
                    if mergeQty <= 0:
                        continue

                    acquireeRate = action["acquireeRate"]

                    cashPayout = 0.0
                    stockPayout = 0.0

                    if actionType == "cash_merger":
                        cashPayout = mergeQty * action["rate"] / acquireeRate
                    elif actionType == "stock_merger":
                        stockPayout = mergeQty * action["acquirerRate"] / acquireeRate
                    else:
                        cashPayout = mergeQty * action["cashRate"] / acquireeRate
                        stockPayout = mergeQty * action["acquirerRate"] / acquireeRate

                    if cashPayout > 0:
                        portfolio.cash += cashPayout

                    if stockPayout > 0:
                        acquirerPrice = self.getCurrentPrice(acquirerTicker)
                        if pd.isna(acquirerPrice):
                            acquirerPrice = 0.0

                        if acquirerTicker not in portfolio.positions:
                            portfolio.positions[acquirerTicker] = Position(acquirerTicker, stockPayout, acquirerPrice)
                        else:
                            portfolio.positions[acquirerTicker].increasePosition(stockPayout, acquirerPrice)

                    del portfolio.positions[ticker]

                    logMessage = f"{actionType.replace('_', ' ')} executed into {acquirerTicker}. "
                    if cashPayout > 0 and stockPayout > 0:
                        logMessage += f"Received ${formatFloat(cashPayout)} and {formatFloat(stockPayout)} shares."
                    elif cashPayout > 0:
                        logMessage += f"Received ${formatFloat(cashPayout)}."
                    elif stockPayout > 0:
                        logMessage += f"Received {formatFloat(stockPayout)} shares."
                    portfolio.addToLog(date, ticker, actionType.replace("_", " ").title(), logMessage)

                elif actionType == "name_change":
                    oldTicker = action["oldTicker"]
                    newTicker = action["newTicker"]

                    if ticker != oldTicker or oldTicker == newTicker:
                        continue

                    if newTicker in portfolio.positions:
                        portfolio.positions[newTicker].increasePosition(position.quantity, position.averagePrice)
                    else:
                        portfolio.positions[newTicker] = Position(newTicker, position.quantity, position.averagePrice)

                    del portfolio.positions[oldTicker]

                    portfolio.addToLog(date, ticker, "Name Change", f"Changed stock ticker into {newTicker}. Position moved to {newTicker}.")

                elif actionType == "worthless_removal":
                    portfolio.addToLog(date, ticker, "Worthless Removal", f"Delisted from exchange due to bankruptcy. Position liquidated in following order.")

                    order = MarketOrder(ticker, OrderSide.SELL, quantity=-1)
                    order.setFillParams(0.0, date)
                    portfolio.executeTrade(order)


        # Finally process dividends that are payable today
        for portfolio in self.userPortfolios.values():
            for dividend in portfolio.potentialDividends:

                payDateNyTs = self.dailyPriceProvider.timestampToNyDay(dividend.payDate)

                if payDateNyTs == dateNyTs:
                    paymentAmount, dividendType = dividend.calculateDividend()

                    if paymentAmount == 0:
                        continue        # All payable shares were sold

                    if dividendType == Dividend.Type.CASH:
                        portfolio.cash += paymentAmount
                        portfolio.addToLog(date, dividend.ticker, "Payout", f"Received cash dividend for {dividend.ticker} - ${paymentAmount:.2f}.")

                    else:   # Stock dividend
                        position = portfolio.positions.get(dividend.ticker)
                        if position is None:
                            portfolio.positions[dividend.ticker] = Position(dividend.ticker, paymentAmount, 0)
                        else:
                            position.increasePosition(paymentAmount, 0)
                        
                   
                elif payDateNyTs < dateNyTs:
                    # Expired dividend
                    portfolio.potentialDividends.remove(dividend)


    def processTomorrowsDelistings(self, today):
        tomorrow = (today + pd.DateOffset(days=1)).normalize()
        if tomorrow > self.endDate:
            return
        
        while not self.isTradingDay(tomorrow):
            tomorrow = (tomorrow + pd.DateOffset(days=1)).normalize()
            if tomorrow > self.endDate:
                return

        todayNyTs = self.dailyPriceProvider.timestampToNyDay(today)
        tomorrowNyTs = self.dailyPriceProvider.timestampToNyDay(tomorrow)

        # Delistings are stored in the ALL_TICKERS_FILE
        for portfolio in self.userPortfolios.values():
            for ticker in list(portfolio.positions.keys()):
                profile = self.tickerDataProvider.getTickerProfile(ticker)
                delistDate = profile.delistDate if profile is not None else None
                if delistDate is None:
                    continue

                delistDateNyTs = self.dailyPriceProvider.timestampToNyDay(delistDate)
                if delistDateNyTs > tomorrowNyTs:
                    # Company is still trading
                    continue

                # Ticker is delisting tomorrow, liquidate at the close price
                # If the close price is below $1, liquidate at $0  (the company likely went bankrupt, not hugely accurate but there is not enough data)
                lastOhlc = self.dailyPriceProvider.getSingleDayTickerData(ticker, today)
                if lastOhlc is None or lastOhlc["close"] < 1:
                    closePrice = 0.0
                    portfolio.addToLog(today, ticker, "Delisting", f"Ceased trading today on {profile.exchange} due to bankruptcy. Position liquidated in following order.")
                else:
                    closePrice = lastOhlc["close"]
                    portfolio.addToLog(today, ticker, "Delisting", f"Ceased trading today on {profile.exchange}. Position liquidated at close price ${closePrice:.2f} in following order.")

                order = MarketOrder(ticker, OrderSide.SELL, quantity=-1)
                order.setFillParams(closePrice, today)
                order.setOrderStatus(OrderStatus.FILLED)
                portfolio.executeTrade(order)

                

    def runNextDay(self):
        if not self.started:
            self.started = True
            currentDate = self.currentDate.normalize()
        else:
            currentDate = (self.currentDate + pd.DateOffset(days=1)).normalize()

        if currentDate > self.endDate:
            return False

        while not self.isTradingDay(currentDate):
            currentDate = (currentDate + pd.DateOffset(days=1)).normalize()
            if currentDate > self.endDate:
                return False

        self.currentDate = currentDate
        
        # First run the full day, and then at close, run the corporate actions, and then check for delistings tomorrow
        self.processDaysTrades(currentDate)
        self.processDaysCorporateActions(currentDate)
        self.processTomorrowsDelistings(currentDate)

        return True
    


    def getCurrentPrice(self, ticker):
        ohlc = self.dailyPriceProvider.getSingleDayTickerData(ticker, self.currentDate)
        if ohlc is None:
            return float("nan")
        return ohlc["close"]        # Use close price as the current price



    def getPortfolioValueAtCurrentDate(self, username):
        portfolio = self.userPortfolios.get(username, None)
        if portfolio is None:
            return None
        
        totalValue = portfolio.cash
        pfDict = {
            "cash": portfolio.cash,
            "positions": {}
        }
        
        for ticker, position in portfolio.positions.items():
            currentPrice = self.getCurrentPrice(ticker)
            totalValue += position.quantity * currentPrice

            positionAbsReturn = (currentPrice - position.averagePrice) * position.quantity
            positionPctReturn = (currentPrice / position.averagePrice - 1) * 100 if position.averagePrice != 0 else 0

            pfDict["positions"][ticker] = {
                "quantity": position.quantity, 
                "currentPrice": currentPrice, 
                "mktPrice": currentPrice,
                "averagePrice": position.averagePrice,
                "value": position.quantity * currentPrice,
                "absReturn": positionAbsReturn,
                "pctReturn": positionPctReturn
            }

        pfDict["totalValue"] = totalValue
        pfDict["startingCash"] = portfolio.startingCash

        pfDict["absReturn"] = totalValue - portfolio.startingCash
        pfDict["pctReturn"] = (totalValue / portfolio.startingCash - 1) * 100 if portfolio.startingCash != 0 else 0

        return pfDict
    
