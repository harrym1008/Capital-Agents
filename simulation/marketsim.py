from enum import Enum
from datetime import timedelta

import pandas as pd

from dataquery.pricemgr import DailyPricesClient
from collectors.mktcalendar import MarketCalendar 
from collectors.constants import NEW_YORK
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
    def __init__(self, startDate, endDate):
        self.startDate = pd.Timestamp(startDate, tz=NEW_YORK)
        self.endDate = pd.Timestamp(endDate, tz=NEW_YORK) - timedelta(days=1)    
        self.currentDate = self.startDate
        self.started = False

        self.marketCalendar = MarketCalendar(startDate, endDate)
        self.dailyPriceClient = DailyPricesClient(self.startDate, self.endDate)

        self.users: list[str] = []                          # Set of all users in the simulation
        self.userPortfolios: dict[str, Portfolio] = {}      # Maps user to their portfolios. The portfolio itself is a dict mapping ticker to Position
        
        self.pendingOrders: list[UserOrder] = []     # Pending UserOrders that have not been filled yet
        self.ordersArchive: list[UserOrder] = []     # All UserOrders that have been processed (filled or failed)


    def initialiseUser(self, username, initialCash=1_000_000.0):
        if username in self.users:
            return

        self.users.append(username)
        self.userPortfolios[username] = Portfolio(initialCash)


    # testingRun should only be true if usernames are a list of tickers!
    def initialiseUsers(self, usernames, initialCash=1_000_000.0, testingRun=False):
        for username in usernames:
            self.initialiseUser(username, initialCash)

            if testingRun:
                self.addOrder(MarketOrder(username, OrderSide.BUY, cashValue=initialCash), username)


    def addOrder(self, order, username):
        userOrder = UserOrder(order, username)
        self.pendingOrders.append(userOrder)
        order.setSubmittedTimestamp(self.currentDate)


    def buildIntradayPath(self, ohlcData):
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
        requiredTickers = set()
        for userOrder in self.pendingOrders:
            requiredTickers.add(userOrder.order.ticker)

        # Process each pending order
        executedThisDay = []
        remainingOrders = list(self.pendingOrders)

        # Get the OHLC price data for this date
        for ticker in requiredTickers:
            ohlc = self.dailyPriceClient.getSingleDayTickerData(ticker, date)
            segments = self.buildIntradayPath(ohlc)

            for segmentStart, segmentEnd in segments:
                executed, remaining = self.processSegment(segmentStart, segmentEnd, remainingOrders, ticker)
                executedThisDay.extend(executed)
                remainingOrders = remaining

        
        # Execute and archive processed orders
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
        todaysActions = self.dailyPriceClient.getSingleDayCorpActions(date)
        dateNyTs = self.dailyPriceClient.timestampToNyDay(date)
        
        for portfolio in self.userPortfolios.values():
            for _, action in todaysActions.iterrows():
                ticker = action["ticker"]
                if ticker not in portfolio.positions:
                    continue

                position = portfolio.positions[ticker]
                actionType = action["actionType"]

                if actionType in ["forward_split", "reverse_split"]:
                    sharesMult = action["newRate"] / action["oldRate"]
                    priceMult = action["oldRate"] / action["newRate"]
                    position.quantity *= sharesMult
                    position.averagePrice *= priceMult

                    portfolio.addToLog(date, f"CorpAction: {ticker} {action['newRate']} stock split. Holding now: {position.quantity} @ ${position.averagePrice:.2f} avg.")

                elif actionType in ["cash_dividend", "stock_divided"]:
                    amount = action["rate"]
                    payDate = action["payDate"]
                    dividendType = Dividend.Type.CASH if actionType == "cash_dividend" else Dividend.Type.STOCK

                    dividend = Dividend(ticker, amount, position.quantity, payDate, dividendType)
                    portfolio.potentialDividends.append(dividend)

                    if dividend.dividendType == Dividend.Type.CASH:
                        portfolio.addToLog(date, f"CorpAction: {ticker} cash dividend ex-date reached - ${amount:.2f} per share. Will be paid on {payDate}.")
                    else:
                        portfolio.addToLog(date, f"CorpAction: {ticker} stock dividend ex-date reached - {amount} per share. Will be paid on {payDate}.")

                # TODO: Handle spin-offs, mergers and name-changes

                elif actionType == "worthless_removal":
                    portfolio.addToLog(date, f"CorpAction: {ticker} removed from exchange due to worthlessness. Position liquidated in following order.")

                    order = MarketOrder(ticker, OrderSide.SELL, quantity=-1)
                    order.setFillParams(0.0, date)
                    portfolio.executeTrade(order)


        # Finally process dividends that are payable today

        for portfolio in self.userPortfolios.values():
            for dividend in portfolio.potentialDividends:

                payDateNyTs = self.dailyPriceClient.timestampToNyDay(dividend.payDate)

                if payDateNyTs == dateNyTs:
                    paymentAmount, dividendType = dividend.calculateDividend()

                    if paymentAmount == 0:
                        continue        # All payable shares were sold

                    if dividendType == Dividend.Type.CASH:
                        portfolio.cash += paymentAmount
                        portfolio.addToLog(date, f"DividendPayment: Received cash dividend for {dividend.ticker} - ${paymentAmount:.2f}.")

                    else:   # Stock dividend
                        position = portfolio.positions.get(dividend.ticker)
                        if position is None:
                            portfolio.positions[dividend.ticker] = Position(dividend.ticker, paymentAmount, 0)
                        else:
                            position.increasePosition(paymentAmount, 0)
                        
                   
                elif payDateNyTs < dateNyTs:
                    # Expired dividend
                    portfolio.potentialDividends.remove(dividend)





    def runNextDay(self):
        if not self.started:
            self.started = True
            currentDate = self.currentDate
        else:
            currentDate = self.currentDate + timedelta(days=1)

        if currentDate > self.endDate:
            return False

        while not self.isTradingDay(currentDate):
            currentDate += timedelta(days=1)
            if currentDate > self.endDate:
                return False

        self.currentDate = currentDate
        
        # First run the full day, and then at close, run the corporate actions
        self.processDaysTrades(currentDate)
        self.processDaysCorporateActions(currentDate)


        return True
    

    def concludeSimulation(self):
        currentDate = self.endDate
        while not self.isTradingDay(currentDate):
            currentDate -= timedelta(days=1)
        self.processDaysTrades(currentDate)



    def getCurrentPrice(self, ticker):
        ohlc = self.dailyPriceClient.getSingleDayTickerData(ticker, self.currentDate)
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
    

    def prettyPrintAllPortfolios(self):
        from simulation.prettysim import prettyPrintPortfolio
        for user in self.users:
            prettyPrintPortfolio(self, user)
