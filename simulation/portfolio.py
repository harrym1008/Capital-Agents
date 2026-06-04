from enum import Enum
from simulation.orders import Order, OrderSide


class Position:
    def __init__(self, ticker, quantity, averagePrice):
        self.ticker = ticker
        self.quantity = quantity
        self.averagePrice = averagePrice


    def increasePosition(self, quantity, price):
        if quantity <= 0:
            raise ValueError("Quantity must be positive when increasing position")
        
        totalCost = self.quantity * self.averagePrice
        additionalCost = quantity * price
        newQuantity = self.quantity + quantity

        if newQuantity == 0:
            self.quantity = 0
            self.averagePrice = 0
            return
        
        self.quantity = newQuantity
        self.averagePrice = (totalCost + additionalCost) / newQuantity

    def decreasePosition(self, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive when decreasing position")
        if quantity > self.quantity:
            raise ValueError("Cannot decrease position by more than current quantity")
        
        self.quantity -= quantity


# Dividends must be instantiated on the ex-dividend date
class Dividend:
    class Type(Enum):
        CASH = 1
        STOCK = 2

    def __init__(self, ticker, amount, sharesHeldAtExDate, payDate, dividendType=Type.CASH):
        self.ticker = ticker
        self.amount = amount
        self.payableShares = sharesHeldAtExDate
        self.payDate = payDate
        self.dividendType = dividendType


    def calculateDividend(self):
        return (self.amount * self.payableShares, self.dividendType)



class Portfolio:
    def __init__(self, initialCash=1_000_000.0):
        self.startingCash = initialCash
        self.cash = initialCash
        self.positions: dict[str, Position] = {}
        self.potentialDividends: list[Dividend] = []

        self.log = []


    def addToLog(self, date, message):
        self.log.append(f"[{date}] {message}")


    def executeTrade(self, order: Order):
        if order.isQuantityBased:
            if order.quantity == -1:
                # -1 quantity means "liquidate entire position"
                if order.ticker in self.positions:
                    order.quantity = self.positions[order.ticker].quantity
                else:
                    raise ValueError("Cannot liquidate a position that does not exist in the portfolio")
                
            cost = order.quantity * order.fillPrice
            quantity = order.quantity
        else:
            cost = order.cashValue
            quantity = cost / order.fillPrice


        if order.side == OrderSide.BUY:
            self.cash -= cost
            if self.cash < 0:
                self.cash += cost  # Revert cash deduction
                raise ValueError("Insufficient cash to execute trade")
            
            if order.ticker not in self.positions:
                self.positions[order.ticker] = Position(order.ticker, 0, 0)
            self.positions[order.ticker].increasePosition(quantity, order.fillPrice)

        else:
            if order.ticker not in self.positions:
                raise ValueError("Cannot sell a ticker that is not in the portfolio")
            
            self.positions[order.ticker].decreasePosition(quantity)
            self.cash += cost

            if self.positions[order.ticker].quantity == 0:
                del self.positions[order.ticker]     # Clean up empty positions
        
        self.addToLog(order.fillTimestamp, f"OrderExecuted: {order.getOrderString()}")

