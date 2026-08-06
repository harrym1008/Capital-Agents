from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    FAILED = "FAILED"


class Order:
    def __init__(self, ticker, side, quantity=None, cashValue=None):
        self.ticker = ticker
        self.quantity = quantity
        self.cashValue = cashValue
        self.side = side

        self.isQuantityBased = quantity is not None
        if quantity is None and cashValue is not None:
            self.isQuantityBased = False
        elif quantity is not None and cashValue is None:
            self.isQuantityBased = True
        else:
            raise ValueError("Order must be either quantity-based or cash-value-based, not both or neither")

        self.status = OrderStatus.PENDING
        self.fillPrice = None
        self.fillTimestamp = None
        self.submittedTimestamp = None
        
    def setSubmittedTimestamp(self, timestamp):
        self.submittedTimestamp = timestamp
            
    def setFillParams(self, price, timestamp):
        self.fillPrice = price
        self.fillTimestamp = timestamp

    def setOrderStatus(self, status):
        self.status = status

    def getOrderString(self):
        if self.fillPrice is None:
            if self.isQuantityBased:
                return f"*{self.status.value}* {self.__class__.__name__} {self.side.value} {self.quantity:.3f} shares of {self.ticker}"
            else:
                return f"*{self.status.value}* {self.__class__.__name__} {self.side.value} ${self.cashValue:.2f} worth of shares of {self.ticker}"

        if self.isQuantityBased:
            quantity = self.quantity
            cashValue = self.quantity * self.fillPrice
        else:
            quantity = self.cashValue / self.fillPrice
            cashValue = self.cashValue

        return f"*{self.status.value}* {self.__class__.__name__} {self.side.value} {quantity:.3f} shares of {self.ticker}, ${self.fillPrice:.2f} each (totalling ${cashValue:.2f})"


class MarketOrder(Order):
    pass
   

class LimitOrder(Order):
    def __init__(self, ticker, side, limitPrice, quantity=None, cashValue=None):
        super().__init__(ticker, side, quantity=quantity, cashValue=cashValue)
        self.limitPrice = limitPrice


class StopOrder(Order):
    def __init__(self, ticker, side, stopPrice, quantity=None, cashValue=None):
        super().__init__(ticker, side, quantity=quantity, cashValue=cashValue)
        self.stopPrice = stopPrice
        

class StopLimitOrder(Order):
    def __init__(self, ticker, side, stopPrice, limitPrice, quantity=None, cashValue=None):
        super().__init__(ticker, side, quantity=quantity, cashValue=cashValue)
        self.stopPrice = stopPrice
        self.limitPrice = limitPrice
        self.stopTriggered = False

