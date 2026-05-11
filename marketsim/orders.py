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
        
    def shouldFill(self, currentPrice):
        raise NotImplementedError("Use a subclass of 'Order', do not use 'Order' itself")
    
    def setFillParams(self, price, timestamp):
        self.fillPrice = price
        self.fillTimestamp = timestamp

    def setOrderStatus(self, status):
        self.status = status


class MarketOrder(Order):
    pass
    

class LimitOrder(Order):
    def __init__(self, ticker, quantity, side, limitPrice):
        super().__init__(ticker, quantity, side)
        self.limitPrice = limitPrice


class StopOrder(Order):
    def __init__(self, ticker, quantity, side, stopPrice):
        super().__init__(ticker, quantity, side)
        self.stopPrice = stopPrice
        

class StopLimitOrder(Order):
    def __init__(self, ticker, quantity, side, stopPrice, limitPrice):
        super().__init__(ticker, quantity, side)
        self.stopPrice = stopPrice
        self.limitPrice = limitPrice
        self.stopTriggered = False

