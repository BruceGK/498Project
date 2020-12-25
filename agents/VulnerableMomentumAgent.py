from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np


class MomentumAgent(TradingAgent):
    """
    Simple Trading Agent that compares the 20 past mid-price observations with the 50 past observations and places a
    buy limit order if the 20 mid-price average >= 50 mid-price average or a
    sell limit order if the 20 mid-price average < 50 mid-price average
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 min_size, max_size, wake_up_freq='60s',
                 subscribe=True, log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = self.random_state.randint(self.min_size, self.max_size)
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list, self.avg_20_list, self.avg_50_list = [], [], []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.buyOrders = 0
        self.sellOrders = 0

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=10, freq=10e9)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bids, asks = self.getKnownBidAsk(self.symbol, best=False)
            if bids and asks:
                buyPercent = MomentumAgent.buyPressure(bids, asks)
                askPercent = 1 - buyPercent
                bid = bids[0][0]
                ask = asks[0][0]
                self.placeOrders(bid, ask, buyPercent, askPercent)
                self.setWakeup(currentTime + self.getWakeFrequency())
                self.state = 'AWAITING_WAKEUP'
        elif self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':
            bids, asks = self.getKnownBidAsk(self.symbol, best=False)
            if bids and asks:
                buyPercent = MomentumAgent.buyPressure(bids, asks)
                askPercent = 1 - buyPercent
                bid = bids[0][0]
                ask = asks[0][0]
                self.placeOrders(bid, ask, buyPercent, askPercent)
            self.state = 'AWAITING_MARKET_DATA'

    def placeOrders(self, bid, ask, bidPercent, askPercent):
        """ Momentum Agent actions logic """
        if bid and ask:
            self.mid_list.append((bid + ask) / 2)
            if len(self.mid_list) > 20: self.avg_20_list.append(MomentumAgent.ma(self.mid_list, n=20)[-1].round(2))
            if len(self.mid_list) > 50: self.avg_50_list.append(MomentumAgent.ma(self.mid_list, n=50)[-1].round(2))
            if len(self.avg_20_list) > 0 and len(self.avg_50_list) > 0:
                #If the averages of the last 20 are bigger than the averages of the last 50 we buy because the momentum is still upwards
                if self.avg_20_list[-1] >= self.avg_50_list[-1]:
                    if bidPercent > .5:
                        self.buyOrders += 1
                        # print("B" + str(self.buyOrders))
                        self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=True, limit_price=ask)
                #If the average of the last 20 are smaller than the averages of the last 50 we sell
                elif self.avg_20_list[-1] < self.avg_50_list[-1]:
                    if askPercent > .5:
                        self.sellOrders += 1
                        # print("S" + str(self.sellOrders))
                        self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=False, limit_price=bid)
    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    #Returns the percent of shares that are buy orders in the orderbook
    @staticmethod
    def buyPressure(bids, asks):
        totalBidVol = 0
        totalAskVol = 0
        for b in bids:
            totalBidVol += b[1]
        for a in asks:
            totalAskVol += a[1]
        
        if totalBidVol + totalAskVol == 0:
            return None
        return totalBidVol/(totalBidVol+totalAskVol)