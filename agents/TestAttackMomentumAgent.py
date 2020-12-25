from agents.AttackMomentumAgent import AttackMomentumAgent
import numpy as np
import mock
import unittest


class TestVulnerableMomentumAgent(unittest.TestCase):

    def setUp(self):
        self.agent = AttackMomentumAgent(id=1,
                             name="Test Agent",
                             type="MomentumAgent",
                             symbol="testSymbol",
                             starting_cash=100,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=False,
                             random_state=np.random.RandomState(seed=1))

    
    def test_BuyPressure(self):
        bids = []
        asks = []
        self.assertEqual(None,AttackMomentumAgent.buyPressure(bids, asks))
        bids = [[0, 10], [1, 10]]
        asks = []
        self.assertAlmostEqual(1,AttackMomentumAgent.buyPressure(bids, asks), places=2)
        asks = [[0,10]]
        self.assertAlmostEqual(.666,AttackMomentumAgent.buyPressure(bids, asks), places=2)
        asks = [[0,100], [1, 100]]
        self.assertAlmostEqual(.0909,AttackMomentumAgent.buyPressure(bids, asks), places=2)

    def test_BuyOrders(self):
        self.agent.placeLimitOrder = mock.MagicMock()
        self.agent.placeOrders(10, 11, .6)
        self.agent.placeOrders(10, 11, .4)
        self.agent.placeLimitOrder.assert_not_called()
        self.agent.placeOrders(10, 11, .46)
        self.agent.placeLimitOrder.assert_called_once_with(self.agent.symbol, quantity=self.agent.size, is_buy_order=True, limit_price=11)

    def test_SellOrders(self):
        self.agent.placeLimitOrder = mock.MagicMock()
        self.agent.placeOrders(10, 11, .6)
        self.agent.placeOrders(10, 11, .4)
        self.agent.placeLimitOrder.assert_not_called()
        self.agent.placeOrders(10, 11, .54)
        self.agent.placeLimitOrder.assert_called_once_with(self.agent.symbol, quantity=self.agent.size, is_buy_order=False, limit_price=10)

if __name__ == '__main__':
    unittest.main()