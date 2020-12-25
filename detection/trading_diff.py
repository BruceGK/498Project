from glob import glob
from typing import Any
import os
import pandas as pd
import torch
from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from detection.measure import Measure
import pandas as pd
import sys
sys.path.append('../..')
from matplotlib.dates import DateFormatter
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta, datetime
import argparse
import json
import matplotlib


class TradingMeasure(Measure):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.ask, self.bid = self.load(log_dir)

    def plot(data, y1='PRICE_IMP', y2='PRICE_NO_IMP'):
        fig, ax = plt.subplots(figsize=(13, 9))

        ax.set_ylabel(" Trading")
        ax.set_xlabel("Time of day")

        fmt = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(fmt)
        plt.title('Trading Difference')


        data = data[100:]
        x = data.index
        y = data[y1]
        plt.plot(x, y, label='Difference')

        plt.legend()

        fig.savefig(f'./Tradingcompare.png', format='png', dpi=300, transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.03)
        plt.show()
    def load(self,  log_dir):
        """
        Load the data from ORDERBOOK_ABM_FULL.bz2

        Raises
        -----
        FileNotFoundError
            When can't find log directories
        NotImplementedError
            If not implemented by child class
        Returns
        -----
        bid/ ask  : (Any, Any)
            Data loaded from impact and non-impact simulation

        """
        ob_path = sorted(glob(os.path.join(".", "log", log_dir, "ORDERBOOK_*.bz2")))[0]
        processed_orderbook,transacted_orders,cleaned_orderbook = self.create_orderbooks(log_dir, ob_path)
        return processed_orderbook.getInsideAsks(), processed_orderbook.getInsideBids()

    def compare(self) -> Any:
        """
        Compare the data and return the measurement.

        Returns
        -----
        Any
            The impact measurement
        Raises
        -----
        NotImplementedError
            If not implemented by child class
        """
        #Using max price - min offer
        diff = np.abs(self.ask - self.bid)
        diff = diff.sum()
        return torch.tensor(diff.values)

    def create_orderbooks(exchange_path, ob_path):
        """ Creates orderbook DataFrames from ABIDES exchange output file and orderbook output file. """

        processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                          hide_liquidity_collapse=False)
        cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
        transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]
        transacted_orders['SIZE'] = transacted_orders['SIZE'] / 2

        return processed_orderbook, transacted_orders, cleaned_orderbook
