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

def plot(data, y1='PRICE_IMP', y2='PRICE_NO_IMP'):
    fig, ax = plt.subplots(figsize=(13, 9))

    ax.set_ylabel("Volume (cents)")
    ax.set_xlabel("Time of day")

    fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(fmt)
    plt.title('Volume comparison')


    data = data[100:]
    x = data.index
    y = data[y1]
    plt.plot(x, y, label='impact')
    y = data[y2]
    plt.plot(x, y, label='no impact')

    plt.legend()

    fig.savefig(f'./volumn compare.png', format='png', dpi=300, transparent=False,
                bbox_inches='tight',
                pad_inches=0.03)
    plt.show()


class TradingMeasure(Measure):
    def __init__(self, log_dir, impact, no_impact):
        super().__init__(log_dir, impact, no_impact)

    def load(self, log_dir):
        """
        Load the data from `self.impact_dir` and `self.no_impact_dir` and
        initialize `self.impact_data` and `self.no_impact_data` .

        Raises
        -----
        FileNotFoundError
            When can't find log directories
        NotImplementedError
            If not implemented by child class
        Returns
        -----
        impact_data, no_impact_data : (Any, Any)
            Data loaded from impact and non-impact simulation
        """
        impact_ex = sorted(glob(os.path.join(".", "log", log_dir, self.impact_dir, "EXCHANGE_*.bz2")))[0]
        impact_ob = sorted(glob(os.path.join(".", "log", log_dir, self.impact_dir, "ORDERBOOK_*.bz2")))[0]
        no_impact_ex = sorted(glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "EXCHANGE_*.bz2")))[0]
        no_impact_ob = sorted(glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "ORDERBOOK_*.bz2")))[0]
        fundamental = glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "fundamental_*.bz2"))[0]

        im_processed_orderbook, im_transacted_orders, im_cleaned_orderbook = self.create_orderbooks(impact_ex, impact_ob)

        no_im_processed_orderbook, no_im_transacted_orders, no_im_cleaned_orderbook = self.create_orderbooks(no_impact_ex,
                                                                                                    no_impact_ob)

        fund_data = pd.read_pickle(fundamental)
        fund_data = fund_data.resample('ms').mean()

        data = im_transacted_orders.join(no_im_transacted_orders, how='outer', lsuffix='_IMP', rsuffix='_NO_IMP').fillna(method='ffill')
        data = data.join(fund_data, how='outer').fillna(method='ffill')

        self.fundamental = data['FundamentalValue'].to_numpy() / 100


        return data['MID_PRICE_IMP'].to_numpy(), data['MID_PRICE_NO_IMP'].to_numpy()

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
        # diff = np.abs(self.impact_data - self.no_impact_data).sum()

        diff = np.abs(self.impact_data - self.fundamental).sum()

        diff2 = np.abs(self.no_impact_data - self.fundamental).sum()


        return torch.tensor(diff / diff2)

    def create_orderbooks(exchange_path, ob_path):
        """ Creates orderbook DataFrames from ABIDES exchange output file and orderbook output file. """

        print("Constructing orderbook...")
        processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                          hide_liquidity_collapse=False)
        cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
        transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]
        transacted_orders['SIZE'] = transacted_orders['SIZE'] / 2

        return processed_orderbook, transacted_orders, cleaned_orderbook
