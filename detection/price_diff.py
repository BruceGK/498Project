from glob import glob
from typing import Any
import os
import pandas as pd
import torch
import numpy as np
from matplotlib.dates import DateFormatter

from detection.measure import Measure
from realism.realism_utils import make_orderbook_for_analysis
from util.formatting.convert_order_stream import extract_events_from_stream
from matplotlib import pyplot as plt


def plot(data, y1='PRICE_IMP', y2='PRICE_NO_IMP'):
    fig, ax = plt.subplots(figsize=(13, 9))

    ax.set_ylabel("Price (cents)")
    ax.set_xlabel("Time of day")

    fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(fmt)
    plt.title('Price comparison')

    # x = self.impact_data.index
    # y = self.impact_data['PRICE_IMP']
    # plt.plot(x, y, label='impact')
    #
    # x = self.no_impact_data.index
    # y = self.no_impact_data['PRICE_NO_IMP']
    # plt.plot(x, y, label='no impact')

    data = data[100:]
    x = data.index
    y = data[y1]
    plt.plot(x, y, label='impact')
    y = data[y2]
    plt.plot(x, y, label='no impact')

    plt.legend()

    fig.savefig(f'./compare.png', format='png', dpi=300, transparent=False,
                bbox_inches='tight',
                pad_inches=0.03)
    plt.show()


class PriceMeasure(Measure):
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
        # Should only have one result, just a demonstration of how to find multiple files
        # impacts = sorted(glob(os.path.join(".", "log",  log_dir, self.impact_dir, "fundamental_*.bz2")))
        # no_impacts = sorted(glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "fundamental_*.bz2")))
        impact_ex = sorted(glob(os.path.join(".", "log", log_dir, self.impact_dir, "EXCHANGE_*.bz2")))[0]
        impact_ob = sorted(glob(os.path.join(".", "log", log_dir, self.impact_dir, "ORDERBOOK_*.bz2")))[0]
        no_impact_ex = sorted(glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "EXCHANGE_*.bz2")))[0]
        no_impact_ob = sorted(glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "ORDERBOOK_*.bz2")))[0]
        fundamental = glob(os.path.join(".", "log", log_dir, self.no_impact_dir, "fundamental_*.bz2"))[0]

        impact_orderbook = make_orderbook_for_analysis(impact_ex, impact_ob, num_levels=1,
                                                          hide_liquidity_collapse=False)
        impact_orderbook = impact_orderbook.loc[impact_orderbook.TYPE == "ORDER_EXECUTED"]
        imp = impact_orderbook[['MID_PRICE']].resample('ms').mean()

        no_impact_orderbook = make_orderbook_for_analysis(no_impact_ex, no_impact_ob, num_levels=1,
                                                       hide_liquidity_collapse=False)
        no_impact_orderbook = no_impact_orderbook.loc[no_impact_orderbook.TYPE == "ORDER_EXECUTED"]
        no_imp = no_impact_orderbook[['MID_PRICE']].resample('ms').mean()

        fund_data = pd.read_pickle(fundamental)
        fund_data = fund_data.resample('ms').mean()

        # impact_data = pd.read_pickle(impact_ex)
        # impact_data = extract_events_from_stream(impact_data.reset_index(), 'ORDER_EXECUTED')
        # impact_data = impact_data[['TIMESTAMP', 'PRICE']].set_index('TIMESTAMP')
        # imp = impact_data.resample('ms').mean()
        #
        # no_impact_data = pd.read_pickle(no_impact_ex)
        # no_impact_data = extract_events_from_stream(no_impact_data.reset_index(), 'ORDER_EXECUTED')
        # no_impact_data = no_impact_data[['TIMESTAMP', 'PRICE']].set_index('TIMESTAMP')
        # no_imp = no_impact_data.resample('ms').mean()

        data = imp.join(no_imp, how='outer', lsuffix='_IMP', rsuffix='_NO_IMP').fillna(method='ffill')
        data = data.join(fund_data, how='outer').fillna(method='ffill')

        self.fundamental = data['FundamentalValue'].to_numpy() / 100

        # plot(data, 'MID_PRICE_IMP', 'MID_PRICE_NO_IMP')
        # return data['PRICE_IMP'].to_numpy(), data['PRICE_NO_IMP'].to_numpy()
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

        # impact_chages = self.impact_data[1:] - self.impact_data[:-1]
        # no_impact_chages = self.no_impact_data[1:] - self.no_impact_data[:-1]
        # diff = np.abs(impact_chages).sum() - np.abs(no_impact_chages).sum()
        # print(f'{diff.item()} {np.abs(impact_chages).sum() / np.abs(no_impact_chages).sum()}')

        # diff = self.impact_data - self.no_impact_data
        # print(f'diff {self.impact_dir} {self.no_impact_dir}: {diff.sum().item()}')
        # if diff.sum() > 0:
        #     diff = diff[diff > 0].sum()
        # else:
        #     diff = diff[diff < 0].sum()

        return torch.tensor(diff / diff2)
