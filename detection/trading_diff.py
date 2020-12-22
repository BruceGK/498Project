from glob import glob
from typing import Any
import os
import pandas as pd
import torch
import numpy as np
from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF

from detection.measure import Measure


class TradingMeasure(Measure):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.ask, self.bid = self.load(log_dir)

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
        impact_data, no_impact_data : (Any, Any)
            Data loaded from impact and non-impact simulation
        """
        # Should only have one result, just a demonstration of how to find multiple files
        asks = sorted(glob(os.path.join(".", "log", log_dir, "ORDERBOOK_ABM_FULL.bz2*.bz2")))
        bids = sorted(glob(os.path.join(".", "log", log_dir, "ORDERBOOK_ABM_FULL.bz2*.bz2")))
        for ask, bid in zip(asks, bids):
            ask = pd.read_pickle(ask)
            bid = pd.read_pickle(bid)
        return ask, bid

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
        diff = np.abs(self.bid - self.ask)
        diff = diff.sum()
        return torch.tensor(diff.values)

