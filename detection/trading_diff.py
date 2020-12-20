from glob import glob
from typing import Any
import os
import pandas as pd
import torch

from detection.measure import Measure


class TradingMeasure(Measure):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.ask_dir = log_dir
        self.bid_dir = f"no_{log_dir}"
        self.ask, self.bid = self.load()

    def load(self):
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
        asks = sorted(glob(os.path.join(".", "log", self.ask_dir, "ExchangeAgent*.bz2")))
        bids = sorted(glob(os.path.join(".", "log", self.bid_dir, "ExchangeAgent*.bz2")))
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
        diff = self.ask - self.bid
        return torch.tensor(diff.values)
