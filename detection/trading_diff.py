from glob import glob
from typing import Any
import os
import pandas as pd
import torch

from detection.measure import Measure


class TradingMeasure(Measure):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.impact_dir = log_dir
        self.no_impact_dir = f"no_{log_dir}"
        self.impact_data, self.no_impact_data = self.load()

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
        impacts = sorted(glob(os.path.join(".", "log", self.impact_dir, "ExchangeAgent*.bz2")))
        no_impacts = sorted(glob(os.path.join(".", "log", self.no_impact_dir, "ExchangeAgent*.bz2")))
        for imp, no_imp in zip(impacts, no_impacts):
            imp = pd.read_pickle(imp)
            no_imp = pd.read_pickle(no_imp)
        return imp, no_imp


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
        diff = self.impact_data - self.no_impact_data
        return torch.tensor(diff.values)
