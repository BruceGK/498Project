from glob import glob
from typing import Any
import os
import pandas as pd
import torch
import numpy as np

from detection.measure import Measure


class PriceMeasure(Measure):
    def __init__(self, log_dir):
        super().__init__(log_dir)

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
        impacts = sorted(glob(os.path.join(".", log_dir, "impact", "fundamental_*.bz2")))
        no_impacts = sorted(glob(os.path.join(".", log_dir, "no_impact", "fundamental_*.bz2")))
        for imp, no_imp in zip(impacts, no_impacts):
            imp = pd.read_pickle(imp).to_numpy()
            no_imp = pd.read_pickle(no_imp).to_numpy()
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
        diff = np.abs(self.impact_data - self.no_impact_data)
        diff = diff.sum()
        return torch.tensor(diff)
