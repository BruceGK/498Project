import os
from typing import Any


class Measure:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # self.impact_dir = os.path.join(log_dir, 'impact')
        # self.no_impact_dir = os.path.join('no_impact')
        self.impact_data, self.no_impact_data = self.load(log_dir)

    def load(self, log_dir):
        """
        Load the data from `self.impact_dir` and `self.no_impact_dir` and
        initialize `self.impact_data` and `self.no_impact_data` .

        Parameters
        -----
        log_dir
            Path to the folder contains "impact" and "no_impact" logs
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
        raise NotImplementedError

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
        raise NotImplementedError
