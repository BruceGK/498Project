from typing import Any


class Measure:
    def __init__(self, log_dir):
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
