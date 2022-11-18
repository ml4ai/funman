"""
This submodule defined the Parameter Synthesis scenario.
"""
from model2smtlib.bilayer.translate import Bilayer
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.examples.chime import CHIME
from funman.model import Model, Parameter, Query
from funman.parameter_space import ParameterSpace
from funman.search import BoxSearch, SMTCheck, SearchConfig
from pysmt.fnode import FNode
from pysmt.shortcuts import get_free_variables, And
from typing import Any, Dict, List, Union
from pysmt.solvers.solver import Model as pysmtModel
import pandas as pd
import matplotlib.pyplot as plt


class ConsistencyScenario(AnalysisScenario):
    """ """

    def __init__(
        self,
        model: Union[str, FNode, Bilayer],
        search=None,
        config: Dict = None,
    ) -> None:
        super(ConsistencyScenario, self).__init__()
        if search is None:
            search = SMTCheck()
        self.search = search
        self.model = model

    def solve(self, config: SearchConfig = None) -> "ConsistencyScenarioResult":
        """
        Check model consistency.

        Parameters
        ----------
        config : SearchConfig
            Options for the Search algorithm.

        Returns
        -------
        result
            ConsistencyScenarioResult indicating whether the model is consistent.
        """
        if config is None:
            config = SearchConfig()

        result = self.search.search(self, config=config)

        return ConsistencyScenarioResult(result, self)


class ConsistencyScenarioResult(AnalysisScenarioResult):
    """
    ConsistencyScenarioResult result, which includes the consistency flag and
    search statistics.
    """

    def __init__(self, result: Any, scenario: ConsistencyScenario) -> None:
        super().__init__()
        self.consistent = result
        self.scenario = scenario

    def plot(self):
        timeseries = self.scenario.model.symbol_timeseries(self.consistent)
        df = pd.DataFrame.from_dict(timeseries)
        df.interpolate(method="linear").plot(marker="o")
        plt.show(block=False)
