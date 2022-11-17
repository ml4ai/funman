"""
This submodule defined the Parameter Synthesis scenario.
"""
from . import AnalysisScenario, AnalysisScenarioResult
from funman.examples.chime import CHIME
from funman.model import Model, Parameter, Query
from funman.parameter_space import ParameterSpace
from funman.search import BoxSearch, SMTCheck, SearchConfig
from pysmt.fnode import FNode
from pysmt.shortcuts import get_free_variables, And
from typing import Any, Dict, List, Union
from pysmt.solvers.solver import Model as pysmtModel


class ConsistencyScenario(AnalysisScenario):
    """ """

    def __init__(
        self,
        model: Union[str, FNode],
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

    def _split_symbol(self, symbol):
        return symbol.symbol_name().rsplit("_", 1)

    def timeseries(self):
        series = {}
        if isinstance(self.consistent, pysmtModel):
            vars = list(self.scenario.model.formula.get_free_variables())
            # vars.sort(key=lambda x: x.symbol_name())
            for var in vars:
                var_name, timepoint = self._split_symbol(var)
                if var_name not in series:
                    series[var_name] = {}
                series[var_name][timepoint] = float(
                    self.consistent.get_py_value(var)
                )
        a_series = {}
        max_t = max(
            [max([int(k) for k in tps.keys()]) for _, tps in series.items()]
        )
        # a_series["index"] = list(range(0, max_t+1))
        for var, tps in series.items():

            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                vals[int(t)] = v
            a_series[var] = vals
        return a_series
