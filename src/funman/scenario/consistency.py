"""
This submodule defined the Parameter Synthesis scenario.
"""
from funman.model import Query
from funman.model.bilayer import Bilayer
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.search import SMTCheck, SearchConfig
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
        query: Query,
        smt_encoder=None,
        config: Dict = None,
    ) -> None:
        super(ConsistencyScenario, self).__init__()
        self.smt_encoder = smt_encoder
        self.model_encoding = None
        self.query_encoding = None

        self.searches = []
        self.model = model
        self.query = query

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

        self.encode()

        if config.search is None:
            search = SMTCheck()
        else:
            search = config.search()
        
        if search not in self.searches:
            self.searches.append(search)
            
        result = search.search(self, config=config)

        return ConsistencyScenarioResult(result, self)

    def encode(self):
        self.model_encoding = self.smt_encoder.encode_model(self.model)
        self.query_encoding = self.smt_encoder.encode_query(
            self.model_encoding, self.query
        )
        return self.model_encoding, self.query_encoding


class ConsistencyScenarioResult(AnalysisScenarioResult):
    """
    ConsistencyScenarioResult result, which includes the consistency flag and
    search statistics.
    """

    def __init__(self, result: Any, scenario: ConsistencyScenario) -> None:
        super().__init__()
        self.consistent = result
        self.scenario = scenario

    def parameters(self):
        if self.consistent:
            parameters = self.scenario.smt_encoder.parameter_values(
                self.scenario.model, self.consistent
            )
            return parameters
        else:
            raise Exception(
                f"Cannot get paratmer values for an inconsistent scenario."
            )

    def dataframe(self, interpolate="linear"):
        if self.consistent:
            timeseries = self.scenario.smt_encoder.symbol_timeseries(
                self.scenario.model_encoding, self.consistent
            )
            df = pd.DataFrame.from_dict(timeseries)
            if interpolate:
                df = df.interpolate(method=interpolate)
            return df
        else:
            raise Exception(f"Cannot plot result for an inconsistent scenario.")

    def plot(self, **kwargs):
        if self.consistent:
            self.dataframe().plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(f"Cannot plot result for an inconsistent scenario.")
