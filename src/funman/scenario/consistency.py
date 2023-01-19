"""
This submodule defines a consistency scenario.  Consistency scenarios specify an existentially quantified model.  If consistent, the solution assigns any unassigned variable, subject to their bounds and other constraints.  
"""
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from pysmt.solvers.solver import Model as pysmt_Model

from funman.model.model import Model
from funman.model.query import Query
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.search.search import SearchConfig, SearchEpisode
from funman.search.smt_check import SMTCheck
from funman.translate import Encoder
from funman.translate.translate import Encoding


class ConsistencyScenario(AnalysisScenario):
    """
    The ConsistencyScenario class is an Analysis Scenario that analyzes a Model to find assignments to all variables, if consistent.

    Parameters
        ----------
        model : Model
            model to check
        query : Query
            model query
        smt_encoder : Encoder, optional
            method to encode the scenario, by default None
    """

    model: Model
    query: Query
    smt_encoder: Encoder = None
    model_encoding: Encoding = None
    query_encoding: Encoding = None
    searches: List[SearchEpisode] = []

    def solve(
        self, config: SearchConfig = None
    ) -> "ConsistencyScenarioResult":
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

        self._encode()

        if config.search is None:
            search = SMTCheck()
        else:
            search = config.search()

        if search not in self.searches:
            self.searches.append(search)

        result = search.search(self, config=config)

        return ConsistencyScenarioResult(result, self)

    def _encode(self):
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

    consistent: pysmt_Model
    scenario: ConsistencyScenario
    _query_satisfied: bool = None

    class Config:
        arbitrary_types_allowed = True

    def query_satisfied(self):
        if self._query_satisfied is None:
            self._query_satisfied = self.consistent is not None

    def _parameters(self):
        if self.consistent:
            parameters = self.scenario.smt_encoder.parameter_values(
                self.scenario.model, self.consistent
            )
            return parameters
        else:
            raise Exception(
                f"Cannot get parameter values for an inconsistent scenario."
            )

    def dataframe(self, interpolate="linear"):
        """
        Extract a timeseries as a Pandas dataframe.

        Parameters
        ----------
        interpolate : str, optional
            interpolate between time points, by default "linear"

        Returns
        -------
        pandas.DataFrame
            the timeseries

        Raises
        ------
        Exception
            fails if scenario is not consistent
        """
        if self.consistent:
            timeseries = self.scenario.smt_encoder.symbol_timeseries(
                self.scenario.model_encoding, self.consistent
            )
            df = pd.DataFrame.from_dict(timeseries)
            if interpolate:
                df = df.interpolate(method=interpolate)
            return df
        else:
            raise Exception(
                f"Cannot create dataframe for an inconsistent scenario."
            )

    def plot(self, **kwargs):
        """
        Plot the results in a matplotlib plot.

        Raises
        ------
        Exception
            failure if scenario is not consistent.
        """
        if self.consistent:
            self.dataframe().plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(
                f"Cannot plot result for an inconsistent scenario."
            )

    def __repr__(self) -> str:
        return str(self.consistent)
