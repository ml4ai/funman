"""
This submodule defines a consistency scenario.  Consistency scenarios specify an existentially quantified model.  If consistent, the solution assigns any unassigned variable, subject to their bounds and other constraints.  
"""
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from pysmt.solvers.solver import Model as pysmt_Model

from funman.model.bilayer import BilayerModel
from funman.model.encoded import EncodedModel
from funman.model.model import Model
from funman.model.query import Query, QueryFunction, QueryLE
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.search.search import Search, SearchConfig, SearchEpisode
from funman.search.smt_check import SMTCheck
from funman.translate import Encoder
from funman.translate.translate import Encoding


class ConsistencyScenario(AnalysisScenario, BaseModel):
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

    class Config:
        underscore_attrs_are_private = True

    model: Union[BilayerModel, EncodedModel]
    query: Union[QueryLE, QueryFunction]
    _smt_encoder: Encoder = None
    _model_encoding: Encoding = None
    _query_encoding: Encoding = None
    _searches: List[SearchEpisode] = []

    def solve(self, config: SearchConfig = None) -> "AnalysisScenarioResult":
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

        if config.search is None:
            search = SMTCheck()
        else:
            search = config.search()

        self._encode(search)

        if search not in self._searches:
            self._searches.append(search)

        result = search.search(self, config=config)

        scenario_result = ConsistencyScenarioResult(
            scenario=self, _consistent=result
        )
        return scenario_result

    def _encode(self, search: Search):
        if self._smt_encoder is None:
            self._smt_encoder = self.model.default_encoder()
        self.model.initialize()
        self._model_encoding = self._smt_encoder.encode_model(self.model)
        self._query_encoding = self._smt_encoder.encode_query(
            self._model_encoding, self.query
        )
        return self._model_encoding, self._query_encoding


class ConsistencyScenarioResult(AnalysisScenarioResult):
    """
    ConsistencyScenarioResult result, which includes the consistency flag and
    search statistics.
    """

    scenario: ConsistencyScenario
    _consistent: pysmt_Model
    _query_satisfied: bool = None

    class Config:
        arbitrary_types_allowed = True

    def query_satisfied(self):
        if self._query_satisfied is None:
            self._query_satisfied = self._consistent is not None

    def _parameters(self):
        if self._consistent:
            parameters = self.scenario._smt_encoder.parameter_values(
                self.scenario.model, self._consistent
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
        if self._consistent:
            timeseries = self.scenario._smt_encoder.symbol_timeseries(
                self.scenario._model_encoding, self._consistent
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
        if self._consistent:
            self.dataframe().plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(
                f"Cannot plot result for an inconsistent scenario."
            )

    def __repr__(self) -> str:
        return str(self.consistent)
