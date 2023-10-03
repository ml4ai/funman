"""
This submodule defines a consistency scenario.  Consistency scenarios specify an existentially quantified model.  If consistent, the solution assigns any unassigned variable, subject to their bounds and other constraints.  
"""
import threading
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pysmt.solvers.solver import Model as pysmt_Model

from funman import ParameterSpace, Point
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.translate import Encoding


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

    _box: Optional[Encoding] = None

    @classmethod
    def get_kind(cls) -> str:
        return "consistency"

    def get_search(self, config: "FUNMANConfig") -> "Search":
        if config._search is None:
            from funman.search.smt_check import SMTCheck

            search = SMTCheck()
        else:
            search = config._search()
        return search

    def solve(
        self,
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "AnalysisScenarioResult":
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
        search = self.initialize(config)

        parameter_space, models, consistent = search.search(
            self,
            config=config,
            haltEvent=haltEvent,
            resultsCallback=resultsCallback,
        )
        parameter_space.num_dimensions =  len(self.parameters)

        scenario_result = ConsistencyScenarioResult(
            scenario=self,
            consistent=consistent,
            parameter_space=parameter_space,
        )
        scenario_result._models = models

        return scenario_result


class ConsistencyScenarioResult(AnalysisScenarioResult, BaseModel):
    """
    ConsistencyScenarioResult result, which includes the consistency flag and
    search statistics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: ConsistencyScenario
    parameter_space: ParameterSpace
    consistent: Dict[Point, Dict[str, float]] = None
    _models: Dict[Point, pysmt_Model] = None

    def _parameters(self, point: Point):
        if point in self.consistent:
            parameters = self.scenario._smt_encoder.parameter_values(
                self.scenario.model, self.consistent[point]
            )
            return parameters
        else:
            raise Exception(
                f"Cannot get parameter values for an inconsistent scenario."
            )

    def dataframe(self, point: Point, interpolate="linear"):
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
            timeseries = self.scenario._smt_encoder.symbol_timeseries(
                self.scenario._encodings[point.values['step_size']], self._models[point]
            )
            df = pd.DataFrame.from_dict(timeseries)
            if interpolate:
                df = df.interpolate(method=interpolate)
            return df
        else:
            raise Exception(
                f"Cannot create dataframe for an inconsistent scenario."
            )

    def plot(self, point: Point, variables=None, **kwargs):
        """
        Plot the results in a matplotlib plot.

        Raises
        ------
        Exception
            failure if scenario is not consistent.
        """
        if self.consistent:
            if variables is not None:
                ax = self.dataframe(point)[variables].plot(
                    marker="o", **kwargs
                )
            else:
                ax = self.dataframe(point).plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(
                f"Cannot plot result for an inconsistent scenario."
            )
        return ax

    def __repr__(self) -> str:
        return str(self.consistent)
