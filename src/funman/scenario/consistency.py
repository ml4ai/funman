"""
This submodule defines a consistency scenario.  Consistency scenarios specify an existentially quantified model.  If consistent, the solution assigns any unassigned variable, subject to their bounds and other constraints.  
"""

from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
from pysmt.solvers.solver import Model as pysmt_Model

from funman.model.bilayer import BilayerModel, validator
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.query import QueryEncoded, QueryFunction, QueryLE, QueryTrue
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
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
        smart_union = True
        extra = "forbid"

    model: Union[DecapodeModel, BilayerModel, EncodedModel]
    query: Union[QueryLE, QueryEncoded, QueryFunction, QueryTrue]
    _smt_encoder: Encoder = None
    _model_encoding: Encoding = None
    _query_encoding: Encoding = None

    def solve(self, config: "FUNMANConfig") -> "AnalysisScenarioResult":
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

        if config._search is None:
            from funman.search.smt_check import SMTCheck

            search = SMTCheck()
        else:
            search = config._search()

        if self.model.structural_parameter_bounds:
            if self._smt_encoder is None:
                self._smt_encoder = self.model.default_encoder(config)
                self._smt_encoder._encode_timed_model_elements(self.model)

            # FIXME these ranges are also computed in the encoder
            num_steps_range = range(
                self.model.structural_parameter_bounds["num_steps"][0],
                self.model.structural_parameter_bounds["num_steps"][1] + 1,
            )
            step_size_range = range(
                self.model.structural_parameter_bounds["step_size"][0],
                self.model.structural_parameter_bounds["step_size"][1] + 1,
            )
            result = [
                [None for i in range(len(step_size_range))]
                for j in range(len(num_steps_range))
            ]

            consistent = None
            for configuration in self._smt_encoder._timed_model_elements[
                "configurations"
            ]:
                num_steps = configuration["num_steps"]
                step_size = configuration["step_size"]
                self._encode_timed(num_steps, step_size, config)
                result[num_steps - num_steps_range.start][
                    step_size - step_size_range.start
                ] = search.search(self, config=config)
                print(self._results_str(result))
                print("-" * 80)
                if result[num_steps - num_steps_range.start][
                    step_size - step_size_range.start
                ]:
                    consistent = result[num_steps - num_steps_range.start][
                        step_size - step_size_range.start
                    ].to_dict()
                    break
            # consistent = [
            #     [result[j][i].to_dict() for i in range(len(step_size_range))]
            #     for j in range(len(num_steps_range))
            # ]
            scenario_result = ConsistencyScenarioResult(
                scenario=self, consistent=consistent
            )
            scenario_result._model = result[num_steps - num_steps_range.start][
                step_size - step_size_range.start
            ]
        else:
            self._encode(config)
            result = search.search(self, config=config)
            # FIXME this to_dict call assumes result is an unusual type
            consistent = result.to_dict() if result else None

            scenario_result = ConsistencyScenarioResult(
                scenario=self, consistent=consistent
            )
            scenario_result._model = (
                result  # Constructor won't assign this private attr :(
            )
        return scenario_result

    def _results_str(self, result):
        return "\n".join(
            [
                x
                for x in [
                    (
                        str(i)
                        + ": ["
                        + "".join(
                            [
                                (
                                    "F"
                                    if s == False
                                    else ("T" if s == True else " ")
                                )
                                for s in t
                            ]
                        )
                        + "]"
                        if any([True for r in t if r is not None])
                        else ""
                    )
                    for i, t in enumerate(result)
                ]
                if x != ""
            ]
        )

    def _encode(self, config: "FUNMANConfig"):
        if self._smt_encoder is None:
            self._smt_encoder = self.model.default_encoder(config)
        self._model_encoding = self._smt_encoder.encode_model(self.model)
        self._query_encoding = self._smt_encoder.encode_query(
            self._model_encoding, self.query
        )
        return self._model_encoding, self._query_encoding

    def _encode_timed(self, num_steps, step_size, config: "FUNMANConfig"):
        # This will overwrite the _model_encoding for each configuration, but the encoder will retain components of the configurations.
        self._model_encoding = self._smt_encoder.encode_model_timed(
            self.model, num_steps, step_size
        )

        # This will create a new formula for each query without caching them (its typically inexpensive)
        self._query_encoding = self._smt_encoder.encode_query(
            self._model_encoding, self.query
        )
        return self._model_encoding, self._query_encoding


class ConsistencyScenarioResult(AnalysisScenarioResult, BaseModel):
    """
    ConsistencyScenarioResult result, which includes the consistency flag and
    search statistics.
    """

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    scenario: ConsistencyScenario
    consistent: Dict[str, float] = None
    _model: pysmt_Model = None

    def _parameters(self):
        if self.consistent:
            parameters = self.scenario._smt_encoder.parameter_values(
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
            timeseries = self.scenario._smt_encoder.symbol_timeseries(
                self.scenario._model_encoding, self._model
            )
            df = pd.DataFrame.from_dict(timeseries)
            if interpolate:
                df = df.interpolate(method=interpolate)
            return df
        else:
            raise Exception(
                f"Cannot create dataframe for an inconsistent scenario."
            )

    def plot(self, variables=None, **kwargs):
        """
        Plot the results in a matplotlib plot.

        Raises
        ------
        Exception
            failure if scenario is not consistent.
        """
        if self.consistent:
            if variables is not None:
                ax = self.dataframe()[variables].plot(marker="o", **kwargs)
            else:
                ax = self.dataframe().plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(
                f"Cannot plot result for an inconsistent scenario."
            )
        return ax

    def __repr__(self) -> str:
        return str(self.consistent)
