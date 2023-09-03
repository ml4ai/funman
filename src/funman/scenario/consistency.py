"""
This submodule defines a consistency scenario.  Consistency scenarios specify an existentially quantified model.  If consistent, the solution assigns any unassigned variable, subject to their bounds and other constraints.  
"""
import threading
from typing import Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ConfigDict, BaseModel
from pysmt.solvers.solver import Model as pysmt_Model

from funman.model.bilayer import BilayerModel, validator
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel, PetrinetModel
from funman.model.query import (
    QueryAnd,
    QueryEncoded,
    QueryFunction,
    QueryLE,
    QueryTrue,
)
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation.representation import (
    ParameterSpace,
    Point,
    StructureParameter,
)
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

    model_config = ConfigDict(extra="forbid")

    model: Union[
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
        RegnetModel,
        EnsembleModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    query: Union[QueryAnd, QueryLE, QueryEncoded, QueryFunction, QueryTrue]
    _smt_encoder: Optional[Encoder] = None
    _model_encoding: Optional[Encoding] = None
    _query_encoding: Optional[Encoding] = None
    _box: Optional[Encoding] = None

    @classmethod
    def get_kind(cls) -> str:
        return "consistency"

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
        if config._search is None:
            from funman.search.smt_check import SMTCheck

            search = SMTCheck()
        else:
            search = config._search()

        if len(self.structure_parameters()) == 0:
            # either undeclared or wrong type
            # if wrong type, recover structure parameters
            self.parameters = [
                (
                    StructureParameter(name=p.name, lb=p.lb, ub=p.ub)
                    if (p.name == "num_steps" or p.name == "step_size")
                    else p
                )
                for p in self.parameters
            ]
            if len(self.structure_parameters()) == 0:
                # Add the structure parameters if still missing
                self.parameters += [
                    StructureParameter(name="num_steps", lb=0, ub=0),
                    StructureParameter(name="step_size", lb=1, ub=1),
                ]

        self._extract_non_overriden_parameters()
        self._filter_parameters()
        num_parameters = len(self.parameters)

        if config.normalization_constant is not None:
            self.normalization_constant = config.normalization_constant
        else:
            self.normalization_constant = (
                self.model.calculate_normalization_constant(self, config)
            )

        if self._smt_encoder is None:
            self._smt_encoder = self.model.default_encoder(config, self)

        parameter_space, models, consistent = search.search(
            self,
            config=config,
            haltEvent=haltEvent,
            resultsCallback=resultsCallback,
        )
        parameter_space.num_dimensions = num_parameters

        scenario_result = ConsistencyScenarioResult(
            scenario=self,
            consistent=consistent,
            parameter_space=parameter_space,
        )
        scenario_result._models = models

        return scenario_result

    def _results_str(self, starting_steps, result):
        return "\n".join(
            [
                x
                for x in [
                    (
                        str(i + starting_steps)
                        + ": ["
                        + "".join(
                            [
                                (
                                    "F"
                                    if s is None
                                    else ("T" if (s is not None and s) else " ")
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

    def _encode_timed(self, num_steps, step_size_idx, config: "FUNMANConfig"):
        # # This will overwrite the _model_encoding for each configuration, but the encoder will retain components of the configurations.
        # self._model_encoding = self._smt_encoder.encode_model_timed(
        #     self, num_steps, step_size
        # )

        # # This will create a new formula for each query without caching them (its typically inexpensive)
        # self._query_encoding = self._smt_encoder.encode_query(
        #     self.query, num_steps, step_size
        # )

        (
            model_encoding,
            query_encoding,
        ) = self._smt_encoder.initialize_encodings(
            self, num_steps, step_size_idx
        )

        self._model_encoding = model_encoding
        self._query_encoding = query_encoding
        return self._model_encoding, self._query_encoding


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
                self.scenario._model_encoding, self._models[point]
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
                ax = self.dataframe(point)[variables].plot(marker="o", **kwargs)
            else:
                ax = self.dataframe(point).plot(marker="o", **kwargs)
            plt.show(block=False)
        else:
            raise Exception(f"Cannot plot result for an inconsistent scenario.")
        return ax

    def __repr__(self) -> str:
        return str(self.consistent)
