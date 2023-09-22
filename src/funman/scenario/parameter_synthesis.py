"""
This module defines the Parameter Synthesis scenario.
"""
import threading
from functools import partial
from typing import Callable, Dict, List, Optional, Union

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict
from pysmt.formula import FNode
from pysmt.shortcuts import BOOL, Iff, Symbol

from funman.model import (
    BilayerModel,
    DecapodeModel,
    EncodedModel,
    GeneratedPetriNetModel,
    GeneratedRegnetModel,
    PetrinetModel,
    QueryTrue,
)
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.query import (
    QueryAnd,
    QueryEncoded,
    QueryFunction,
    QueryGE,
    QueryLE,
)
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation.representation import (
    Box,
    ParameterSpace,
    Point,
)
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    ConsistencyScenario,
)
from funman.representation.assumption import Assumption
from funman.translate.translate import Encoder, Encoding
from funman.utils.math_utils import minus


class ParameterSynthesisScenario(AnalysisScenario, BaseModel):
    """
    Parameter synthesis problem description that identifies the parameters to
    synthesize for a particular model.  The general problem is to identify
    multi-dimensional (one dimension per parameter) regions where either all
    points in the region are valid (true) parameters for the model or invalid
    (false) parameters.
    """

    model_config = ConfigDict(extra="forbid")

    model: Union[
        GeneratedPetriNetModel,
        GeneratedRegnetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    query: Union[
        QueryAnd, QueryGE, QueryLE, QueryEncoded, QueryFunction, QueryTrue
    ] = QueryTrue()
    _search: str = "BoxSearch"
    _smt_encoder: Optional[
        Encoder
    ] = None  # TODO set to model.default_encoder()
    _model_encoding: Optional[Dict[int, Encoding]] = {}
    _query_encoding: Optional[Dict[int, Encoding]] = {}

    _assume_model: Optional[FNode] = None
    _assume_query: Optional[FNode] = None
    _original_parameter_widths: Dict[str, float] = {}

    @classmethod
    def get_kind(cls) -> str:
        return "parameter_synthesis"

    def solve(
        self,
        config: "FUNMANConfig",
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "ParameterSynthesisScenarioResult":
        """
        Synthesize parameters for a model.  Use the BoxSearch algorithm to
        decompose the parameter space and identify the feasible and infeasible
        parameter values.

        Parameters
        ----------
        config : SearchConfig
            Options for the Search algorithm.

        Returns
        -------
        ParameterSpace
            The parameter space.
        """

        if config._search is None:
            from funman.search.box_search import BoxSearch

            search = BoxSearch()
        else:
            search = config._search()

        self._process_parameters()

        self._set_normalization(config)

        num_parameters = len(self.parameters)
        self._initialize_encodings(config)

        self._original_parameter_widths = {
            p: minus(p.ub, p.lb) for p in self.parameters
        }

        parameter_space: ParameterSpace = search.search(
            self,
            config,
            haltEvent=haltEvent,
            resultsCallback=resultsCallback,
        )

        parameter_space.num_dimensions = num_parameters
        return ParameterSynthesisScenarioResult(
            parameter_space=parameter_space, scenario=self
        )

    def _initialize_encodings(self, config: "FUNMANConfig"):
        # self._assume_model = Symbol("assume_model")
        self._smt_encoder = self.model.default_encoder(config, self)
        assert self._smt_encoder._timed_model_elements

        times = list(
            set(
                [
                    t
                    for s in self._smt_encoder._timed_model_elements[
                        "state_timepoints"
                    ]
                    for t in s
                ]
            )
        )
        times.sort()

        # Initialize Assumptions
        # Maintain backward support for query as a single constraint
        self._assumptions.append(Assumption(constraint=self.query))

        #self._assume_query = [Symbol(f"assume_query_{t}") for t in times]
        for step_size_idx, step_size in enumerate(
            self._smt_encoder._timed_model_elements["step_sizes"]
        ):
            num_steps = max(
                self._smt_encoder._timed_model_elements["state_timepoints"][
                    step_size_idx
                ]
            )
            (
                model_encoding,
                query_encoding,
            ) = self._smt_encoder.initialize_encodings(
                self, num_steps, step_size_idx
            )
            # self._smt_encoder.encode_model_timed(
            #     self, num_steps, step_size
            # )

            self._model_encoding[step_size] = model_encoding
            self._query_encoding[step_size] = query_encoding



class ParameterSynthesisScenarioResult(AnalysisScenarioResult, BaseModel):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # episode: SearchEpisode
    scenario: ParameterSynthesisScenario
    parameter_space: ParameterSpace

    def plot(self, **kwargs):
        """
        Plot the results

        Raises
        ------
        NotImplementedError
            TODO
        """
        raise NotImplementedError(
            "ParameterSynthesisScenario.plot() is not implemented"
        )

    # points are of the form (see Point)
    # [
    #     {"values": {"beta": 0.1}}
    # ]
    # Or
    # List[Point]
    def true_point_timeseries(
        self, points: Union[List[Point], List[dict]] = List[DataFrame]
    ):
        """
        Get a timeseries for each of the points, assuming that the points are in the parameter space.

        Parameters
        ----------
        points : Union[List[Point], List[dict]], optional
            points to use to genrate a timeseries, by default None

        Returns
        -------
        List[Dataframe]
            a list of dataframes that list a timeseries of values for each model variable.

        Raises
        ------
        Exception
            malformed points
        """
        # for each true box
        from funman.config import FUNMANConfig

        dfs = []
        for point in points:
            if isinstance(point, dict):
                point = Point.model_validate(point)
            if not isinstance(point, Point):
                raise Exception("Provided point is not of type Point")
            # update the model with the
            for p, v in point.values.items():
                # assign that parameter to the value of the picked point
                self.scenario.model.parameter_bounds[p.name] = [v, v]

            # check the consistency
            scenario = ConsistencyScenario(
                self.scenario.model,
                self.scenario.query,
                _smt_encoder=self.scenario._smt_encoder,
            )
            result = scenario.solve(
                config=FUNMANConfig(solver="dreal", _search="SMTCheck")
            )
            assert result
            # plot the results
            # result.plot(logy=True)
            # print(result.dataframe())
            dfs.append(result.dataframe())
        return dfs

    def __repr__(self) -> str:
        return str(self.model_dump())
