"""
This module defines the Parameter Synthesis scenario.
"""
import threading
from functools import partial
from typing import Callable, Dict, List, Optional, Union

from pandas import DataFrame
from pydantic import ConfigDict, BaseModel
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
from funman.representation import ModelParameter, StructureParameter
from funman.representation.representation import (
    Box,
    ModelParameter,
    ParameterSpace,
    Point,
    StructureParameter,
)
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    ConsistencyScenario,
)
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
    _model_encoding: Optional[Encoding] = None
    _query_encoding: Optional[Encoding] = None
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

        if config.normalize:
            if isinstance(config.normalize, float):
                self.normalization_constant = config.normalize
            else: # is bool True
                self.normalization_constant = self.model.calculate_normalization_constant(self.parameters)
         

        num_parameters = len(self.parameters)
        if self._smt_encoder is None:
            self._smt_encoder = self.model.default_encoder(config, self)

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

    def _results_str(self, result: List[Dict]):
        return "\n".join(
            ["num_steps\tstep_size\t|true|\t|false|"]
            + [
                f"{r['num_steps']}\t\t{r['step_size']}\t\t{len(r['parameter_space'].true_boxes)}\t{len(r['parameter_space'].false_boxes)}"
                for r in result
            ]
        )

    def _encode(self, config: "FUNMANConfig"):
        """
        The encoding uses assumption symbols for the model and query so that it is possible to push/pop the (possibly negated) symbols to reason about cases where the model or query must be true or false.

        Returns
        -------
        _type_
            _description_
        """
        if self._smt_encoder is None:
            self._smt_encoder = self.model.default_encoder(config)
        self._assume_model = Symbol("assume_model")
        self._assume_query = Symbol("assume_query")
        self._model_encoding = self._smt_encoder.encode_model(self.model)
        self._model_encoding._formula = Iff(
            self._assume_model, self._model_encoding._formula
        )
        self._query_encoding = self._smt_encoder.encode_query(
            self._model_encoding, self.query
        )
        self._query_encoding._formula = Iff(
            self._assume_query, self._query_encoding._formula
        )
        return self._model_encoding, self._query_encoding

    def _encode_timed(self, num_steps, step_size_idx, config: "FUNMANConfig"):
        # self._assume_model = Symbol("assume_model")
        if self._smt_encoder._timed_model_elements:
            step_size = self._smt_encoder._timed_model_elements["step_sizes"][
                step_size_idx
            ]
            self._assume_query = [
                Symbol(f"assume_query_{t}")
                for t in range(0, (num_steps * step_size) + 1, step_size)
            ]
        # This will overwrite the _model_encoding for each configuration, but the encoder will retain components of the configurations.
        (
            model_encoding,
            query_encoding,
        ) = self._smt_encoder.initialize_encodings(
            self, num_steps, step_size_idx
        )
        # self._smt_encoder.encode_model_timed(
        #     self, num_steps, step_size
        # )

        self._model_encoding = model_encoding
        self._query_encoding = query_encoding

        # This will create a new formula for each query without caching them (its typically inexpensive)
        # self._query_encoding = self._smt_encoder.initialize_encoding(self, num_steps, step_size)
        # self._smt_encoder.encode_query(
        #     self.query, num_steps, step_size
        # )
        # self._query_encoding.assume(self._assume_query)
        return self._model_encoding, self._query_encoding

    def encode_simplified(self, box: Box, timepoint: int):
        model_encoding = self._model_encoding.encoding(
            self._model_encoding._encoder.encode_model_layer,
            layers=list(range(timepoint + 1)),
            box=box,
        )
        query_encoding = self._query_encoding.encoding(
            partial(
                self._query_encoding._encoder.encode_query_layer,
                self.query,
            ),
            layers=[timepoint],
            box=box,
            assumptions=self._assume_query,
        )
        step_size_idx = self._smt_encoder._timed_model_elements[
            "step_sizes"
        ].index(self._model_encoding.step_size)

        return self._smt_encoder.encode_simplified(
            model_encoding, query_encoding, step_size_idx
        )


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
        dfs = []
        for point in points:
            if isinstance(point, dict):
                point = Point.parse_obj(point)
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
        return str(self.dict())
