"""
This module defines the Parameter Synthesis scenario.
"""
from typing import Dict, List, Union

from pandas import DataFrame
from pydantic import BaseModel
from pysmt.formula import FNode
from pysmt.shortcuts import BOOL, Iff, Symbol

from funman.model import QueryTrue
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.query import QueryEncoded, QueryFunction, QueryLE
from funman.representation import Parameter
from funman.representation.representation import ParameterSpace, Point
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

    class Config:
        underscore_attrs_are_private = True
        smart_union = True
        extra = "forbid"

    parameters: List[Parameter]
    model: Union[DecapodeModel, BilayerModel, EncodedModel]
    query: Union[QueryLE, QueryEncoded, QueryFunction, QueryTrue] = None
    _search: str = "BoxSearch"
    _smt_encoder: Encoder = None  # TODO set to model.default_encoder()
    _model_encoding: Encoding = None
    _query_encoding: Encoding = None
    _assume_model: FNode = None
    _assume_query: FNode = None
    _original_parameter_widths: Dict[str, float] = {}

    def solve(
        self, config: "FUNMANConfig"
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
            result = []

            consistent = None
            for configuration in self._smt_encoder._timed_model_elements[
                "configurations"
            ]:
                num_steps = configuration["num_steps"]
                step_size = configuration["step_size"]
                self._encode_timed(num_steps, step_size, config)
                r = search.search(self, config=config)
                result.append(
                    {
                        "num_steps": num_steps,
                        "step_size": step_size,
                        "parameter_space": r,
                    }
                )
                print(self._results_str(result))
                print("-" * 80)

            parameter_space = ParameterSpace._from_configurations(result)
        else:
            self._encode(config)

            self._original_parameter_widths = {
                p: minus(p.ub, p.lb) for p in self.parameters
            }
            parameter_space: ParameterSpace = search.search(self, config)

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

    def _encode_timed(self, num_steps, step_size, config: "FUNMANConfig"):
        self._assume_model = Symbol("assume_model")
        self._assume_query = Symbol("assume_query")
        # This will overwrite the _model_encoding for each configuration, but the encoder will retain components of the configurations.
        self._model_encoding = self._smt_encoder.encode_model_timed(
            self.model, num_steps, step_size
        )
        self._model_encoding._formula = Iff(
            self._assume_model, self._model_encoding._formula
        )

        # This will create a new formula for each query without caching them (its typically inexpensive)
        self._query_encoding = self._smt_encoder.encode_query(
            self._model_encoding, self.query
        )
        self._query_encoding._formula = Iff(
            self._assume_query, self._query_encoding._formula
        )
        return self._model_encoding, self._query_encoding


class ParameterSynthesisScenarioResult(AnalysisScenarioResult, BaseModel):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    class Config:
        arbitrary_types_allowed = True

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
