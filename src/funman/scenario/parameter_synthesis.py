"""
This module defines the Parameter Synthesis scenario.
"""
from typing import Any, Dict, List, Union

from pandas import DataFrame
from pysmt.shortcuts import Iff, Symbol

from funman.model import Model, Parameter, Query, QueryTrue
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    ConsistencyScenario,
)
from funman.search.box_search import BoxSearch
from funman.search.representation import ParameterSpace, Point
from funman.search.search import SearchConfig, SearchEpisode
from funman.search.smt_check import SMTCheck
from funman.translate import EncodedEncoder


class ParameterSynthesisScenario(AnalysisScenario):
    """
    Parameter synthesis problem description that identifies the parameters to
    synthesize for a particular model.  The general problem is to identify
    multi-dimensional (one dimension per parameter) regions where either all
    points in the region are valid (true) parameters for the model or invalid
    (false) parameters.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        model: Model,
        query: Query = QueryTrue(),
        search: BoxSearch = BoxSearch(),
        smt_encoder=None,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.smt_encoder = (
            smt_encoder if smt_encoder else model.default_encoder()
        )
        self.model_encoding = None
        self.query_encoding = None

        if search is None:
            search = BoxSearch()
        self.search = search
        self.model = model
        self.query = query

    def solve(
        self, config: SearchConfig = None
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
        if config is None:
            config = SearchConfig()

        self._encode()
        result = self.search.search(self, config=config)
        return ParameterSynthesisScenarioResult(result, self)

    def _encode(self):
        """
        The encoding uses assumption symbols for the model and query so that it is possible to push/pop the (possibly negated) symbols to reason about cases where the model or query must be true or false.

        Returns
        -------
        _type_
            _description_
        """
        self.assume_model = Symbol("assume_model")
        self.assume_query = Symbol("assume_query")
        self.model_encoding = self.smt_encoder.encode_model(self.model)
        self.model_encoding.formula = Iff(
            self.assume_model, self.model_encoding.formula
        )
        self.query_encoding = self.smt_encoder.encode_query(
            self.model_encoding, self.query
        )
        self.query_encoding.formula = Iff(
            self.assume_query, self.query_encoding.formula
        )
        return self.model_encoding, self.query_encoding


class ParameterSynthesisScenarioResult(AnalysisScenarioResult):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    def __init__(
        self, episode: SearchEpisode, scenario: ParameterSynthesisScenario
    ) -> None:
        super().__init__()
        self.episode = episode
        self.scenario = scenario
        self.parameter_space = ParameterSpace(
            episode.true_boxes,
            episode.false_boxes,
            episode.true_points,
            episode.false_points,
        )

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

    # points are of the form (see Point.to_dict())
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
                point = Point.from_dict(point)
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
                smt_encoder=self.scenario.smt_encoder,
            )
            result = scenario.solve(
                config=SearchConfig(solver="dreal", search=SMTCheck)
            )
            assert result
            # plot the results
            # result.plot(logy=True)
            # print(result.dataframe())
            dfs.append(result.dataframe())
        return dfs

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_space": self.parameter_space,
        }
