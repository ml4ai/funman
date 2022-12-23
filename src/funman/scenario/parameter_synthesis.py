"""
This submodule defined the Parameter Synthesis scenario.
"""
from typing import Dict, List, Union

from funman.model import Model, Parameter, Query
from funman.model2smtlib.translate import Encoder
from funman.parameter_space import ParameterSpace
from funman.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.scenario.consistency import ConsistencyScenario
from funman.search import BoxSearch, SMTCheck
from funman.search_episode import SearchEpisode
from funman.utils.search_utils import Point, SearchConfig


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
        query: Query,
        search=None,
        smt_encoder=Encoder(),
        config: Dict = None,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.smt_encoder = smt_encoder
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

        self.encode()
        result = self.search.search(self, config=config)
        return ParameterSynthesisScenarioResult(result, self)

    def encode(self):
        self.model_encoding = self.smt_encoder.encode_model(self.model)
        self.query_encoding = self.smt_encoder.encode_query(
            self.model_encoding, self.query
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

    # points are of the form (see Point.to_dict())
    # [
    #     {"values": {"beta": 0.1}}
    # ]
    # Or
    # List[Point]
    def true_point_timeseries(
        self, points: Union[List[Point], List[dict]] = None
    ):
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
