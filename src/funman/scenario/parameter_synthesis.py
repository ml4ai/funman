"""
This submodule defined the Parameter Synthesis scenario.
"""
from . import AnalysisScenario, AnalysisScenarioResult
from funman.examples.chime import CHIME
from funman.model import Model, Parameter, Query
from funman.parameter_space import ParameterSpace
from funman.search import BoxSearch, SearchConfig
from pysmt.fnode import FNode
from typing import Any, Dict, List, Union


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
        smt_encoder=None,
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
        parameter_space = ParameterSpace(result.true_boxes, result.false_boxes)

        return ParameterSynthesisScenarioResult(parameter_space)

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

    def __init__(self, result: Any) -> None:
        super().__init__()
        self.parameter_space = result
