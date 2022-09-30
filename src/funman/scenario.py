from abc import abstractclassmethod
from typing import Any, List
from funman.config import Config
from funman.parameter_space import ParameterSpace
from funman.search import SearchConfig
from funman.model import Parameter

class AnalysisScenario(object):
    """
    Abstract class for Analysis Scenarios.
    """
    @abstractclassmethod
    def simulate():
        pass

    @abstractclassmethod
    def solve(self, config: Config):
        pass

class ParameterSynthesisScenario(AnalysisScenario):
    """
    Parameter synthesis problem description that identifies the parameters to synthesize for a particular model.  The general problem is to identify multi-dimensional (one dimension per parameter) regions where either all points in the region are valid (true) parameters for the model or invalid (false) parameters.
    """

    def __init__(self, parameters: List[Parameter], model, search) -> None:
        super().__init__()
        self.parameters = parameters
        self.model = model
        self.search = search

    def solve(self, config: SearchConfig) -> ParameterSpace:
        """
        Synthesize parameters for a model.  Use the BoxSearch algorithm to decompose the parameter space and identify the feasible and infeasible parameter values.

        Parameters
        ----------
        config : SearchConfig
            Options for the Search algorithm.

        Returns
        -------
        ParameterSpace
            The parameter space.
        """
        result = self.search.search(self, config=config)

        result.plot(
            title=f"Residual = {result.statistics.current_residual}, |boxes|={result.statistics.num_unknown}",
        )

        parameter_space = ParameterSpace(result)

        return parameter_space


class AnalysisScenarioResult(object):
    """
    Abstract class for AnalysisScenario result data.
    """
    pass

class ParameterSynthesisScenarioResult(AnalysisScenarioResult):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result


