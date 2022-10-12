from abc import abstractclassmethod
from typing import Any, Dict, List, Union
from funman.config import Config
from funman.examples.chime import CHIME
from funman.parameter_space import ParameterSpace
from funman.search import BoxSearch, SearchConfig
from funman.model import Model, Parameter

from pysmt.fnode import FNode
from pysmt.shortcuts import get_free_variables, And


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

    def __init__(
        self,
        parameters: List[Parameter],
        model: Union[str, FNode],
        search = None,
        config: Dict = None,
    ) -> None:
        super().__init__()
        self.parameters = parameters

        if search is None:
            search = BoxSearch()
        self.search = search

        self.search.real_time_plotting = config.get("real_time_plotting", True)
        self.search.write_cache_parameter_space = config.get("write_cache_parameter_space", None)
        self.search.read_cache_parameter_space = config.get("read_cache_parameter_space", None)

        if isinstance(model, str):
            self.chime = CHIME()
            epochs = config["epochs"]
            population_size = config["population_size"]
            infectious_days = config["infectious_days"]
            vars, model = self.chime.make_model(
                epochs=epochs,
                population_size=population_size,
                infectious_days=infectious_days,
            )
            self.vars = vars
        else:
            self.vars = model.get_free_variables()

        self.model = model

        # Associate parameters with symbols in the model
        symbol_map = {
            s.symbol_name(): s for p in self.model[0] for s in get_free_variables(p)
        }
        for p in self.parameters:
            if not p.symbol:
                p.symbol = symbol_map[p.name]

        param_symbols = set({p.name for p in self.parameters})
        assigned_parameters = [
            p
            for p in self.model[0]
            if len(
                set({q.symbol_name() for q in get_free_variables(p)}).intersection(
                    param_symbols
                )
            )
            == 0
        ]
        self.model = Model(
            And(
                And(assigned_parameters),
                self.model[1],
                (
                    And([And(layer) for step in self.model[2] for layer in step])
                    if isinstance(self.model[2], list)
                    else self.model[2]
                ),
                (
                    And(self.model[3])
                    if isinstance(self.model[3], list)
                    else self.model[3]
                ),
            )
        )

    def solve(self, config: SearchConfig = None) -> "ParameterSynthesisScenarioResult":
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
        if config is None:
            config = SearchConfig()

        result = self.search.search(self, config=config)
        parameter_space = ParameterSpace(result.true_boxes, result.false_boxes)

        return ParameterSynthesisScenarioResult(parameter_space)


class AnalysisScenarioResult(object):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractclassmethod
    def plot():
        pass

    pass


class ParameterSynthesisScenarioResult(AnalysisScenarioResult):
    """
    ParameterSynthesisScenario result, which includes the parameter space and
    search statistics.
    """

    def __init__(self, result: Any) -> None:
        super().__init__()
        self.parameter_space = result

    def plot():
        return "foo"
