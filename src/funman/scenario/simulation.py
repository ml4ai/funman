"""
This module wraps a simulator invocation as a Scenario.
"""
from importlib import import_module
from typing import Any

from pydantic import BaseModel

from funman.model.model import Model
from funman.model.query import Query, QueryFunction
from funman.model.simulator import SimulatorModel
from funman.scenario import AnalysisScenario, AnalysisScenarioResult


class SimulationScenario(AnalysisScenario, BaseModel):
    model: SimulatorModel
    query: Query

    @classmethod
    def get_kind(cls) -> str:
        return "simulation"

    def solve(self, config: "FUNMANConfig"):
        try:
            p, m = self.model.main_fn.rsplit(".", 1)
            mod = import_module(p)
            main_fn = getattr(mod, m)
        except NameError:
            # print("Not in scope!")
            pass

        results = main_fn()
        query_satisfied = self._evaluate_query(results)
        return SimulationScenarioResult(
            scenario=self, results=results, query_satisfied=query_satisfied
        )

    def _evaluate_query(self, results):
        if isinstance(self.query, QueryFunction):
            result = self.query.function(results)
        elif isinstance(self.query, Query):
            result = results is not None  # Return results without query
        else:
            raise Exception(
                f"SimulationScenario cannot evaluate query of type {type(self.query)}"
            )
        return result

    def _encode(self):
        pass


class SimulationScenarioResult(AnalysisScenarioResult, BaseModel):
    scenario: SimulationScenario
    results: Any = None
    query_satisfied: bool

    def plot(self, **kwargs):
        pass
