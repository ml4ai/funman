"""
This module wraps a simulator invocation as a Scenario.
"""
from typing import Any

from pydantic import BaseModel

from funman.model.model import Model, Query, QueryFunction
from funman.model.simulator import SimulatorModel
from funman.scenario import AnalysisScenario, AnalysisScenarioResult


class SimulationScenario(AnalysisScenario, BaseModel):
    model: SimulatorModel
    query: Query

    def solve(self, config: "FUNMANConfig"):
        results = self.model.main_fn()
        query_satisfied = self._evaluate_query(results)
        return SimulationScenarioResult(
            scenario=self, results=results, query_satisfied=query_satisfied
        )

    def _evaluate_query(self, results):
        if isinstance(self.query, QueryFunction):
            result = self.query.function(results)
        else:
            raise Exception(
                f"SimulationScenario cannot evaluate query of type {type(self.query)}"
            )
        return result

    def _encode(self):
        pass


class SimulationScenarioResult(AnalysisScenarioResult, BaseModel):
    scenario: SimulationScenario
    results: Any
    query_satisfied: bool

    def plot(self, **kwargs):
        pass
