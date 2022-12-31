"""
This module wraps a simulator invocation as a Scenario.
"""
from funman.model.model import Model, Query, QueryFunction
from funman.scenario import AnalysisScenario, AnalysisScenarioResult, Config


class SimulationScenario(AnalysisScenario):
    def __init__(self, model: Model, query: Query):
        super().__init__()
        self.model = model
        self.query = query

    def solve(self, config: Config):
        results = self.model.main_fn()
        query_satisfied = self._evaluate_query(results)
        return SimulationScenarioResult(self, results, query_satisfied)

    def _evaluate_query(self, results):
        if isinstance(self.query, QueryFunction):
            result = self.query.function(results)
        else:
            raise Exception(
                f"SimulationScenario cannot evaluate query of type {type(self.query)}"
            )
        return result


class SimulationScenarioResult(AnalysisScenarioResult):
    def __init__(
        self, scenario: SimulationScenario, results, query_satisfied: bool
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.results = results
        self.query_satisfied = query_satisfied

    def plot(self, **kwargs):
        pass
