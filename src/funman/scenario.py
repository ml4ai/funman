
from typing import Any


class AnalysisScenario(object):
    pass

class ParameterSynthesisScenario(AnalysisScenario):
    def __init__(self, parameters, model) -> None:
        super().__init__()
        self.parameters = parameters
        self.model = model
    
class AnalysisScenarioResult(object):
    def __init__(self, scenario: AnalysisScenario) -> None:
        self.scenario = scenario

class ParameterSynthesisScenarioResult(AnalysisScenarioResult):
    def __init__(self, scenario: ParameterSynthesisScenario, result: Any) -> None:
        super().__init__(scenario)
        self.result = result