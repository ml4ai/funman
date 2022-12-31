from abc import ABC, abstractclassmethod


class Config(object):
    """
    Base definition of a configuration object
    """


class AnalysisScenario(object):
    """
    Abstract class for Analysis Scenarios.
    """

    def __init__(self):
        self.parameters = []

    @abstractclassmethod
    def solve(self, config: Config):
        pass

    @abstractclassmethod
    def _encode(self, config: Config):
        pass


class AnalysisScenarioResult(ABC):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractclassmethod
    def plot(self, **kwargs):
        pass
