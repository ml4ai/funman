"""
This subpackage contains definitions of scenarios that FUNMAN
is capable of solving.
"""
from abc import abstractclassmethod
from funman.config import Config


class AnalysisScenario(object):
    """
    Abstract class for Analysis Scenarios.
    """

    def __init__(self):
        self.parameters = []

    @abstractclassmethod
    def simulate():
        pass

    @abstractclassmethod
    def solve(self, config: Config):
        pass


class AnalysisScenarioResult(object):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractclassmethod
    def plot(self):
        pass
