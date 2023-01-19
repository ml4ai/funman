from abc import ABC, abstractclassmethod

from pydantic import BaseModel

from funman.model.model import Parameter


class Config(BaseModel):
    """
    Base definition of a configuration object
    """


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    @abstractclassmethod
    def solve(self, config: Config):
        pass

    @abstractclassmethod
    def _encode(self, config: Config):
        pass


class AnalysisScenarioResult(ABC, BaseModel):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractclassmethod
    def plot(self, **kwargs):
        pass
