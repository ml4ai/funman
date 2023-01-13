from abc import ABC, abstractclassmethod
from typing import List

from pydantic import BaseModel

from funman.model.model import Parameter


class Config(BaseModel):
    """
    Base definition of a configuration object
    """


class AnalysisScenario(BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Parameter]

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
