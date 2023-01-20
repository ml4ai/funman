from abc import ABC, abstractclassmethod

import multiprocess as mp
from pydantic import BaseModel

from funman.utils.handlers import NoopResultHandler, ResultHandler


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    @abstractclassmethod
    def solve(self, config: "FUNMANConfig"):
        pass

    @abstractclassmethod
    def _encode(self, config: "FUNMANConfig"):
        pass


class AnalysisScenarioResult(ABC):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractclassmethod
    def plot(self, **kwargs):
        pass


class AnalysisScenarioResultException(BaseModel, AnalysisScenarioResult):
    exception: str

    def plot(self, **kwargs):
        raise NotImplemented(
            "AnalysisScenarioResultException cannot be plotted with plot()"
        )
