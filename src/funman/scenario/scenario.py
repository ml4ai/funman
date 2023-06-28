import threading
from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Optional
from funman.representation.representation import (
    ModelParameter,
    Parameter,
    ParameterSpace,
    StructureParameter,
)

from pydantic import BaseModel


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Parameter]

    @abstractclassmethod
    def get_kind(cls) -> str:
        pass

    @abstractmethod
    def solve(
        self, config: "FUNMANConfig", haltEvent: Optional[threading.Event]
    ):
        pass

    @abstractmethod
    def _encode(self, config: "FUNMANConfig"):
        pass

    def num_dimensions(self):
        """
        Return the number of parameters (dimensions) that are synthesized.  A parameter is synthesized if it has a domain with width greater than zero and it is either labeled as LABEL_ALL or is a structural parameter (which are LABEL_ALL by default).
        """
        return len([p for p in self.parameters])

    def structure_parameters(self):
        return [p for p in self.parameters if isinstance(p, StructureParameter)]

    def model_parameters(self):
        return [p for p in self.parameters if isinstance(p, ModelParameter)]

    def synthesized_parameters(self):
        return [p for p in self.parameters if p.is_synthesized()]

    def structure_parameter(self, name: str) -> StructureParameter:
        return next(p for p in self.parameters if p.name == name)


class AnalysisScenarioResult(ABC):
    """
    Abstract class for AnalysisScenario result data.
    """

    @abstractmethod
    def plot(self, **kwargs):
        pass


class AnalysisScenarioResultException(BaseModel, AnalysisScenarioResult):
    exception: str

    def plot(self, **kwargs):
        raise NotImplemented(
            "AnalysisScenarioResultException cannot be plotted with plot()"
        )
