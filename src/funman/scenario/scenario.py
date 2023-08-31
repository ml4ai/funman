import threading
from abc import ABC, abstractclassmethod, abstractmethod
from decimal import Decimal
from typing import List, Optional, Union

from pydantic import BaseModel

from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.encoded import EncodedModel
from funman.representation.representation import (
    Box,
    Interval,
    ModelParameter,
    ParameterSpace,
    StructureParameter,
)


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Union[ModelParameter, StructureParameter]]
    normalization_constant: Optional[float] = None

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
        return len(self.parameters)

    def search_space_volume(self) -> Decimal:
        bounds = {}
        for param in self.parameters:
            bounds[param.name] = Interval(lb=param.lb, ub=param.ub)
        return Box(bounds=bounds).volume()

    def representable_space_volume(self) -> Decimal:
        bounds = {}
        for param in self.parameters:
            bounds[param.name] = Interval(lb=NEG_INFINITY, ub=POS_INFINITY)
        return Box(bounds=bounds).volume()

    def structure_parameters(self):
        return [
            p for p in self.parameters if isinstance(p, StructureParameter)
        ]

    def model_parameters(self):
        return [p for p in self.parameters if isinstance(p, ModelParameter)]

    def synthesized_parameters(self):
        return [p for p in self.parameters if p.is_synthesized()]

    def structure_parameter(self, name: str) -> StructureParameter:
        return next(p for p in self.parameters if p.name == name)

    def _extract_non_overriden_parameters(self):
        from funman.server.query import LABEL_ANY

        # If a model has parameters that are not overridden by the scenario, then add them to the scenario
        model_parameters = self.model._parameter_names()
        model_parameter_values = self.model._parameter_values()
        model_parameters = [] if model_parameters is None else model_parameters
        non_overriden_parameters = []
        for p in [
            param
            for param in model_parameters
            if param
            not in [
                overridden_param.name for overridden_param in self.parameters
            ]
        ]:
            bounds = {}
            lb = self.model._parameter_lb(p)
            ub = self.model._parameter_ub(p)
            if ub is not None and lb is not None:
                bounds["ub"] = ub
                bounds["lb"] = lb
            elif model_parameter_values[p]:
                value = model_parameter_values[p]
                bounds["lb"] = bounds["ub"] = value
            else:
                bounds = {}
            non_overriden_parameters.append(
                ModelParameter(name=p, **bounds, label=LABEL_ANY)
            )

        self.parameters += non_overriden_parameters

    def _filter_parameters(self):
        # If the scenario has parameters that are not in the model, then remove them from the scenario
        model_parameters = self.model._parameter_names()

        if model_parameters is not None:
            filtered_parameters = [
                p
                for p in self.parameters
                if p.name in model_parameters
                or isinstance(p, StructureParameter)
            ]
            self.parameters = filtered_parameters


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
