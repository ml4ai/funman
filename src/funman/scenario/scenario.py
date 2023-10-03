import threading
from abc import ABC, abstractclassmethod, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from funman import (
    NEG_INFINITY,
    POS_INFINITY,
    Assumption,
    BilayerModel,
    Box,
    DecapodeModel,
    EncodedModel,
    GeneratedPetriNetModel,
    GeneratedRegnetModel,
    Interval,
    ModelConstraint,
    ModelParameter,
    Parameter,
    ParameterConstraint,
    PetrinetModel,
    QueryAnd,
    QueryConstraint,
    QueryEncoded,
    QueryFunction,
    QueryGE,
    QueryLE,
    QueryTrue,
    StateVariableConstraint,
    StructureParameter,
)
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.regnet import GeneratedRegnetModel, RegnetModel


class AnalysisScenario(ABC, BaseModel):
    """
    Abstract class for Analysis Scenarios.
    """

    parameters: List[Parameter]
    normalization_constant: Optional[float] = None
    constraints: Optional[
        List[
            Union[
                "ModelConstraint",
                "ParameterConstraint",
                "StateVariableConstraint",
                "QueryConstraint",
            ]
        ]
    ] = None
    model_config = ConfigDict(extra="forbid")

    model: Union[
        GeneratedPetriNetModel,
        GeneratedRegnetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
        EnsembleModel
    ]
    query: Union[
        QueryAnd, QueryGE, QueryLE, QueryEncoded, QueryFunction, QueryTrue
    ] = QueryTrue()
    _assumptions: List[Assumption] = []
    _smt_encoder: Optional["Encoder"] = None
    # Encoding for different step sizes (key)
    _encodings: Optional[Dict[int, "Encoding"]] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create default constraints
        if self.constraints is None:
            self.constraints = [
                ModelConstraint(name="model_dynamics", model=self.model)
            ]

        if not any(
            c for c in self.constraints if isinstance(c, ModelConstraint)
        ):
            self.constraints.append(
                ModelConstraint(name="model_dynamics", model=self.model)
            )

        # create assumptions for each constraint that may be assumed.
        if self.constraints is not None:
            for constraint in self.constraints:
                if constraint.assumable():
                    self._assumptions.append(Assumption(constraint=constraint))

    @abstractclassmethod
    def get_kind(cls) -> str:
        pass

    @abstractmethod
    def solve(
        self, config: "FUNMANConfig", haltEvent: Optional[threading.Event]
    ):
        pass

    @abstractmethod
    def get_search(config: "FUNMANConfig") -> "Search":
        pass

    def initialize(self, config: "FUNMANConfig") -> "Search":
        search = self.get_search(config)
        self._process_parameters()

        self.constraints += [
            ParameterConstraint(name=parameter.name, parameter=parameter)
            for parameter in self.parameters
        ]

        self._set_normalization(config)
        self._initialize_encodings(config)
        return search

    def _initialize_encodings(self, config: "FUNMANConfig"):
        # self._assume_model = Symbol("assume_model")
        self._smt_encoder = self.model.default_encoder(config, self)
        assert self._smt_encoder._timed_model_elements

        times = list(
            set(
                [
                    t
                    for s in self._smt_encoder._timed_model_elements[
                        "state_timepoints"
                    ]
                    for t in s
                ]
            )
        )
        times.sort()

        # Initialize Assumptions
        # Maintain backward support for query as a single constraint
        if self.query is not None and not isinstance(self.query, QueryTrue):
            query_constraint = QueryConstraint(name="query", query=self.query)
            self.constraints += [query_constraint]
            self._assumptions.append(Assumption(constraint=query_constraint))

        # self._assume_query = [Symbol(f"assume_query_{t}") for t in times]
        for step_size_idx, step_size in enumerate(
            self._smt_encoder._timed_model_elements["step_sizes"]
        ):
            num_steps = max(
                self._smt_encoder._timed_model_elements["state_timepoints"][
                    step_size_idx
                ]
            )
            (
                # model_encoding,
                # query_encoding,
                encoding
            ) = self._smt_encoder.initialize_encodings(self, num_steps)
            # self._smt_encoder.encode_model_timed(
            #     self, num_steps, step_size
            # )

            self._encodings[step_size] = encoding

            # self._model_encoding[step_size] = model_encoding
            # self._query_encoding[step_size] = query_encoding

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

    def _process_parameters(self):
        if len(self.structure_parameters()) == 0:
            # either undeclared or wrong type
            # if wrong type, recover structure parameters
            self.parameters = [
                (
                    StructureParameter(name=p.name, lb=p.lb, ub=p.ub)
                    if (p.name == "num_steps" or p.name == "step_size")
                    else p
                )
                for p in self.parameters
            ]
            if len(self.structure_parameters()) == 0:
                # Add the structure parameters if still missing
                self.parameters += [
                    StructureParameter(name="num_steps", lb=0, ub=0),
                    StructureParameter(name="step_size", lb=1, ub=1),
                ]

        self._extract_non_overriden_parameters()
        self._filter_parameters()

    def _extract_non_overriden_parameters(self):
        from funman.constants import LABEL_ANY

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

    def _set_normalization(self, config):
        if config.normalization_constant is not None:
            self.normalization_constant = config.normalization_constant
        else:
            self.normalization_constant = (
                self.model.calculate_normalization_constant(self, config)
            )


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
