from typing import List, Optional, Union

from pydantic import BaseModel

from funman.funman import FUNMANConfig
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel, PetrinetModel
from funman.model.query import QueryAnd, QueryFunction, QueryLE, QueryTrue
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation import Parameter
from funman.representation.representation import (
    LABEL_ANY,
    LABEL_TRUE,
    LabeledParameter,
    ModelParameter,
    ParameterSpace,
    Point,
    StructureParameter,
)
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)


class FunmanWorkRequest(BaseModel):
    query: Union[QueryAnd, QueryLE, QueryFunction, QueryTrue]
    parameters: Optional[List[LabeledParameter]] = None
    config: Optional[FUNMANConfig] = None
    structure_parameters: Optional[List[LabeledParameter]] = None


class FunmanWorkUnit(BaseModel):
    """
    Fields
    ------
    id : The UUID assigned to the request
    request : A copy of the request associated with this response
    """

    id: str
    model: Union[
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
    ]
    request: FunmanWorkRequest

    def to_scenario(
        self,
    ) -> Union[ConsistencyScenario, ParameterSynthesisScenario]:
        parameters = []
        for data in self.request.parameters:
            parameters.append(
                ModelParameter(
                    name=data.name, ub=data.ub, lb=data.lb, label=data.label
                )
            )
        for data in self.request.structure_parameters:
            parameters.append(
                StructureParameter(
                    name=data.name, ub=data.ub, lb=data.lb, label=data.label
                )
            )

        if self.request.parameters is None or all(
            p.label == LABEL_ANY for p in self.request.parameters
        ):
            return ConsistencyScenario(
                model=self.model,
                query=self.request.query,
                parameters=parameters,
            )

        if isinstance(self.model, EnsembleModel):
            raise Exception(
                "TODO handle EnsembleModel for ParameterSynthesisScenario"
            )

        return ParameterSynthesisScenario(
            model=self.model, query=self.request.query, parameters=parameters
        )


class FunmanResults(BaseModel):
    id: str
    model: Union[
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    request: FunmanWorkRequest
    done: bool
    parameter_space: ParameterSpace

    def finalize_result(
        self,
        result: Union[
            ConsistencyScenarioResult, ParameterSynthesisScenarioResult
        ],
    ):
        ps = None
        if isinstance(result, ConsistencyScenarioResult):
            ps = result.parameter_space
        if isinstance(result, ParameterSynthesisScenarioResult):
            ps = result.parameter_space

        if ps is None:
            raise Exception("No ParameterSpace for result")

        self.parameter_space = ps
        self.done = True
