from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from funman.funman import FUNMANConfig
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import PetrinetModel
from funman.model.query import (
    QueryAnd,
    QueryEncoded,
    QueryFunction,
    QueryLE,
    QueryTrue,
)
from funman.model.regnet import RegnetModel
from funman.representation import Parameter
from funman.representation.representation import (
    LABEL_TRUE,
    ParameterSpace,
    Point,
)
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)

LABEL_ANY = "any"
LABEL_ALL = "all"


class LabeledParameter(Parameter):
    label: Literal["any", "all"] = LABEL_ANY


class FunmanWorkRequest(BaseModel):
    query: Union[QueryAnd, QueryLE, QueryEncoded, QueryFunction, QueryTrue]
    parameters: Optional[List[LabeledParameter]] = None
    config: Optional[FUNMANConfig] = None


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
        EncodedModel,
    ]
    request: FunmanWorkRequest

    def to_scenario(
        self,
    ) -> Union[ConsistencyScenario, ParameterSynthesisScenario]:
        if self.request.parameters is None or all(
            p.label == LABEL_ANY for p in self.request.parameters
        ):
            return ConsistencyScenario(
                model=self.model, query=self.request.query
            )

        if isinstance(self.model, EnsembleModel):
            raise Exception(
                "TODO handle EnsembleModel for ParameterSynthesisScenario"
            )

        # resolve all 'any' parameters to a Parameter object
        parameters = []
        for data in self.request.parameters:
            # skip any parameters that are not of kind 'any'
            if data.label != LABEL_ALL:
                continue
            parameters.append(
                Parameter(name=data.name, ub=data.ub, lb=data.lb)
            )
        return ParameterSynthesisScenario(
            model=self.model, query=self.request.query, parameters=parameters
        )


class FunmanResults(BaseModel):
    id: str
    model: Union[
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    request: FunmanWorkRequest
    done: bool
    parameter_space: ParameterSpace

    @staticmethod
    def from_result(
        work: FunmanWorkUnit,
        result: Union[
            ConsistencyScenarioResult, ParameterSynthesisScenarioResult
        ],
    ):
        ps = None
        if isinstance(result, ConsistencyScenarioResult):
            ps = ParameterSpace()
            if result.consistent is not None:
                point = Point(values=result.consistent, label=LABEL_TRUE)
                ps.true_points.append(point)
        if isinstance(result, ParameterSynthesisScenarioResult):
            ps = result.parameter_space

        if ps is None:
            raise Exception("No ParameterSpace for result")

        return FunmanResults(
            id=work.id,
            model=work.model,
            request=work.request,
            done=False,
            parameter_space=ps,
        )
