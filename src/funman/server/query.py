from typing import List, Literal, Optional, Union

from pydantic import BaseModel

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


class QueryRequest(BaseModel):
    model: Union[
        RegnetModel,
        EnsembleModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    query: Union[QueryAnd, QueryLE, QueryEncoded, QueryFunction, QueryTrue]
    parameters: Optional[List[LabeledParameter]] = None

    def to_scenario(
        self,
    ) -> Union[ConsistencyScenario, ParameterSynthesisScenario]:
        if self.parameters is None or all(
            p.label == LABEL_ANY for p in self.parameters
        ):
            return ConsistencyScenario(model=self.model, query=self.query)

        if isinstance(self.model, EnsembleModel):
            raise Exception(
                "TODO handle EnsembleModel for ParameterSynthesisScenario"
            )

        # resolve all 'any' parameters to a Parameter object
        parameters = []
        for data in self.parameters:
            # skip any parameters that are not of kind 'any'
            if data.label != LABEL_ALL:
                continue
            parameters.append(
                Parameter(name=data.name, ub=data.ub, lb=data.lb)
            )
        return ParameterSynthesisScenario(
            model=self.model, query=self.query, parameters=parameters
        )


class QueryResponse(BaseModel):
    id: str
    request: QueryRequest
    parameter_space: ParameterSpace

    @staticmethod
    def from_result(
        id: str,
        request: QueryRequest,
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

        return QueryResponse(id=id, request=request, parameter_space=ps)
