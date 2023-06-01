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
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)


class LabeledParameter(Parameter):
    label: Literal["one", "any"] = "one"


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
            p.label == "one" for p in self.parameters
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
            if data.label != "any":
                continue
            parameters.append(
                Parameter(name=data.name, ub=data.ub, lb=data.lb)
            )
        return ParameterSynthesisScenario(
            model=self.model, query=self.query, parameters=parameters
        )


class QueryResponse(BaseModel):
    id: str
    kind: Literal["consistency", "parameter_synthesis"]
    result: Union[ConsistencyScenarioResult, ParameterSynthesisScenarioResult]
