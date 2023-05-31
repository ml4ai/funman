from typing import Dict, Literal, Optional, Union

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
from funman.scenario.consistency import ConsistencyScenarioResult
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenarioResult,
)


class ParameterDescription(BaseModel):
    kind: Literal["one", "any"] = "one"
    description: Optional[Parameter] = None


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
    parameters: Optional[Dict[str, ParameterDescription]] = None


class QueryResponse(BaseModel):
    id: str
    scenario: Literal["consistency", "parameter_synthesis"]
    result: Union[ConsistencyScenarioResult, ParameterSynthesisScenarioResult]
