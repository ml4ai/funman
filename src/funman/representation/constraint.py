from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from funman.model import Model
from funman.model.query import Query

from .interval import Interval
from .parameter import Parameter, StructureParameter


class Constraint(BaseModel):
    _assumable: bool = True
    name: str

    model_config = ConfigDict(extra="forbid")

    def assumable(self) -> bool:
        return self._assumable

    def __hash__(self) -> int:
        return 1

    def encodable(self) -> bool:
        return True

    def relevant_at_time(self, time: int) -> bool:
        return True


class ModelConstraint(Constraint):
    _assumable: bool = False
    model: Model

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 2


class ParameterConstraint(Constraint):
    _assumable: bool = False
    parameter: Parameter

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 1

    def encodable(self) -> bool:
        return not isinstance(self.parameter, StructureParameter)

    def relevant_at_time(self, time: int) -> bool:
        return time == 0


class QueryConstraint(Constraint):
    _assumable: bool = True
    query: Query

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 4


class StateVariableConstraint(Constraint):
    variable: str
    bounds: "Interval" = None
    timepoints: Optional["Interval"] = None

    model_config = ConfigDict(extra="forbid")

    def __hash__(self) -> int:
        return 3

    def contains_time(self, time: Union[float, int]) -> bool:
        return self.timepoints is None or self.timepoints.contains_value(time)

    def relevant_at_time(self, time: int) -> bool:
        return self.contains_time(time)
