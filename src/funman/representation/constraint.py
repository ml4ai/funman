from typing import Optional, List, Union
from pydantic import BaseModel, field_validator


class Constraint(BaseModel):
    _assumable: bool = True

    def assumable(self) -> bool:
        return self._assumable


class ModelConstraint(Constraint):
    _assumable: bool = False


class ParameterConstraint(Constraint):
    _assumable: bool = False


class StateVariableConstraint(Constraint):
    variable: str
    bounds: Optional[List[Union[float, int]]] = None
    timepoints: Optional[List[Union[float, int]]] = None

    @field_validator("bounds", "timepoints")
    @classmethod
    def well_formed_bounds(cls, b: Optional[List[Union[float, int]]]):
        if b is None:
            return b
        else:
            if len(b) == 2 and b[0] <= b[1]:
                return b
            else:
                raise ValueError(
                    "must be a pair of numbers, and the lower bound must be no more than the upper bound"
                )


