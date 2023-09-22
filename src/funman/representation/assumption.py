from funman.model.query import Query
from pydantic import BaseModel
from typing import Union
from .constraint import (
    ModelConstraint,
    ParameterConstraint,
    StateVariableConstraint,
)


class Assumption(BaseModel):
    constraint: Union[
        ModelConstraint, ParameterConstraint, StateVariableConstraint, Query
    ]
