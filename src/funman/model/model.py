"""
This module represents the abstract base classes for models.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Union

from pydantic import BaseModel, validator
from pysmt.fnode import FNode
from pysmt.shortcuts import REAL, Symbol

from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.query import *


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    init_values: Dict[str, float] = None
    parameter_bounds: Dict[str, List[float]] = {}

    @abstractmethod
    def default_encoder(self) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        pass


class Parameter(BaseModel):
    """
    A parameter is a free variable for a Model.  It has the following attributes:

    * lb: lower bound

    * ub: upper bound

    * symbol: a pysmt FNode corresponding to the parameter variable

    """

    class Config:
        underscore_attrs_are_private = True
        # arbitrary_types_allowed = True

    name: str
    lb: Union[str, float] = NEG_INFINITY
    ub: Union[str, float] = POS_INFINITY
    _symbol: FNode = None

    def symbol(self):
        if not self._symbol:
            self._symbol = Symbol(self.name, REAL)
        return self._symbol

    def timed_copy(self, timepoint: int):
        """
        Create a time-stamped copy of a parameter.  E.g., beta becomes beta_t for a timepoint t

        Parameters
        ----------
        timepoint : int
            Integer timepoint

        Returns
        -------
        Parameter
            A time-stamped copy of self.
        """
        timed_parameter = deepcopy(self)
        timed_parameter.name = f"{timed_parameter.name}_{timepoint}"
        return timed_parameter

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and (
            not (self.symbol() and other.symbol())
            or (self.symbol().symbol_name() == other.symbol().symbol_name())
        )

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}[{self.lb}, {self.ub})"
