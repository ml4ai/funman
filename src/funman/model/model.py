"""
This module represents the abstract base classes for models.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Union

from pydantic import BaseModel
from pysmt.fnode import FNode
from pysmt.shortcuts import REAL, Symbol

from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.query import *


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    init_values: Dict[str, float] = None
    parameter_bounds: Dict[str, List[float]] = None

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

    name: str
    lb: Union[float, str] = NEG_INFINITY
    ub: Union[float, str] = POS_INFINITY
    # symbol: FNode = None

    class Config:
        arbitrary_types_allowed = True

    def _symbol(self):
        if self.__symbol is None:
            self.__symbol = Symbol(self.name, REAL)
        return self.__symbol

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
            not (self.__symbol and other.__symbol)
            or (self.__symbol.symbol_name() == other.__symbol.symbol_name())
        )

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}[{self.lb}, {self.ub})"
