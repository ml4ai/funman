import copy
import logging
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict
from pysmt.fnode import FNode
from pysmt.shortcuts import REAL, Symbol

import funman.utils.math_utils as math_utils
from funman import (
    LABEL_ALL,
    LABEL_ANY,
    NEG_INFINITY,
    POS_INFINITY,
)

from .symbol import ModelSymbol

l = logging.getLogger(__name__)


class Parameter(BaseModel):
    name: Union[str, ModelSymbol]
    lb: Union[float, str] = NEG_INFINITY
    ub: Union[float, str] = POS_INFINITY

    def width(self) -> Union[str, float]:
        return math_utils.minus(self.ub, self.lb)

    def is_unbound(self) -> bool:
        return self.lb == NEG_INFINITY and self.ub == POS_INFINITY

    def __hash__(self):
        return abs(hash(self.name))


class LabeledParameter(Parameter):
    label: Literal["any", "all"] = LABEL_ANY

    def is_synthesized(self) -> bool:
        return self.label == LABEL_ALL and self.width() > 0.0


class StructureParameter(LabeledParameter):
    def is_synthesized(self):
        return True


class ModelParameter(LabeledParameter):
    """
    A parameter is a free variable for a Model.  It has the following attributes:

    * lb: lower bound

    * ub: upper bound

    * symbol: a pysmt FNode corresponding to the parameter variable

    """

    model_config = ConfigDict(extra="forbid")

    _symbol: FNode = None

    def symbol(self):
        """
        Get a pysmt Symbol for the parameter

        Returns
        -------
        pysmt.fnode.FNode
            _description_
        """
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
        timed_parameter = copy.deepcopy(self)
        timed_parameter.name = f"{timed_parameter.name}_{timepoint}"
        return timed_parameter

    def __eq__(self, other):
        if not isinstance(other, ModelParameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}[{self.lb}, {self.ub})"
