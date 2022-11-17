"""
This submodule contains class definitions used to represent and interact with
models in FUNMAN.
"""
from typing import Union
from funman.constants import NEG_INFINITY, POS_INFINITY
from copy import deepcopy
from pysmt.shortcuts import Symbol, REAL


class Model(object):
    def __init__(self, formula) -> None:
        self.formula = formula


class Query(object):
    def __init__(self, formula) -> None:
        self.formula = formula


class Parameter(object):
    def __init__(
        self,
        name,
        lb: Union[float, str] = NEG_INFINITY,
        ub: Union[float, str] = POS_INFINITY,
        symbol=None,
    ) -> None:
        self.name = name
        self.lb = lb
        self.ub = ub

        # if the symbol is None, then need to get the symbol from a solver
        self._symbol = symbol

    def symbol(self):
        if self._symbol is None:
            self._symbol = Symbol(self.name, REAL)
        return self._symbol

    def timed_copy(self, timepoint):
        timed_parameter = deepcopy(self)
        timed_parameter.name = f"{timed_parameter.name}_{timepoint}"
        return timed_parameter

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name and (
            not (self._symbol and other._symbol)
            or (self._symbol.symbol_name() == other._symbol.symbol_name())
        )

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)
