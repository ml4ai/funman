"""
This submodule contains class definitions used to represent and interact with
models in FUNMAN.
"""
from typing import Union
from funman.constants import NEG_INFINITY, POS_INFINITY
from copy import deepcopy
from funman.examples.chime import CHIME
from pysmt.shortcuts import Symbol, REAL, get_free_variables, And, Real, LE


class Model(object):
    def __init__(self, formula) -> None:
        self.formula = formula


class EncodedModel(Model):
    def __init__(self, formula) -> None:
        super().__init__(formula)


class CannedModel(Model):
    pass


class ChimeModel(CannedModel):
    def __init__(self, name, config, chime: CHIME) -> None:
        super().__init__(None)
        self.name = name
        self.config = config
        self.chime = chime
        self.formula = self._encode()

    def _encode(self):

        epochs = self.config["epochs"]
        population_size = self.config["population_size"]
        infectious_days = self.config["infectious_days"]
        # infected_threshold = config["infected_threhold"]
        vars, model = self.chime.make_model(
            epochs=epochs,
            population_size=population_size,
            infectious_days=infectious_days,
            infected_threshold=0.1,
            linearize=self.config.get("linearize", False),
        )
        # Associate parameters with symbols in the model
        symbol_map = {
            s.symbol_name(): s for p in model[0] for s in get_free_variables(p)
        }
        for p in self.parameters:
            if not p._symbol:
                p._symbol = symbol_map[p.name]

        param_symbols = set({p.name for p in self.parameters})
        assigned_parameters = [
            p
            for p in model[0]
            if len(
                set(
                    {q.symbol_name() for q in get_free_variables(p)}
                ).intersection(param_symbols)
            )
            == 0
        ]

            # Query(And(model[3]) if isinstance(model[3], list) else model[3]),
        return And(
                And(assigned_parameters),
                model[1],
                (
                    And([And(layer) for step in model[2] for layer in step])
                    if isinstance(model[2], list)
                    else model[2]
                ),
            )
        


class Query(object):
    def __init__(self, formula) -> None:
        self.formula = formula


class QueryLE(Query):
    def __init__(self, model, variable, ub) -> None:
        super().__init__(None)
        timepoints = model.symbols[variable]
        self.formula = And([LE(s, Real(ub)) for s in timepoints.values()])


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
