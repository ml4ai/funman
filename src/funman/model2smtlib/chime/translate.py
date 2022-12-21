import logging
from typing import Dict, List, Union

import pysmt
from pysmt.shortcuts import (
    FALSE,
    GE,
    GT,
    LE,
    LT,
    TRUE,
    And,
    Equals,
    ForAll,
    Function,
    FunctionType,
    Iff,
    Int,
    Plus,
    Real,
    Symbol,
    Times,
    get_free_variables,
    get_model,
    simplify,
    substitute,
)
from pysmt.typing import BOOL, INT, REAL

from funman.model import Model, Parameter, QueryLE, QueryTrue
from funman.model2smtlib import QueryableModel
from funman.model2smtlib.translate import Encoder, Encoding, EncodingOptions
from funman.model.bilayer import Bilayer, BilayerMeasurement, BilayerModel
from funman.model.chime import ChimeModel
from funman.utils.search_utils import Box

l = logging.Logger(__name__)


class ChimeEncoder(Encoder):
    def encode_model(self, model: ChimeModel):
        epochs = model.config["epochs"]
        population_size = model.config["population_size"]
        infectious_days = model.config["infectious_days"]
        # infected_threshold = config["infected_threhold"]
        vars, model = model.chime.make_model(
            epochs=epochs,
            population_size=population_size,
            infectious_days=infectious_days,
            infected_threshold=0.1,
            linearize=model.config.get("linearize", False),
        )
        (default_parameters, init, dynamics, query) = model

        # Associate parameters with symbols in the model
        # symbol_map = {
        #     s.symbol_name(): s
        #     for p in parameters
        #     for s in get_free_variables(p)
        # }
        # for p in parameters:
        #     if not p._symbol:
        #         p._symbol = symbol_map[p.name]

        # param_symbols = set({p.name for p in self.parameters})
        # assigned_parameters = [
        #     p
        #     for p in parameters
        #     if len(
        #         set(
        #             {q.symbol_name() for q in get_free_variables(p)}
        #         ).intersection(param_symbols)
        #     )
        #     == 0
        # ]

        formula = And(
            # And(assigned_parameters),
            init,
            (
                And([And(layer) for step in dynamics for layer in step])
                if isinstance(dynamics, list)
                else dynamics
            ),
        )
        symbols = self._symbols(formula)
        return Encoding(formula=formula, symbols=symbols)

    def _encode_measurements_timepoint(self, measurements, t):
        observable_defs = And(
            [
                Equals(
                    o.to_smtlib(t), self._observable_defn(measurements, o, t)
                )
                for o in measurements.observable.values()
            ]
        )
        return observable_defs

    def _observable_defn(self, measurements, obs, t):
        # flux * incoming1 * incoming2 ...
        obs_in_edges = measurements.node_incoming_edges[obs]
        result = Real(0.0)
        for src in obs_in_edges:
            # src is a flux
            f_t = src.to_smtlib(t)
            src_srcs = [
                s.to_smtlib(t) for s in measurements.node_incoming_edges[src]
            ]
            result = Plus(result, Times([f_t] + src_srcs)).simplify()
        # flux = next([measurements.output_edges])
        return result

    def _set_parameters_constant(self, parameters, formula):
        params = {
            parameter: Symbol(parameter, REAL) for parameter in parameters
        }

        symbols = self._symbols(formula)
        all_equal = And(
            [
                And([Equals(params[p], s) for t, s in symbols[p].items()])
                for p in params
            ]
        )
        return all_equal

    def _split_symbol(self, symbol):
        if "_" in symbol.symbol_name():
            return symbol.symbol_name().rsplit("_", 1)
        else:
            return symbol.symbol_name(), None

    def symbol_values(self, model_encoding, pysmtModel):
        vars = model_encoding.symbols
        vals = {}
        for var in vars:
            vals[var] = {}
            for t in vars[var]:
                try:
                    symbol = vars[var][t]
                    vals[var][t] = float(pysmtModel.get_py_value(symbol))
                except OverflowError as e:
                    l.warn(e)
        return vals

    def parameter_values(
        self, model, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, List[Union[float, None]]]:
        try:
            parameters = {
                node.parameter: pysmtModel[Symbol(node.parameter, REAL)]
                for _, node in model.bilayer.flux.items()
            }
            return parameters
        except OverflowError as e:
            l.warn(e)
            return {}
