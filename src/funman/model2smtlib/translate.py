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
    get_model,
    simplify,
    substitute,
)

from funman.model import QueryLE, QueryTrue


class Encoding(object):
    def __init__(self, formula=None, symbols=None) -> None:
        self.formula = formula
        self.symbols = symbols


class EncodingOptions(object):
    def __init__(self, max_steps=2) -> None:
        self.max_steps = max_steps


class Encoder(object):
    def __init__(self, config: EncodingOptions = EncodingOptions()) -> None:
        self.config = config

    def _symbols(self, formula):
        symbols = {}
        vars = list(formula.get_free_variables())
        # vars.sort(key=lambda x: x.symbol_name())
        for var in vars:
            var_name, timepoint = self._split_symbol(var)
            if timepoint:
                if var_name not in symbols:
                    symbols[var_name] = {}
                symbols[var_name][timepoint] = var
        return symbols

    def encode_model(self, model: "EncodedModel"):
        return model.encoding

    def encode_query(self, model_encoding, query):
        query_handlers = {
            QueryLE: self._encode_query_le,
            QueryTrue: self._encode_query_true,
        }

        if type(query) in query_handlers:
            return query_handlers[type(query)](model_encoding, query)
        else:
            raise NotImplementedError(
                f"Do not know how to encode query of type {type(query)}"
            )

    def _encode_query_le(self, model_encoding, query):
        timepoints = model_encoding.symbols[query.variable]
        return Encoding(
            And([LE(s, Real(query.ub)) for s in timepoints.values()])
        )

    def _encode_query_true(self, model_encoding, query):
        return Encoding(TRUE())

    def symbol_timeseries(
        self, model_encoding, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Generate a symbol (str) to timeseries (list) of values

        Parameters
        ----------
        pysmtModel : pysmt.solvers.solver.Model
            variable assignment
        """
        series = self.symbol_values(model_encoding, pysmtModel)
        a_series = {}  # timeseries as array/list
        max_t = max(
            [max([int(k) for k in tps.keys()]) for _, tps in series.items()]
        )
        a_series["index"] = list(range(0, max_t + 1))
        for var, tps in series.items():

            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                vals[int(t)] = v
            a_series[var] = vals
        return a_series
