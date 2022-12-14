"""
This module defines the abstract base classes for the model encoder classes in funman.translate package.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pysmt
from pysmt.formula import FNode
from pysmt.shortcuts import LE, TRUE, And, Real

from funman.model.query import Query, QueryEncoded, QueryLE, QueryTrue


class Encoding(object):
    """
    An encoding comprises a formula over a set of symbols.

    """

    def __init__(
        self,
        formula: FNode = None,
        symbols: Dict[str, Dict[str, FNode]] = None,
    ) -> None:
        self.formula = formula
        self.symbols = symbols


class EncodingOptions(object):
    """
    EncodingOptions
    """

    def __init__(self, max_steps=2) -> None:
        self.max_steps = max_steps


class Encoder(ABC):
    """
    An Encoder translates a Model into an SMTLib formula.

    """

    def __init__(self, config: EncodingOptions = EncodingOptions()) -> None:
        self.config = config

    def _symbols(self, formula: FNode) -> Dict[str, Dict[str, FNode]]:
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

    @abstractmethod
    def encode_model(self, model: "Model") -> Encoding:
        """
        Encode a model into an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """
        pass

    def encode_query(self, model_encoding: Encoding, query: Query) -> Encoding:
        """
        Encode a query into an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """
        query_handlers = {
            QueryLE: self._encode_query_le,
            QueryTrue: self._encode_query_true,
            QueryEncoded: self._return_encoded_query,
        }

        if type(query) in query_handlers:
            return query_handlers[type(query)](model_encoding, query)
        else:
            raise NotImplementedError(
                f"Do not know how to encode query of type {type(query)}"
            )

    def _return_encoded_query(self, model_encoding, query):
        return query

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


class DefaultEncoder(Encoder):
    """
    The DefaultEncoder will not actually encode a model as SMT.  It is used to provide an Encoder for SimulatorModel objects, but the encoder will not be used.
    """

    pass
