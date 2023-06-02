"""
This module defines the abstract base classes for the model encoder classes in funman.translate package.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

import pysmt
from numpy import isin
from pydantic import BaseModel, Extra
from pysmt.constants import Numeral
from pysmt.formula import FNode
from pysmt.shortcuts import GE, LE, LT, REAL, TRUE, And, Equals, Real, Symbol
from pysmt.solvers.solver import Model as pysmtModel

from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.funman import FUNMANConfig
from funman.model.model import Model
from funman.model.query import (
    Query,
    QueryAnd,
    QueryEncoded,
    QueryGE,
    QueryLE,
    QueryTrue,
)
from funman.representation import Parameter
from funman.representation.representation import Box, Interval, Point


class Encoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    _formula: FNode = None
    _symbols: Union[List[FNode], Dict[str, Dict[str, FNode]]] = None
    _substitutions: Dict[FNode, FNode] = {}

    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)


class EncodingOptions(object):
    """
    EncodingOptions
    """

    def __init__(self, max_steps=2) -> None:
        self.max_steps = max_steps


class Encoder(ABC, BaseModel):
    """
    An Encoder translates a Model into an SMTLib formula.

    """

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    config: FUNMANConfig
    _timed_model_elements: Dict = None
    _min_time_point: int
    _min_step_size: int
    _untimed_symbols: Set[str] = set([])
    _timed_symbols: Set[str] = set([])
    _untimed_constraints: FNode
    _assignments: Dict[str, float]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model = kwargs["model"]
        self._encode_timed_model_elements(model)

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

    def _encode_next_step(
        self, model: Model, step: int, next_step: int
    ) -> FNode:
        pass

    def encode_model_timed(
        self, model: "Model", num_steps: int, step_size: int
    ) -> Encoding:
        """
        Encode a model into an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode
        num_steps: int
            number of encoding steps (e.g., time steps)
        step_size: int
            size of a step

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """

        state_timepoints, transition_timepoints = self._get_timepoints(
            num_steps, step_size
        )
        # parameters = model._parameters()

        constraints = []

        for i, timepoint in enumerate(transition_timepoints):
            c = self._timed_model_elements["time_step_constraints"][timepoint][
                step_size - self._min_step_size
            ]
            if c is None:
                c = self._encode_next_step(
                    model,
                    state_timepoints[i],
                    state_timepoints[i + 1],
                )
                self._timed_model_elements["time_step_constraints"][timepoint][
                    step_size - self._min_step_size
                ] = c
            constraints.append(c)

        #     if time_dependent_parameters:
        #         params = self._timed_model_elements["timed_parameters"][
        #             timepoint
        #         ][step_size - self._min_step_size]
        #         if params == []:
        #             params = [p.timed_copy(timepoint) for p in parameters]
        #             self._timed_model_elements["timed_parameters"][timepoint][
        #                 step_size - self._min_step_size
        #             ] = params
        #         constraints.append(
        #             self.box_to_smt(
        #                 Box(
        #                     bounds={
        #                         p.name: Interval(lb=p.lb, ub=p.ub)
        #                         for p in timed_parameters
        #                     }
        #                 ),
        #                 closed_upper_bound=True,
        #             )
        #         )

        # if time_dependent_parameters:
        #     # FIXME cache this computation
        #     ## Assume that all parameters are constant
        #     constraints.append(
        #         self._set_parameters_constant(
        #             parameters,
        #             constraints,
        #         ),
        #     )

        formula = And(
            And(
                [
                    self._timed_model_elements["init"],
                    self._timed_model_elements["untimed_constraints"],
                ]
                + constraints
            ).simplify(),
            (model._extra_constraints if model._extra_constraints else TRUE()),
        ).simplify()

        symbols = self._symbols(formula)
        if self.config.substitute_subformulas:
            (
                substituted_formulas,
                substitutions,
                assignments,
            ) = self._substitute_subformulas(
                [
                    self._timed_model_elements["untimed_constraints"],
                    self._timed_model_elements["init"],
                ]
                + constraints,
                model,
            )
            self._assignments = assignments
            substituted_formula = And(substituted_formulas).simplify()
            return Encoding(
                _formula=substituted_formula,
                _symbols=symbols,
                _substitutions=substitutions,
            )
        else:
            formula = And(
                And(
                    [
                        self._timed_model_elements["init"],
                        self._timed_model_elements["untimed_constraints"],
                    ]
                    + constraints
                ).simplify(),
                (
                    model._extra_constraints
                    if model._extra_constraints
                    else TRUE()
                ),
            ).simplify()

            symbols = self._symbols(formula)
            return Encoding(_formula=formula, _symbols=symbols)

    def _substitute_subformulas(self, formula: List[FNode], model: "Model"):
        # Add parameter assignments
        parameters = model._parameters()
        parameter_assignments = {
            self._encode_state_var(k.name): (
                Real(float(k.lb))
                if k.name not in model.parameter_bounds
                else Real(model.parameter_bounds[k.name][0])
            )
            for k in parameters
            if k.lb == k.ub
            and (
                (not k.name in model.parameter_bounds)
                or model.parameter_bounds[k.name][0]
                == model.parameter_bounds[k.name][1]
            )
        }

        init_assignments = {
            self._encode_state_var(k, time=0): Real(model._get_init_value(k))
            for k in model._state_var_names()
        }

        substitutions = {**parameter_assignments, **init_assignments}

        processed = []
        for f in formula:
            fs = f.substitute(substitutions)
            fs = fs.simplify()
            assn = self._get_assignments(fs)
            substitutions = {**substitutions, **assn}
            fs = fs.substitute(
                {k: v for k, v in assn.items() if v.is_constant()}
            ).simplify()
            processed.append(fs)

        assignments = {
            k.symbol_name(): float(v.constant_value())
            for k, v in substitutions.items()
            if v.is_constant()
        }

        return processed, substitutions, assignments

    def _get_assignments(self, formula: FNode):
        atoms = formula.get_atoms()
        assignments = {
            (a.args()[0] if a.args()[0].is_symbol() else a.args()[1]): (
                a.args()[1] if not a.args()[1].is_symbol() else a.args()[0]
            )
            for a in atoms
            if a.is_equals()
            and ((a.args()[0].is_symbol()) or (a.args()[1].is_symbol()))
        }
        return assignments

    def parameter_values(
        self, model: Model, pysmtModel: pysmtModel
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Gather values assigned to model parameters.

        Parameters
        ----------
        model : Model
            model encoded by self
        pysmtModel : pysmt.solvers.solver.Model
            the assignment to symbols

        Returns
        -------
        Dict[str, List[Union[float, None]]]
            mapping from parameter symbol name to value
        """
        try:
            parameters = {
                parameter.name: pysmtModel[parameter.name]
                for parameter in model._parameters()
                if parameter.name in pysmtModel
            }

            return parameters
        except OverflowError as e:
            l.warning(e)
            return {}

    def _get_timed_symbols(self, model: Model) -> List[str]:
        """
        Get the names of the state (i.e., timed) variables of the model.

        Parameters
        ----------
        model : Model
            The petrinet model

        Returns
        -------
        List[str]
            state variable names
        """
        pass

    def _get_untimed_symbols(self, model: Model) -> List[str]:
        untimed_symbols = []
        # All flux nodes correspond to untimed symbols
        for var_name in model._parameter_names():
            untimed_symbols.append(var_name)
        return untimed_symbols

    def _encode_state_var(self, var: str, time: int = None):
        timing = f"_{time}" if time is not None else ""
        return Symbol(f"{var}{timing}", REAL)

    def _get_structural_configurations(self, model: Model):
        configurations: List[Dict[str, int]] = []
        if len(model.structural_parameter_bounds) == 0:
            self._min_time_point = 0
            self._min_step_size = 1
            num_steps = [1, self.config.num_steps]
            step_size = [1, self.config.step_size]
            max_step_size = self.config.step_size
            max_step_index = self.config.num_steps * max_step_size
            configurations.append(
                {
                    "num_steps": self.config.num_steps,
                    "step_size": self.config.step_size,
                }
            )
        else:
            num_steps = model.structural_parameter_bounds["num_steps"]
            step_size = model.structural_parameter_bounds["step_size"]
            self._min_time_point = num_steps[0]
            self._min_step_size = step_size[0]
            max_step_size = step_size[1]
            max_step_index = num_steps[1] * max_step_size
            configurations += [
                {"num_steps": ns, "step_size": ss}
                for ns in range(num_steps[0], num_steps[1] + 1)
                for ss in range(step_size[0], step_size[1] + 1)
            ]
        return configurations, max_step_index, max_step_size

    def _define_init_term(self, model: Model, var: str, init_time: int):
        value = model._get_init_value(var)

        if (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, str)
        ):
            value_symbol = (
                Symbol(value, REAL) if isinstance(value, str) else Real(value)
            )
            return Equals(
                self._encode_state_var(var, time=init_time),
                value_symbol,
            )
        elif isinstance(value, list):
            return And(
                GE(
                    self._encode_state_var(var, time=init_time),
                    Real(value[0]),
                ),
                LT(
                    self._encode_state_var(var, time=init_time),
                    Real(value[1]),
                ),
            )
        else:
            return TRUE()

    def _define_init(self, model: Model, init_time: int = 0) -> FNode:
        state_var_names = model._state_var_names()
        return And(
            [
                self._define_init_term(model, var, init_time)
                for var in state_var_names
            ]
        )

    def _encode_untimed_constraints(self, model: Model) -> FNode:
        untimed_constraints = []
        parameters = model._parameters()

        # If parameter_bounds exist, then override those encoded in the original model
        overridden_parameters = [
            (
                p
                if p.name not in model.parameter_bounds
                else Parameter(
                    name=p.name,
                    lb=model.parameter_bounds[p.name][0],
                    ub=model.parameter_bounds[p.name][1],
                )
            )
            for p in parameters
        ]

        # Create bounds on parameters, but not necessarily synthesize the parameters
        untimed_constraints.append(
            self.box_to_smt(
                Box(
                    bounds={
                        p.name: Interval(lb=p.lb, ub=p.ub)
                        for p in overridden_parameters
                    },
                    closed_upper_bound=True,
                )
            )
        )

        return And(untimed_constraints).simplify()

    def _encode_timed_model_elements(self, model: Model):
        self._timed_symbols = self._get_timed_symbols(model)
        self._untimed_symbols = self._get_untimed_symbols(model)

        (
            configurations,
            max_step_index,
            max_step_size,
        ) = self._get_structural_configurations(model)
        self._timed_model_elements = {
            "init": self._define_init(model),
            "time_step_constraints": [
                [None for i in range(max_step_size)]
                for j in range(max_step_index)
            ],
            "configurations": configurations,
            "untimed_constraints": self._encode_untimed_constraints(model),
            "timed_parameters": [
                [None for i in range(max_step_size)]
                for j in range(max_step_index)
            ],
        }

    def _get_timepoints(
        self, num_steps: int, step_size: int
    ) -> Tuple[List[int], List[int]]:
        state_timepoints = range(
            0,
            (step_size * num_steps) + 1,
            step_size,
        )

        if len(list(state_timepoints)) == 0:
            raise Exception(
                f"Could not identify timepoints from step_size = {step_size} and num_steps = {num_steps}"
            )

        transition_timepoints = range(0, step_size * num_steps, step_size)
        return list(state_timepoints), list(transition_timepoints)

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
            QueryAnd: self._encode_query_and,
            QueryLE: self._encode_query_le,
            QueryGE: self._encode_query_ge,
            QueryTrue: self._encode_query_true,
            QueryEncoded: self._return_encoded_query,
        }

        if type(query) in query_handlers:
            encoded_query = query_handlers[type(query)](model_encoding, query)
            simplified_query = encoded_query._formula.substitute(
                model_encoding._substitutions
            )
            encoded_query._formula = simplified_query.simplify()
            return encoded_query
        else:
            raise NotImplementedError(
                f"Do not know how to encode query of type {type(query)}"
            )

    def _return_encoded_query(self, model_encoding, query):
        return Encoding(_formula=query._formula)

    def _query_variable_name(self, query):
        return (
            query.variable
            if query.model is None
            else f"model_{query.model.name}_{query.variable}"
        )

    def _encode_query_and(self, model_encoding, query):
        encodings = [
            self.encode_query(model_encoding, q) for q in query.queries
        ]
        return Encoding(_formula=And([e._formula for e in encodings]))

    def _encode_query_le(self, model_encoding, query):
        query_variable_name = self._query_variable_name(query)
        if query_variable_name not in model_encoding._symbols:
            raise Exception(
                f"Could not encode QueryLE because {query_variable_name} does not appear in the model_encoding symbols."
            )
        timepoints = model_encoding._symbols[query_variable_name]
        return Encoding(
            _formula=And([LE(s, Real(query.ub)) for s in timepoints.values()])
        )

    def _encode_query_ge(self, model_encoding, query):
        query_variable_name = self._query_variable_name(query)

        if query_variable_name not in model_encoding._symbols:
            raise Exception(
                f"Could not encode QueryGE because {query_variable_name} does not appear in the model_encoding symbols."
            )
        timepoints = model_encoding._symbols[query_variable_name]
        return Encoding(
            _formula=And([GE(s, Real(query.lb)) for s in timepoints.values()])
        )

    def _encode_query_true(self, model_encoding, query):
        return Encoding(_formula=TRUE())

    def symbol_timeseries(
        self, model_encoding, pysmtModel: pysmtModel
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
            [
                max([int(k) for k in tps.keys() if k.isdigit()] + [0])
                for _, tps in series.items()
            ]
        )
        a_series["index"] = list(range(0, max_t + 1))
        for var, tps in series.items():
            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                if t.isdigit():
                    vals[int(t)] = v
            a_series[var] = vals
        return a_series

    def symbol_values(
        self, model_encoding: Encoding, pysmtModel: pysmtModel
    ) -> Dict[str, Dict[str, float]]:
        """
         Get the value assigned to each symbol in the pysmtModel.

        Parameters
        ----------
        model_encoding : Encoding
            encoding using the symbols
        pysmtModel : pysmt.solvers.solver.Model
            assignment to symbols

        Returns
        -------
        Dict[str, Dict[str, float]]
            mapping from symbol and timepoint to value
        """

        vars = model_encoding._symbols
        vals = {}
        for var in vars:
            vals[var] = {}
            for t in vars[var]:
                try:
                    symbol = vars[var][t]
                    value = pysmtModel.get_py_value(symbol)
                    if isinstance(value, Numeral):
                        value = 0.0
                    vals[var][t] = float(value)
                except OverflowError as e:
                    l.warning(e)
        return vals

    def interval_to_smt(
        self, p: str, i: Interval, closed_upper_bound: bool = False
    ) -> FNode:
        """
        Convert the interval into contraints on parameter p.

        Parameters
        ----------
        p : Parameter
            parameter to constrain
        closed_upper_bound : bool, optional
            interpret interval as closed (i.e., p <= ub), by default False

        Returns
        -------
        FNode
            formula constraining p to the interval
        """
        if i.lb == i.ub and i.lb != NEG_INFINITY and i.lb != POS_INFINITY:
            return Equals(Symbol(p, REAL), Real(i.lb))
        else:
            lower = (
                GE(Symbol(p, REAL), Real(i.lb))
                if i.lb != NEG_INFINITY
                else TRUE()
            )
            upper_ineq = LE if closed_upper_bound else LT
            upper = (
                upper_ineq(Symbol(p, REAL), Real(i.ub))
                if i.ub != POS_INFINITY
                else TRUE()
            )
            return And(
                lower,
                upper,
            ).simplify()

    def point_to_smt(self, pt: Point):
        return And(
            [Equals(p.symbol(), Real(value)) for p, value in pt.values.items()]
        )

    def box_to_smt(self, box: Box, closed_upper_bound: bool = False):
        """
        Compile the interval for each parameter into SMT constraints on the corresponding parameter.

        Parameters
        ----------
        closed_upper_bound : bool, optional
            use closed upper bounds for each interval, by default False

        Returns
        -------
        FNode
            formula representing the box as a conjunction of interval constraints.
        """
        return And(
            [
                self.interval_to_smt(
                    p, interval, closed_upper_bound=closed_upper_bound
                )
                for p, interval in box.bounds.items()
            ]
        )

    def _split_symbol(self, symbol: FNode) -> Tuple[str, str]:
        if symbol.symbol_name() in self._untimed_symbols:
            return symbol.symbol_name(), None
        else:
            s, t = symbol.symbol_name().rsplit("_", 1)
            if s not in self._timed_symbols or not t.isdigit():
                raise Exception(
                    f"Cannot determine if symbol {symbol} is timed."
                )
            return s, t


class DefaultEncoder(Encoder):
    """
    The DefaultEncoder will not actually encode a model as SMT.  It is used to provide an Encoder for SimulatorModel objects, but the encoder will not be used.
    """

    pass
