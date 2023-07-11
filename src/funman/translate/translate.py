"""
This module defines the abstract base classes for the model encoder classes in funman.translate package.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

import pysmt
from numpy import isin
from pydantic import BaseModel, Extra
from pysmt.constants import Numeral
from pysmt.formula import FNode
from pysmt.shortcuts import (
    GE,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Equals,
    Iff,
    Real,
    Symbol,
    get_env,
)
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
from funman.representation.representation import (
    Box,
    Interval,
    ModelParameter,
    Point,
)
from funman.representation.symbol import ModelSymbol
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils.sympy_utils import FUNMANFormulaManager

l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)


class Encoding(BaseModel):
    _substitutions: Dict[FNode, FNode] = {}


class FlatEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    _formula: FNode = None
    _symbols: Union[List[FNode], Dict[str, Dict[str, FNode]]] = None

    def encoding(self):
        return _formula

    def assume(self, assumption: FNode):
        _formula = Iff(assumption, _formula)

    def symbols(self):
        return self._symbols

    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)


class LayeredEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow

    _layers: List[
        Tuple[FNode, Union[List[FNode], Dict[str, Dict[str, FNode]]]]
    ] = []
    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)

    def encoding(self, layers=None):
        if layers:
            return And([self._layers[i][0] for i in layers])
        else:
            return And([l[0] for l in self._layers])

    def assume(self, assumption: List[FNode], layers=None):
        for i, l in enumerate(self._layers):
            (f, s) = l
            self._layers[i] = (
                (Iff(assumption[i], f), s)
                if not layers or i in layers
                else (f, s)
            )

    def substitute(self, substitutions: Dict[FNode, FNode]):
        self._layers = [
            (layer[0].substitute(substitutions), layer[1])
            for layer in self._layers
        ]

    def simplify(self):
        self._layers = [
            (layer[0].simplify(), layer[1]) for layer in self._layers
        ]

    def symbols(self):
        return { k:v for layer in self._layers for k,v in layer[1].items()}



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
    # _assignments: Dict[str, float] = {}
    env = get_env()
    env._simplifier = FUNMANSimplifier(env)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        scenario = kwargs["scenario"]
        self._encode_timed_model_elements(scenario)

    def _symbols(self, vars: List[FNode]) -> Dict[str, Dict[str, FNode]]:
        symbols = {}
        # vars.sort(key=lambda x: x.symbol_name())
        for var in vars.values():
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
        self, model: Model, step: int, next_step: int, substitutions={}
    ) -> FNode:
        pass

    def encode_model_timed(
        self, scenario: "AnalysisScenario", num_steps: int, step_size: int
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
        l.debug("Encoding step: {current_step} to {next_step}")
        env = get_env()
        if not isinstance(env._formula_manager, FUNMANFormulaManager):
            env._formula_manager = FUNMANFormulaManager(env._formula_manager)
            # Before calling substitute, need to replace the formula_manager
            env._substituter.mgr = env.formula_manager

        state_timepoints, transition_timepoints = self._get_timepoints(
            num_steps, step_size
        )

        # Need to initialize pysmt symbols for parameters to help with parsing custom rate equations
        variables = [
            p.name for p in scenario.model._parameters()
        ] + scenario.model._state_var_names()
        variable_symbols = [self._encode_state_var(p) for p in variables]

        substitutions = self._initialize_substitutions(scenario)

        initial_state = self._timed_model_elements["init"]
        initial_symbols = initial_state.get_free_variables()
        layers = [
(           initial_state,
            {str(s):s for s in initial_symbols})
        ]

        for i, timepoint in enumerate(transition_timepoints):
            c = self._timed_model_elements["time_step_constraints"][timepoint][
                step_size - self._min_step_size
            ]
            c_substitutions = self._timed_model_elements[
                "time_step_substitutions"
            ][timepoint][step_size - self._min_step_size]
            if c_substitutions is not None:
                substitutions = {**substitutions, **c_substitutions}
            if c is None:
                c, substitutions = self._encode_next_step(
                    scenario,
                    state_timepoints[i],
                    state_timepoints[i + 1],
                    substitutions=substitutions,
                )
                self._timed_model_elements["time_step_constraints"][timepoint][
                    step_size - self._min_step_size
                ] = c
                self._timed_model_elements["time_step_substitutions"][
                    timepoint
                ][step_size - self._min_step_size] = substitutions
            layers.append((c,{str(s):s for s in c.get_free_variables()}))

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

        # formula = And(
        #     And(
        #         [
        #             ,
        #             # self._timed_model_elements["untimed_constraints"],
        #         ]
        #         + constraints
        #     ).simplify(),
        #     (model._extra_constraints if model._extra_constraints else TRUE()),
        # ).simplify()
        # symbols = self._symbols(formula)

        # if self.config.substitute_subformulas:
        #     # (
        #     #     substituted_formulas,
        #     #     substitutions,
        #     #     assignments,
        #     # ) = self._substitute_subformulas(
        #     #     [
        #     #         self._timed_model_elements["untimed_constraints"],
        #     #         self._timed_model_elements["init"],
        #     #     ]
        #     #     + constraints,
        #     #     model,
        #     # )
        #     self._assignments = {
        #         k.symbol_name(): float(v.constant_value())
        #         for k, v in substitutions.items()
        #         if v.is_constant()
        #     }

        return LayeredEncoding(
            _layers=layers,
            substitutions=substitutions,
        )

    def _initialize_substitutions(self, scenario: "AnalysisScenario"):
        # Add parameter assignments
        parameters = scenario.model_parameters()
        parameter_assignments = {
            self._encode_state_var(k.name): (Real(float(k.lb)))
            for k in parameters
            if k.lb == k.ub
        }

        init_assignments = {
            self._encode_state_var(k, time=0): Real(
                scenario.model._get_init_value(k)
            )
            for k in scenario.model._state_var_names()
        }

        time_var = scenario.model._time_var()
        if time_var is not None:
            time_var_name = scenario.model._time_var_id(time_var)
            time_symbol = self._encode_state_var(
                time_var_name, time=0
            )  # Needed so that there is a pysmt symbol for 't'

            if self.config.substitute_subformulas:
                init_assignments[time_symbol] = Real(0.0)

        substitutions = {**parameter_assignments, **init_assignments}

        return substitutions

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

    def _get_structural_configurations(self, scenario: "AnalysisScenario"):
        configurations: List[Dict[str, int]] = []
        structure_parameters = scenario.structure_parameters()
        if len(structure_parameters) == 0:
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
            num_steps = scenario.structure_parameter("num_steps")
            step_size = scenario.structure_parameter("step_size")
            self._min_time_point = int(num_steps.lb)
            self._min_step_size = int(step_size.lb)
            max_step_size = int(step_size.ub)
            max_step_index = int(num_steps.ub) * int(max_step_size)
            configurations += [
                {"num_steps": ns, "step_size": ss}
                for ns in range(int(num_steps.lb), int(num_steps.ub) + 1)
                for ss in range(int(step_size.lb), int(step_size.ub) + 1)
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

    def _encode_untimed_constraints(
        self, scenario: "AnalysisScenario"
    ) -> FNode:
        untimed_constraints = []
        parameters = [
            p
            for p in scenario.model._parameters()
            if p not in scenario.parameters
        ] + scenario.parameters

        # Create bounds on parameters, but not necessarily synthesize the parameters
        untimed_constraints.append(
            self.box_to_smt(
                Box(
                    bounds={
                        p.name: Interval(lb=p.lb, ub=p.ub)
                        for p in parameters
                        if isinstance(p, ModelParameter)
                    },
                    closed_upper_bound=True,
                )
            )
        )

        return And(untimed_constraints).simplify()

    def _encode_timed_model_elements(self, scenario: "AnalysisScenario"):
        model = scenario.model
        self._timed_symbols = self._get_timed_symbols(model)
        self._untimed_symbols = self._get_untimed_symbols(model)

        (
            configurations,
            max_step_index,
            max_step_size,
        ) = self._get_structural_configurations(scenario)
        self._timed_model_elements = {
            "init": self._define_init(model),
            "time_step_constraints": [
                [None for i in range(max_step_size)]
                for j in range(max_step_index)
            ],
            "time_step_substitutions": [
                [None for i in range(max_step_size)]
                for j in range(max_step_index)
            ],
            "configurations": configurations,
            "untimed_constraints": self._encode_untimed_constraints(scenario),
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

    def encode_query(
        self, query: Query, num_steps: int, step_size: int, substitutions={}
    ) -> Encoding:
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
            encoded_query = query_handlers[type(query)](
                query, num_steps, step_size
            )
            encoded_query.substitute(substitutions)
            encoded_query.simplify()
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
            if not isinstance(query.variable, ModelSymbol)
            else str(query.variable)
        )

    def _encode_query_and(self, query, num_steps, step_size):
        queries = [
            self.encode_query(q, num_steps, step_size) for q in query.queries
        ]
        timepoints = range(0, (step_size * num_steps) + 1, step_size)
        q_layers = [q._layers for q in queries]
        layers = [
            (
                And([q_layer[i][0] for q_layer in q_layers]),
                {str(s):s for q_layer in q_layers for s in q_layer[i][1]},
            )
            for i, t in enumerate(timepoints)
        ]
        return LayeredEncoding(_layers=layers)

    def _encode_query_le(self, query, num_steps, step_size):
        query_variable_name = self._query_variable_name(query)
        timepoints = range(0, (step_size * num_steps) + 1, step_size)
        layers = [
            LE(
                self._encode_state_var(var=query.variable, time=s),
                Real(query.ub),
            )
            for s in timepoints
        ]
        return LayeredEncoding(
            _layers=[(l, {str(v):v for v in l.get_free_variables()}) for l in layers]
        )

    def _encode_query_ge(self, query, num_steps, step_size):
        query_variable_name = self._query_variable_name(query)
        timepoints = range(0, (step_size * num_steps) + 1, step_size)
        layers = [
            GE(
                self._encode_state_var(var=query.variable, time=s),
                Real(query.lb),
            )
            for s in timepoints
        ]
        return LayeredEncoding(
            _layers=[(l, {str(v): v for v in l.get_free_variables()}) for l in layers]
        )

    def _encode_query_true(self, query, num_steps, step_size):
        timepoints = range(0, (step_size * num_steps) + 1, step_size)
        return LayeredEncoding(_layers=[(TRUE(), {}) for t in timepoints])

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

        vars = self._symbols(model_encoding.symbols())
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
