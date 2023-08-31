"""
This module defines the abstract base classes for the model encoder classes in funman.translate package.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

from pydantic import ConfigDict, BaseModel
from pysmt.constants import Numeral
from pysmt.formula import FNode
from pysmt.shortcuts import (
    GE,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Div,
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
from funman.representation import ModelParameter
from funman.representation.representation import (
    Box,
    Interval,
    ModelParameter,
    Point,
)
from funman.representation.symbol import ModelSymbol
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils.sympy_utils import (
    FUNMANFormulaManager,
    sympy_to_pysmt,
    to_sympy,
)

l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)


class Encoding(BaseModel):
    _substitutions: Dict[FNode, FNode] = {}


class FlatEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

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


class Encoder:
    pass


class LayeredEncoding(BaseModel):
    """
    An encoding comprises a formula over a set of symbols.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    step_size: int
    _layers: List[
        Tuple[FNode, Union[List[FNode], Dict[str, Dict[str, FNode]]]]
    ] = []
    _encoder: Encoder

    # @validator("formula")
    # def set_symbols(cls, v: FNode):
    #     cls.symbols = Symbol(v, REAL)

    def encoding(
        self,
        encoding_fn,
        layers=None,
        box: Box = None,
        assumptions: List[FNode] = None,
    ):
        if layers:
            # return And([self._layers[i][0] for i in layers])
            return And(
                [
                    self._get_or_create_layer(
                        encoding_fn, i, box=box, assumptions=assumptions
                    )[0]
                    for i in layers
                ]
            )
        else:
            return And(
                [
                    self._get_or_create_layer(
                        encoding_fn, i, box=box, assumptions=assumptions
                    )[0]
                    for i, l in enumerate(self._layers)
                ]
            )

    def _get_or_create_layer(
        self,
        encoding_fn,
        layer_idx: int,
        box: Box = None,
        assumptions: List[FNode] = None,
    ):
        if self._layers[layer_idx] is None:
            layer = encoding_fn(layer_idx, step_size=self.step_size)
            if assumptions:
                layer = (Iff(And(assumptions[layer_idx]), layer[0]), layer[1])
            self._layers[layer_idx] = layer
        return self._layers[layer_idx]

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
        return {k: v for layer in self._layers for k, v in layer[1].items()}


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    config: FUNMANConfig
    _timed_model_elements: Dict = None
    _min_time_point: int
    _min_step_size: int
    _untimed_symbols: Set[str] = set([])
    _timed_symbols: Set[str] = set([])
    _untimed_constraints: FNode
    _scenario: "AnalysisScenario"
    # _assignments: Dict[str, float] = {}
    _env = get_env()
    _env._simplifier = FUNMANSimplifier(_env)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scenario = kwargs["scenario"]

        env = get_env()
        if not isinstance(env._formula_manager, FUNMANFormulaManager):
            env._formula_manager = FUNMANFormulaManager(env._formula_manager)
            # Before calling substitute, need to replace the formula_manager
            env._substituter.mgr = env.formula_manager

        # Need to initialize pysmt symbols for parameters to help with parsing custom rate equations
        variables = [
            p.name for p in self._scenario.model._parameters()
        ] + self._scenario.model._state_var_names()
        variable_symbols = [self._encode_state_var(p) for p in variables]

        self._encode_timed_model_elements(self._scenario)

    def can_encode():
        """
        Return boolean indicating if the scenario can be encoded with the FUNMANConfig
        """
        return True

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

    def initialize_encodings(self, scenario, num_steps, step_size_idx):
        # state_timepoints, transition_timepoints = self._get_timepoints(
        #     num_steps, step_size
        # )
        # self.state_timepoints = state_timepoints
        # self.transition_timepoints = transition_timepoints

        if self._timed_model_elements:
            step_size = self._timed_model_elements["step_sizes"][step_size_idx]
        else:
            step_size = 1

        model_encoding = LayeredEncoding(
            step_size=step_size,
        )
        model_encoding._layers = [None] * (num_steps + 1)
        model_encoding._encoder = self
        query_encoding = LayeredEncoding(
            step_size=step_size,
        )
        query_encoding._layers = [None] * (num_steps + 1)
        query_encoding._encoder = self
        return model_encoding, query_encoding

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

    def encode_simplified(self, model, query, step_size_idx: int):
        formula = And(model, query)
        if self.config.substitute_subformulas:
            sub_formula = formula.substitute(
                self._timed_model_elements["time_step_substitutions"][
                    step_size_idx
                ]
            ).simplify()
            return sub_formula
        else:
            return formula

    def _encode_next_step(
        self, model: Model, step: int, next_step: int, substitutions={}
    ) -> FNode:
        pass

    def encode_model_layer(self, layer_idx: int, step_size: int = None):
        if layer_idx == 0:
            return self.encode_init_layer()
        else:
            return self.encode_transition_layer(layer_idx, step_size=step_size)

    def encode_init_layer(self):
        initial_state = self._timed_model_elements["init"]
        initial_symbols = initial_state.get_free_variables()

        return (initial_state, {str(s): s for s in initial_symbols})

    def encode_transition_layer(self, layer_idx: int, step_size: int = None):
        c = self._timed_model_elements["time_step_constraints"][layer_idx - 1][
            step_size - self._min_step_size
        ]
        step_size_idx = self._timed_model_elements["step_sizes"].index(
            step_size
        )

        substitutions = self._timed_model_elements["time_step_substitutions"][
            step_size_idx
        ]

        if c is None:
            timepoint = self._timed_model_elements["state_timepoints"][
                step_size_idx
            ][layer_idx - 1]
            next_timepoint = self._timed_model_elements["state_timepoints"][
                step_size_idx
            ][layer_idx]
            c, substitutions = self._encode_next_step(
                self._scenario,
                timepoint,
                next_timepoint,
                substitutions=substitutions,
            )
            self._timed_model_elements["time_step_constraints"][layer_idx - 1][
                step_size - self._min_step_size
            ] = c
            self._timed_model_elements["time_step_substitutions"][
                step_size_idx
            ] = substitutions
        return (c, {str(s): s for s in c.get_free_variables()})

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
        init_layer = self.encode_init_layer()
        layers = [init_layer]

        for i in range(num_steps + 1):
            layer = self.encode_transition_layer(i + 1, step_size=step_size)
            layers.append(layer)

        encoding = LayeredEncoding(
            substitutions=self.substitutions,
        )
        encoding._layers = layers
        return encoding

    def _initialize_substitutions(
        self, scenario: "AnalysisScenario", normalization=True
    ):
        # Add parameter assignments
        parameters = scenario.model_parameters()

        # Normalize if constant value
        parameter_assignments = {
            self._encode_state_var(k.name): Real(float(k.lb))
            for k in parameters
            if k.lb == k.ub
        }

        return parameter_assignments

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

    def _define_init_term(
        self, scenario: "AnalysisScenario", var: str, init_time: int, substitutions=None
    ):
        value = scenario.model._get_init_value(var, scenario)

        init_term = None
        substitution = ()

        if isinstance(value, FNode):
            value_expr = to_sympy(value, scenario.model._symbols())
            if self.config.substitute_subformulas and substitutions:
                value_expr = sympy_to_pysmt(value_expr).substitute(substitutions).simplify()
                # value_expr = FUNMANSimplifier.sympy_simplify(
                #             value_expr,
                #             parameters=scenario.model_parameters(),
                #             substitutions=substitutions,
                #             threshold=self.config.series_approximation_threshold,
                #             taylor_series_order=self.config.taylor_series_order,
                #         )
            else: 
                value_expr = sympy_to_pysmt(value_expr)

            substitution = (
                self._encode_state_var(var, time=init_time),
                value_expr,
            )
            init_term = Equals(*substitution)
            return init_term, substitution
        elif isinstance(value, list):
            substitution = None
            init_term = And(
                GE(
                    self._encode_state_var(var, time=init_time),
                    Real(value[0]),
                ),
                LT(
                    self._encode_state_var(var, time=init_time),
                    Real(value[1]),
                ),
            )
            return init_term, substitution
        else:
            return TRUE(), None

    def _define_init(
        self, scenario: "AnalysisScenario", init_time: int = 0
    ) -> FNode:
        # Generate Parameter symbols and assignments
        substitutions = self._initialize_substitutions(scenario)
        initial_state = And([Equals(k, v) for k, v in substitutions.items()])


        state_var_names = scenario.model._state_var_names()

        time_var = scenario.model._time_var()
        if time_var is not None:
            time_var_name = scenario.model._time_var_id(time_var)
            time_symbol = self._encode_state_var(
                time_var_name, time=0
            )  # Needed so that there is a pysmt symbol for 't'

            substitutions[time_symbol] = Real(0.0)
            time_var_init = Equals(time_symbol, Real(0.0))
        else:
            time_var_init = TRUE()

        initial_state_vars_and_subs = [
            self._define_init_term(
                scenario, var, init_time, substitutions=substitutions
            )
            for var in state_var_names
        ]

        substitutions = {
            **substitutions,
            **{
                sv[1][0]: sv[1][1]
                for sv in initial_state_vars_and_subs
                if sv[1]
            },
        }
        initial_state = And(
            And([sv[0] for sv in initial_state_vars_and_subs]),
            time_var_init,
            initial_state,
        )

        substitutions = self._propagate_substitutions(substitutions)

        return initial_state, substitutions

    def _propagate_substitutions(self, substitutions):
        change = True
        while change:
            next_subs = {}
            for var in substitutions:
                next_subs[var] = (
                    substitutions[var].substitute(substitutions).simplify()
                )
                change = change or next_subs[var] != substitutions[var]
            substitutions = next_subs
            change = False
        return substitutions

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
        step_sizes = scenario.structure_parameter("step_size")
        num_steps = scenario.structure_parameter("num_steps")

        state_timepoints = []
        transition_timepoints = []
        initial_state, initial_substitutions = self._define_init(scenario)
        for i, step_size in enumerate(
            range(int(step_sizes.lb), int(step_sizes.ub) + 1)
        ):
            s_timepoints, t_timepoints = self._get_timepoints(
                num_steps.ub, step_size
            )
            state_timepoints.append(s_timepoints)
            transition_timepoints.append(transition_timepoints)

        (
            configurations,
            max_step_index,
            max_step_size,
        ) = self._get_structural_configurations(scenario)
        self._timed_model_elements = {
            "step_sizes": list(
                range(int(step_sizes.lb), int(step_sizes.ub + 1))
            ),
            "state_timepoints": state_timepoints,
            "transition_timepoints": transition_timepoints,
            "init": initial_state,
            "time_step_constraints": [
                [None for i in range(max_step_size)]
                for j in range(max_step_index)
            ],
            "time_step_substitutions": [
                initial_substitutions.copy() for i in range(max_step_size)
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
            int(step_size) * int(num_steps) + 1,
            int(step_size),
        )

        if len(list(state_timepoints)) == 0:
            raise Exception(
                f"Could not identify timepoints from step_size = {step_size} and num_steps = {num_steps}"
            )

        transition_timepoints = range(
            0, int(step_size) * int(num_steps), int(step_size)
        )
        return list(state_timepoints), list(transition_timepoints)

    def encode_query_layer(
        self,
        query: Query,
        layer_idx: int,
        step_size: int = None,
        normalize=True,
    ):
        """
        Encode a query into an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode

        Returns
        -------

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
            layer = query_handlers[type(query)](
                query, layer_idx, step_size, normalize=normalize
            )
            return layer
            # encoded_query.substitute(substitutions)
            # encoded_query.simplify()
            # return encoded_query
        else:
            raise NotImplementedError(
                f"Do not know how to encode query of type {type(query)}"
            )

    def _return_encoded_query(self, query, layer_idx, step_size, normalize=True):
        return (
            query._formula,
            {str(v): v for v in query._formula.get_free_variables()},
        )

    def _query_variable_name(self, query):
        return (
            query.variable
            if not isinstance(query.variable, ModelSymbol)
            else str(query.variable)
        )

    def _encode_query_and(self, query, layer_idx, step_size, normalize=True):
        queries = [
            self.encode_query_layer(
                q, layer_idx, step_size, normalize=normalize
            )
            for q in query.queries
        ]

        layer = (
            And([q[0] for q in queries]),
            {str(s): s for q in queries for s in q[1]},
        )

        return layer

    def _normalize(self, value):
        return sympy_to_pysmt(
            to_sympy(
                Div(value, Real(self._scenario.normalization_constant)),
                self._scenario.model._symbols(),
            )
        )

    def _encode_query_le(self, query, layer_idx, step_size, normalize=True):
        step_size_idx = self._timed_model_elements["step_sizes"].index(
            step_size
        )
        time = self._timed_model_elements["state_timepoints"][step_size_idx][
            layer_idx
        ]
        if normalize:
            ub = self._normalize(Real(query.ub))
        else:
            ub = Real(query.ub)
        q = LE(
            self._encode_state_var(var=query.variable, time=time),
            ub,
        )

        return (q, {str(v): v for v in q.get_free_variables()})

    def _encode_query_ge(self, query, layer_idx, step_size, normalize=True):
        step_size_idx = self._timed_model_elements["step_sizes"].index(
            step_size
        )
        time = self._timed_model_elements["state_timepoints"][step_size_idx][
            layer_idx
        ]
        if normalize:
            lb = self._normalize(query.lb)
        else:
            lb = Real(query.lb)
        q = GE(
            self._encode_state_var(var=query.variable, time=time),
            lb,
        )
        return (q, {str(v): v for v in q.get_free_variables()})

    def _encode_query_true(self, query, layer_idx, step_size, normalize=True):
        return (TRUE(), {})

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
        self, p: str, i: Interval, closed_upper_bound: bool = False, infinity_constraints=False
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
                if i.lb != NEG_INFINITY or infinity_constraints
                else TRUE()
            )
            upper_ineq = LE if closed_upper_bound else LT
            upper = (
                upper_ineq(Symbol(p, REAL), Real(i.ub))
                if i.ub != POS_INFINITY or infinity_constraints
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

    def box_to_smt(self, box: Box, closed_upper_bound: bool = False, infinity_constraints = False):
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
                    p, interval, closed_upper_bound=closed_upper_bound, infinity_constraints=infinity_constraints
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
