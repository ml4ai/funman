import logging
from typing import Dict, List, Union, Set

import sympy
from pysmt.formula import FNode
from pysmt.shortcuts import (
    GE,
    LE,
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Or,
    Plus,
    Real,
    Symbol,
    Times,
)

from funman.model.model import Model
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils.sympy_utils import (
    rate_expr_to_pysmt,
    series_approx,
    sympy_subs,
    to_sympy,
)

from .translate import Encoder, Encoding

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


class PetrinetEncoder(Encoder):
    _transition_rate_cache: Dict[str, List[sympy.Expr]] = {}

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
        return Encoding(formula=TRUE(), symbols={})

    def _encode_next_step(
        self,
        scenario: "AnalysisScenario",
        step: int,
        next_step: int,
        substitutions={},
    ) -> FNode:
        l.debug(f"Encoding step: {step} to {next_step}")
        state_vars = scenario.model._state_vars()
        transitions = scenario.model._transitions()

        step_size = next_step - step
        current_state = {
            scenario.model._state_var_id(s): self._encode_state_var(
                scenario.model._state_var_name(s), time=step
            )
            for s in state_vars
        }
        next_state = {
            scenario.model._state_var_id(s): self._encode_state_var(
                scenario.model._state_var_name(s), time=next_step
            )
            for s in state_vars
        }

        time_var = scenario.model._time_var()
        if time_var:
            time_var_name = scenario.model._time_var_id(time_var)
            time_symbol = self._encode_state_var(
                time_var_name
            )  # Needed so that there is a pysmt symbol for 't'
            current_time_var = self._encode_state_var(time_var_name, time=step)
            next_time_var = self._encode_state_var(
                time_var_name, time=next_step
            )
            current_state[time_var_name] = current_time_var
            next_state[time_var_name] = next_time_var

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        transition_terms = {
            scenario.model._transition_id(t): self._encode_transition_term(
                t,
                current_state,
                next_state,
                scenario,
                substitutions=substitutions,
            )
            for t in transitions
        }

        if self.config.substitute_subformulas:
            if all(
                isinstance(t, sympy.Expr)
                for k, v in transition_terms.items()
                for t in v
            ):
                # substitutions are FNodes
                # transition terms are sympy.Expr
                # convert relevant substitutions to sympy.Expr
                # sympy subs transition term with converted subs
                # simplify/approximate substituted formula
                # convert to pysmt formula
                # TODO store substitutions as both FNode and pysmt.Expr to avoid extra conversion
                transition_terms = {
                    k: [
                        FUNMANSimplifier.sympy_simplify(
                            t,
                            parameters=scenario.model_parameters(),
                            substitutions=substitutions,
                            threshold=self.config.series_approximation_threshold,
                            taylor_series_order=self.config.taylor_series_order,
                        )
                        for t in v
                    ]
                    for k, v in transition_terms.items()
                }
            else:
                transition_terms = {
                    k: v.substitute(substitutions)
                    for k, v in transition_terms.items()
                }
        else:
            # Need to convert transition terms to pysmt without substituting
            transition_terms = {
                k: [rate_expr_to_pysmt(t, current_state) for t in v]
                for k, v in transition_terms.items()
            }

        # for each var, next state is the net flow for the var: sum(inflow) - sum(outflow)
        net_flows = []
        for var in state_vars:
            state_var_flows = []
            for transition in transitions:
                state_var_id = scenario.model._state_var_id(var)

                transition_id = scenario.model._transition_id(transition)
                outflow = scenario.model._num_flow_from_state_to_transition(
                    state_var_id, transition_id
                )
                inflow = scenario.model._flow_into_state_via_transition(
                    state_var_id, transition_id
                )
                net_flow = inflow - outflow

                if net_flow != 0:
                    state_var_flows.append(
                        [
                            Times(Real(net_flow) * t)
                            for t in transition_terms[transition_id]
                        ]
                    )
            if len(state_var_flows) > 0:
                # FIXME: the below should involve computing update as the cross product of all transition_rate equations
                flows = Plus(
                    Times(
                        Real(step_size),
                        Plus(
                            [s[0] for s in state_var_flows]
                        ),  # FIXME see above
                    ),  # .simplify()
                    current_state[state_var_id],
                )  # .simplify()
                if self.config.substitute_subformulas:
                    # flows = flows.substitute(substitutions)
                    flows = FUNMANSimplifier.sympy_simplify(
                        # flows.substitute(substitutions),
                        to_sympy(
                            flows.substitute(substitutions).simplify(),
                            scenario.model._symbols(),
                        ),
                        parameters=scenario.model_parameters(),
                        threshold=self.config.series_approximation_threshold,
                        taylor_series_order=self.config.taylor_series_order,
                    )
            else:
                flows = current_state[state_var_id]
                # .substitute(substitutions)

            net_flows.append(Equals(next_state[state_var_id], flows))
            if self.config.substitute_subformulas:
                substitutions[next_state[state_var_id]] = flows

        if self.config.use_compartmental_constraints:
            compartmental_bounds = self._encode_compartmental_bounds(
                scenario, next_step, substitutions=substitutions
            ).simplify()
        else:
            compartmental_bounds = TRUE()

        # If any variables depend upon time, then time updates need to be encoded.
        if time_var is not None:
            time_increment = (
                Plus(current_time_var, Real(step_size))
                .substitute(substitutions)
                .simplify()
            )
            time_update = Equals(next_time_var, time_increment)
            if self.config.substitute_subformulas:
                substitutions[next_time_var] = time_increment
        else:
            time_update = TRUE()

        return (
            And(net_flows + [compartmental_bounds, time_update]),
            substitutions,
        )

    def _define_init(
        self, scenario: "AnalysisScenario", init_time: int = 0
    ) -> FNode:
        initial_state, substitutions = super()._define_init(
            scenario, init_time=init_time
        )
        # state_var_names = scenario.model._state_var_names()
        # initial_substitution = {}

        if self.config.use_compartmental_constraints:
            compartmental_bounds = self._encode_compartmental_bounds(
                scenario, 0
            )
            if self.config.substitute_subformulas and substitutions:
                compartmental_bounds = compartmental_bounds.substitute(
                    substitutions
                ).simplify()
        else:
            compartmental_bounds = TRUE()
        initial_state = And(initial_state, compartmental_bounds).simplify()

        return initial_state, substitutions

    def _encode_compartmental_bounds(
        self,
        scenario: "AnalysisScenario",
        step,
        substitutions: Dict[FNode, FNode] = {},
    ):
        bounds = []
        for var in scenario.model._state_vars():
            lb = (
                GE(
                    self._encode_state_var(
                        scenario.model._state_var_name(var), time=step
                    ),
                    Real(0.0),
                )
                # .substitute(substitutions)
                # .simplify()
            )
            population = (
                Real(1.0)
                if self.config.normalize
                else Real(scenario.normalization_constant)
            )
            ub = LE(
                self._encode_state_var(
                    scenario.model._state_var_name(var), time=step
                ),
                population
                # Plus(
                #     [
                #         self._encode_state_var(
                #             model._state_var_name(var1), time=step
                #         )
                #         for var1 in model._state_vars()
                #     ]
                # )
                # .substitute(substitutions)
                # .simplify(),
            )

            bounds += [lb, ub]
        # noise_var = Symbol("noise", REAL)
        noise_const = Real(1e-3)
        sum_vars = Plus(
            [
                self._encode_state_var(
                    scenario.model._state_var_name(var), time=step
                )
                for var in scenario.model._state_vars()
            ]
        )
        total = And(
            LE(sum_vars, Plus(population, noise_const)),
            LE(Minus(population, noise_const), sum_vars),
        )

        return And(bounds + [total])

    def _encode_transition_term(
        self, transition, current_state, next_state, scenario, substitutions={}
    ) -> Union[sympy.Expr, FNode]:
        transition_id = scenario.model._transition_id(transition)
        input_edges = scenario.model._input_edges()
        output_edges = scenario.model._output_edges()
        state_subs = {s: str(f) for s, f in current_state.items()}

        ins = [
            current_state[scenario.model._edge_source(edge)]
            for edge in input_edges
            if scenario.model._edge_target(edge) == transition_id
        ]
        # The model expresses each rate with untimed variable symbols.
        # If not yet approximated, approximate and cache term.
        # Substitute current time variable symbols
        if (
            scenario.model._transition_id(transition)
            not in self._transition_rate_cache
        ):
            model_transition_rates = scenario.model._transition_rate(
                transition
            )
            if all(
                isinstance(r, str) or isinstance(r, float)
                for r in model_transition_rates
            ):
                self._transition_rate_cache[
                    scenario.model._transition_id(transition)
                ] = model_transition_rates
            elif all(
                isinstance(r, sympy.Expr) for r in model_transition_rates
            ):
                self._transition_rate_cache[
                    scenario.model._transition_id(transition)
                ] = [
                    series_approx(
                        (
                            r
                            if isinstance(r, sympy.Expr)
                            else to_sympy(r, scenario.model._symbols())
                        ),
                        vars=[
                            mp.name
                            for mp in scenario.model_parameters()
                            if mp in scenario.synthesized_parameters()
                        ],
                    )
                    for r in scenario.model._transition_rate(transition)
                ]
            else:
                raise Exception(
                    f"Cannot encode model transition rate: {model_transition_rates}"
                )

        transition_rates = []
        for r in self._transition_rate_cache[
            scenario.model._transition_id(transition)
        ]:
            if isinstance(r, sympy.Expr):
                # is a custom rate expression
                transition_rates.append(sympy_subs(r, state_subs))
            elif isinstance(r, str):
                # Is a single parameter
                transition_rates.append(substitutions[Symbol(r, REAL)])
            elif isinstance(r, float):
                # Is a constant
                transition_rates.append(Real(r))

        if all(isinstance(t, sympy.Expr) for t in transition_rates):
            return transition_rates  # Need to build Or(transition_rates) later after converting to FNodes
        else:
            return (
                Or([(Times([tr] + ins)) for tr in transition_rates])
                # .substitute(substitutions)
                # .simplify()
            )

    def _get_timed_symbols(self, model: Model) -> Set[str]:
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
        state_vars = set(model._state_var_names())
        time_var = model._time_var()
        if time_var:
            state_vars.add(f"timer_{time_var.id}")
        return state_vars
