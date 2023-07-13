from typing import Dict, List

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
    get_env,
)

from funman.model.model import Model
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils.sympy_utils import rate_expr_to_pysmt, sympy_to_pysmt

from .translate import Encoder, Encoding

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


class PetrinetEncoder(Encoder):
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
        time_var = scenario.model._time_var()
        time_var_name = scenario.model._time_var_id(time_var)
        time_symbol = self._encode_state_var(
            time_var_name
        )  # Needed so that there is a pysmt symbol for 't'
        current_time_var = self._encode_state_var(time_var_name, time=step)
        next_time_var = self._encode_state_var(time_var_name, time=next_step)

        step_size = next_step - step
        current_state = {
            scenario.model._state_var_id(s): self._encode_state_var(
                scenario.model._state_var_name(s), time=step
            )
            for s in state_vars
        }
        current_state[time_var_name] = current_time_var

        next_state = {
            scenario.model._state_var_id(s): self._encode_state_var(
                scenario.model._state_var_name(s), time=next_step
            )
            for s in state_vars
        }
        next_state[time_var_name] = next_time_var

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        transition_terms = {
            scenario.model._transition_id(t): self._encode_transition_term(
                t,
                current_state,
                next_state,
                scenario.model,
                substitutions=substitutions,
            )
            for t in transitions
        }

        if self.config.substitute_subformulas:
            transition_terms = {
                k: v.substitute(substitutions)
                for k, v in transition_terms.items()
            }
            transition_terms = {
                k: FUNMANSimplifier.sympy_simplify(v,
                    parameters=scenario.model_parameters(),
                )
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
                        Times(Real(net_flow) * transition_terms[transition_id])
                    )
            if len(state_var_flows) > 0:
                flows = Plus(
                    Times(
                        Real(step_size),
                        Plus(state_var_flows),
                    ),  # .simplify()
                    current_state[state_var_id],
                )  # .simplify()
                if self.config.substitute_subformulas:
                    flows = flows.substitute(substitutions)
                    # flows = FUNMANSimplifier.sympy_simplify(
                    #     flows.substitute(substitutions),
                    #     parameters=scenario.model_parameters(),
                    # )
            else:
                flows = current_state[state_var_id]
                # .substitute(substitutions)

            net_flows.append(Equals(next_state[state_var_id], flows))
            if self.config.substitute_subformulas:
                substitutions[next_state[state_var_id]] = flows

        if self.config.use_compartmental_constraints:
            compartmental_bounds = self._encode_compartmental_bounds(
                scenario.model, next_step, substitutions=substitutions
            )
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

    def _define_init(self, model: Model, init_time: int = 0) -> FNode:
        state_var_names = model._state_var_names()

        if self.config.use_compartmental_constraints:
            compartmental_bounds = self._encode_compartmental_bounds(model, 0)
        else:
            compartmental_bounds = TRUE()

        time_var = model._time_var()

        if time_var:
            time_var_name = model._time_var_id(time_var)
            time_symbol = self._encode_state_var(
                time_var_name, time=0
            )  # Needed so that there is a pysmt symbol for 't'

            time_var_init = Equals(time_symbol, Real(0.0))
        else:
            time_var_init = TRUE()

        return And(
            And(
                [
                    self._define_init_term(model, var, init_time)
                    for var in state_var_names
                ]
            ),
            compartmental_bounds,
            time_var_init,
        )

    def _encode_compartmental_bounds(
        self, model: "Model", step, substitutions: Dict[FNode, FNode] = {}
    ):
        bounds = []
        for var in model._state_vars():
            lb = (
                GE(
                    self._encode_state_var(
                        model._state_var_name(var), time=step
                    ),
                    Real(0.0),
                )
                .substitute(substitutions)
                .simplify()
            )
            ub = LE(
                self._encode_state_var(model._state_var_name(var), time=step),
                Plus(
                    [
                        self._encode_state_var(
                            model._state_var_name(var1), time=step
                        )
                        for var1 in model._state_vars()
                    ]
                )
                .substitute(substitutions)
                .simplify(),
            )
            bounds += [lb, ub]

        return And(bounds)

    def _encode_transition_term(
        self, transition, current_state, next_state, model, substitutions={}
    ):
        transition_id = model._transition_id(transition)
        input_edges = model._input_edges()
        output_edges = model._output_edges()
        ins = [
            current_state[model._edge_source(edge)]
            for edge in input_edges
            if model._edge_target(edge) == transition_id
        ]
        transition_rates = [
            rate_expr_to_pysmt(r, current_state)
            for r in model._transition_rate(transition)
        ]

        return (
            Or(
                [
                    (
                        tr
                        if len(ins) == 0
                        or len(tr.args()) > 1  # if tr is complete expression
                        else Times([tr] + ins)  # else build expression
                    )
                    for tr in transition_rates
                ]
            )
            # .substitute(substitutions)
            # .simplify()
        )

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
        state_vars = model._state_var_names()
        time_var = model._time_var()
        state_vars.append(f"timer_{time_var.id}")
        return state_vars
