from typing import List

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
from funman.utils.sympy_utils import rate_expr_to_pysmt, sympy_to_pysmt

from .translate import Encoder, Encoding


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
        self, model: "Model", step: int, next_step: int, substitutions={}
    ) -> FNode:
        state_vars = model._state_vars()
        transitions = model._transitions()
        step_size = next_step - step
        current_state = {
            model._state_var_id(s): self._encode_state_var(
                model._state_var_name(s), time=step
            )
            for s in state_vars
        }
        next_state = {
            model._state_var_id(s): self._encode_state_var(
                model._state_var_name(s), time=next_step
            )
            for s in state_vars
        }

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        transition_terms = {
            model._transition_id(t): self._encode_transition_term(
                t,
                current_state,
                next_state,
                model,
                substitutions=substitutions,
            )
            for t in transitions
        }

        # for each var, next state is the net flow for the var: sum(inflow) - sum(outflow)
        net_flows = []
        for var in state_vars:
            state_var_flows = []
            for transition in transitions:
                state_var_id = model._state_var_id(var)

                transition_id = model._transition_id(transition)
                outflow = model._num_flow_from_state_to_transition(
                    state_var_id, transition_id
                )
                inflow = model._flow_into_state_via_transition(
                    state_var_id, transition_id
                )
                net_flow = inflow - outflow

                if net_flow != 0:
                    state_var_flows.append(
                        Times(
                            Real(net_flow) * transition_terms[transition_id]
                        ).substitute(substitutions)
                    )
            if len(state_var_flows) > 0:
                flows = Plus(
                    Times(
                        Real(step_size),
                        Plus(state_var_flows).substitute(substitutions),
                    ).simplify(),
                    current_state[state_var_id].substitute(substitutions),
                ).simplify()
            else:
                flows = current_state[state_var_id].substitute(substitutions)

            net_flows.append(Equals(next_state[state_var_id], flows))
            substitutions[next_state[state_var_id]] = flows

        compartmental_bounds = self._encode_compartmental_bounds(
            model, next_step
        )

        return And(net_flows + [compartmental_bounds]), substitutions

    def _define_init(self, model: Model, init_time: int = 0) -> FNode:
        state_var_names = model._state_var_names()
        compartmental_bounds = self._encode_compartmental_bounds(model, 0)
        return And(
            And(
                [
                    self._define_init_term(model, var, init_time)
                    for var in state_var_names
                ]
            ),
            compartmental_bounds,
        )

    def _encode_compartmental_bounds(self, model: "Model", step):
        bounds = []
        for var in model._state_vars():
            lb = GE(
                self._encode_state_var(model._state_var_name(var), time=step),
                Real(0.0),
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
                ),
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
            rate_expr_to_pysmt(r.expression)
            for r in model._transition_rate(transition)
        ]
        # Before calling substitute, need to replace the formula_manager
        get_env()._substituter.mgr = get_env().formula_manager

        return (
            Or([Times([tr] + ins) for tr in transition_rates])
            .substitute(substitutions)
            .simplify()
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
        return state_vars
