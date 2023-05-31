from typing import List

from pysmt.formula import FNode
from pysmt.shortcuts import (
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Plus,
    Real,
    Symbol,
    Times,
)

from funman.model.model import Model

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
        self, model: "Model", step: int, next_step: int
    ) -> FNode:
        state_vars = model._state_vars()
        transitions = model._transitions()
        step_size = next_step - step
        current_state = [
            self._encode_state_var(s["sname"], time=step) for s in state_vars
        ]
        next_state = [
            self._encode_state_var(s["sname"], time=next_step)
            for s in state_vars
        ]

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        transition_terms = [
            self._encode_transition_term(
                i,
                t,
                current_state,
                next_state,
                model._input_edges(),
                model._output_edges(),
            )
            for i, t in enumerate(transitions)
        ]

        # for each var, next state is the net flow for the var: sum(inflow) - sum(outflow)
        net_flows = []
        for v_index, var in enumerate(state_vars):
            state_var_flows = []
            for t_index, transition in enumerate(transitions):
                outflow = model._num_flow_from_state_to_transition(
                    v_index + 1, t_index + 1
                )
                inflow = model._flow_into_state_via_transition(
                    v_index + 1, t_index + 1
                )
                net_flow = inflow - outflow

                if net_flow != 0:
                    state_var_flows.append(
                        Times(
                            Real(net_flow) * transition_terms[t_index]
                        ).simplify()
                    )
            if len(state_var_flows) > 0:
                flows = Plus(
                    Times(Real(step_size), Plus(state_var_flows)).simplify(),
                    current_state[v_index],
                ).simplify()
            else:
                flows = current_state[v_index]

            net_flows.append(Equals(next_state[v_index], flows))

        return And(net_flows)

    def _encode_transition_term(
        self,
        t_index,
        transition,
        current_state,
        next_state,
        input_edges,
        output_edges,
    ):
        ins = [
            current_state[edge["is"] - 1]
            for edge in input_edges
            if edge["it"] == t_index + 1
        ]
        param_symbol = self._encode_state_var(
            transition["tprop"]["parameter_name"]
        )

        return Times([param_symbol] + ins)

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
        state_vars = model._state_vars()
        return [s["sname"] for s in state_vars]
