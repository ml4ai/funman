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
        current_state = {
            model._state_var_id(s): self._encode_state_var(model._state_var_name(s), time=step) for s in state_vars
        }
        next_state = {
            model._state_var_id(s): self._encode_state_var(model._state_var_name(s), time=next_step)
            for s in state_vars
        }

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        transition_terms = {
            model._transition_id(t):
            self._encode_transition_term(
                t,
                current_state,
                next_state,
                model
            )
            for t in transitions
        }

        # for each var, next state is the net flow for the var: sum(inflow) - sum(outflow)
        net_flows = []
        for var in state_vars:
            state_var_flows = []
            for transition in transitions:
                state_var_id = model._state_var_id(var)

                transition_id =  model._transition_id(transition)
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
                        ).simplify()
                    )
            if len(state_var_flows) > 0:
                flows = Plus(
                    Times(Real(step_size), Plus(state_var_flows)).simplify(),
                    current_state[state_var_id],
                ).simplify()
            else:
                flows = current_state[state_var_id]

            net_flows.append(Equals(next_state[state_var_id], flows))

        return And(net_flows)

    def _encode_transition_term(
        self,
        
        transition,
        current_state,
        next_state,
        model
    ):
        transition_id = model._transition_id(transition)
        input_edges = model._input_edges()
        output_edges= model._output_edges()
        ins = [
            current_state[model._edge_source(edge)]
            for edge in input_edges
            if model._edge_target(edge) == transition_id
        ]
        param_symbol = self._encode_state_var(
            model._transition_parameter(transition)
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
        state_vars = model._state_var_names()
        return state_vars
