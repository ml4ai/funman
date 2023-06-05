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
        self, model: "Model", step: int, next_step: int, substitutions={}
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
                substitutions=substitutions,
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
                    state_var_flow = Times(
                        Real(net_flow) * transition_terms[t_index]
                    ).substitute(substitutions)
                    state_var_flow = state_var_flow.simplify()
                    state_var_flows.append(state_var_flow)
            if len(state_var_flows) > 0:
                combined_flows = Plus(state_var_flows).substitute(substitutions)
                combined_flows = combined_flows.simplify()
                flows = Plus(
                    Times(Real(step_size), combined_flows).simplify(),
                    current_state[v_index].substitute(substitutions),
                ).simplify()
            else:
                flows = (
                    current_state[v_index].substitute(substitutions).simplify()
                )

            substitutions[next_state[v_index]] = flows
            net_flows.append(Equals(next_state[v_index], flows))

        return And(net_flows), substitutions

    # (= Susceptible_2 (- 
    #                   (+ c1 (+ 
    #                          (* (* (+ (* beta c2) c3) c4) (* (- c5 (* beta c6)) c7)) 
    #                          (* (* beta c8) (* (- c9 (* beta c10)) c11)))) 
    #                   (+ (* beta c12)  
    #                    (+ 
    #                     (* (- c13 (* beta c14)) c15) --> (- (* c15 c13) (* beta c14 c15)) distribute mult. 
    #                    
    #                     (* (- c16 (* beta c17)) c18)))))
    #                    --> (+ (* beta c12) (- (* c15 c13) (* beta c14 c15)) (- (* c16 c18) (* beta c17 c18)))
    #                    --> (+ (* beta (+ c12 (* -c14 c15) (* -c17 c18))) (* -c15 c13) (* -c16 c18)))
    #                    --> (+ (* beta d1) d2))

    def _encode_transition_term(
        self,
        t_index,
        transition,
        current_state,
        next_state,
        input_edges,
        output_edges,
        substitutions={},
    ):
        ins = [
            current_state[edge["is"] - 1]
            for edge in input_edges
            if edge["it"] == t_index + 1
        ]
        param_symbol = self._encode_state_var(
            transition["tprop"]["parameter_name"]
        )

        return Times([param_symbol] + ins).substitute(substitutions).simplify()

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
