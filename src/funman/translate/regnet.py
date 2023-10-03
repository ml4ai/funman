from typing import List

from pysmt.formula import FNode
from pysmt.shortcuts import (
    REAL,
    TRUE,
    And,
    Equals,
    Plus,
    Real,
    Symbol,
    Times,
)

from funman.model.model import Model

from .translate import Encoder, Encoding


class RegnetEncoder(Encoder):
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
        state_var_names = model._state_var_names()
        transitions = model._transitions()
        parameters = model._parameters()
        state_vars = model._state_vars()
        step_size = next_step - step
        current_state = {
            s: self._encode_state_var(s, time=step) for s in state_var_names
        }
        next_state = {
            s: self._encode_state_var(s, time=next_step)
            for s in state_var_names
        }

        # Each transition corresponds to a term that is the product of current state vars and a parameter
        edge_terms = [
            self._encode_transition_term(model, t, current_state)
            for t in transitions
        ]

        self_terms = [
            self._encode_self_term(model, s, current_state) for s in state_vars
        ]
        transition_terms = edge_terms + self_terms

        # for each var, next state is the net flow for the var: sum(inflow) - sum(outflow)
        net_flows = []
        for var in state_var_names:
            state_var_flows = []
            for transition in [
                trans for (tgt, trans) in transition_terms if tgt == var
            ]:
                state_var_flows.append(transition)
            if len(state_var_flows) > 0:
                flows = Plus(
                    Times(Real(step_size), Plus(state_var_flows)).simplify(),
                    current_state[var],
                ).simplify()
            else:
                flows = current_state[v_index]

            net_flows.append(Equals(next_state[var], flows))

        return And(net_flows)

    def _get_rate_parameter_symbol(self, rate):
        if isinstance(rate, float):
            return Real(rate)
        else:
            return Symbol(rate, REAL)

    def _encode_self_term(
        self, model: "AbstractRegnetModel", vertex, current_state
    ):
        src = tgt = model._vertice_id(vertex)

        sign = 1 if model._vertice_sign(vertex) else -1
        rate = model._vertice_rate_constant(vertex)

        return tgt, self._encode_term(sign, src, None, rate, current_state)

    def _encode_transition_term(self, model, transition, current_state):
        src = model._transition_source(transition)
        tgt = model._transition_target(transition)
        sign = 1 if model._transition_sign(transition) else -1

        rate = model._transition_rate_constant(transition)

        return tgt, self._encode_term(sign, src, tgt, rate, current_state)

    def _encode_term(self, sign, src, tgt, rate, current_state):
        src_symbol = current_state[src]
        tgt_symbol = current_state[tgt] if tgt is not None else None
        rate_symbol = self._get_rate_parameter_symbol(rate)

        if tgt_symbol is not None:
            formula = Times([Real(sign), tgt_symbol, src_symbol, rate_symbol])
        else:
            formula = Times([Real(sign), src_symbol, rate_symbol])

        return formula

    def _get_timed_symbols(self, model: Model) -> Set[str]:
        """
        Get the names of the state (i.e., timed) variables of the model.

        Parameters
        ----------
        model : Model
            The regnet model

        Returns
        -------
        List[str]
            state variable names
        """

        return set(model._state_var_names())
