from typing import Dict, List, Set

from pysmt.formula import FNode
from pysmt.shortcuts import (
    REAL,
    TRUE,
    And,
    Symbol,
    Times,
    substitute,
)

from funman.model.model import Model

from .translate import Encoder, Encoding


class EnsembleEncoder(Encoder):
    def encode_model(self, scenario: "AnalysisScenario") -> Encoding:
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
        model_steps = {
            model.name: substitute(
                model.default_encoder(self.config)._encode_next_step(
                    model, step, next_step
                ),
                self._submodel_substitution_map(
                    model, step=step, next_step=next_step
                ),
            )
            for model in model.models
        }

        return And(list(model_steps.values()))

    def _submodel_substitution_map(
        self, model: Model, step=None, next_step=None
    ) -> Dict[Symbol, Symbol]:
        curr_var_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}_{step}", REAL): Symbol(
                f"model_{model.name}_{variable}_{step}", REAL
            )
            for variable in model._state_var_names()
        }
        next_var_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}_{next_step}", REAL): Symbol(
                f"model_{model.name}_{variable}_{next_step}", REAL
            )
            for variable in model._state_var_names()
        }
        parameter_sub_map: Dict[Symbol, Symbol] = {
            Symbol(f"{variable}", REAL): Symbol(
                f"model_{model.name}_{variable}", REAL
            )
            for variable in model._parameter_names()
        }
        return {**curr_var_sub_map, **next_var_sub_map, **parameter_sub_map}

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
        return set(model._state_var_names())
