"""
This module represents the abstract base classes for models.
"""
import copy
import uuid
from abc import ABC
from typing import Dict, List, Optional

from pydantic import ConfigDict, BaseModel
from pysmt.formula import FNode
from pysmt.shortcuts import REAL, Div, Plus, Real, Symbol

from funman.representation.representation import ModelParameter


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    model_config = ConfigDict(allow_inf_nan=True)

    name: str = f"model_{uuid.uuid4()}"
    init_values: Dict[str, float] = {}
    parameter_bounds: Dict[str, List[float]] = {}
    _normalization_constant: Optional[float] = None
    _extra_constraints: FNode = None
    _normalization_term: Optional[FNode] = None

    # @abstractmethod
    # def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
    #     """
    #     Return the default Encoder for the model

    #     Returns
    #     -------
    #     Encoder
    #         SMT encoder for model
    #     """
    #     pass

    def _symbols(self):
        return self._state_var_names() + self._parameter_names()

    def _get_init_value(self, var: str, scenario: "AnalysisScenario", normalize: bool = True):
        if var in self.init_values:
            value = self.init_values[var]
        elif var in self.parameter_bounds:
            # get parameter for value
            value = self.parameter_bounds[var]
        else:
            value = None

        if isinstance(value, str):
            value = Symbol(value, REAL)
        elif isinstance(value, float):
            value = Real(value)

        if value is not None and scenario.normalization_constant:
            norm = Real(scenario.normalization_constant)
            value = Div(value, norm)
        return value

    def variables(self, include_next_state=False):
        """
        Get all initial values and parameters.
        """
        vars = copy.copy(self.init_values)

        if include_next_state:
            next_vars = {f"{k}'": v for k, v in vars.items()}
            vars.update(next_vars)

        vars.update(self.parameter_bounds)

        return vars
    
    def calculate_normalization_constant(self, scenario: "AnalysisScenario") -> float:
        raise NotImplementedError(f"Cannot Calculate a normalization constant for a model of type {type(self)}")

    # def normalization(self):
    #     if self._normalization_constant:
    #         self._normalization_term = Real(self._normalization_constant)
    #     return self._normalization_term

    def _is_normalized(self, var: str):
        try:
            name, time = var.rsplit("_", 1)
            return name in self._state_var_names()
        except:
            return False

    def _parameters(self) -> List[ModelParameter]:
        param_names = self._parameter_names()
        param_values = self._parameter_values()

        # Get Parameter Bounds in FunmanModel (potentially wrapping an AMR model),
        # if they are overriden by the outer model.
        params = (
            [
                ModelParameter(
                    name=p,
                    lb=self.parameter_bounds[p][0],
                    ub=self.parameter_bounds[p][1],
                )
                for p in param_names
                if self.parameter_bounds
                # and p not in param_values
                and p in self.parameter_bounds and self.parameter_bounds[p]
            ]
            if param_names
            else []
        )

        # Get values from wrapped model if not overridden by outer model

        params += (
            [
                (
                    ModelParameter(
                        name=p,
                        lb=param_values[p],
                        ub=param_values[p],
                    )
                    if param_values[p]
                    else ModelParameter(name=p)
                )
                for p in param_names
                if p in param_values and p not in self.parameter_bounds
            ]
            if param_names
            else []
        )

        return params

    def _parameter_names(self) -> List[str]:
        return []

    def _state_var_names(self) -> List[str]:
        return []

    def _parameter_names(self):
        return []

    def _parameter_values(self):
        return {}

    def _parameter_lb(self, param_name: str):
        return None

    def _parameter_ub(self, param_name: str):
        return None

    def _time_var(self):
        return None
