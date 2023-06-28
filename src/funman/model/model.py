"""
This module represents the abstract base classes for models.
"""
import copy
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List
from funman.representation.representation import Parameter

from pydantic import BaseModel
from pysmt.formula import FNode


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    class Config:
        allow_inf_nan = True

    name: str = f"model_{uuid.uuid4()}"
    init_values: Dict[str, float] = {}
    parameter_bounds: Dict[str, List[float]] = {}
    _extra_constraints: FNode = None

    @abstractmethod
    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        pass

    def _get_init_value(self, var: str):
        if var in self.init_values:
            return self.init_values[var]
        elif var in self.parameter_bounds:
            # get parameter for value
            return self.parameter_bounds[var]
        else:
            return None

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

    def _parameters(self) -> List[Parameter]:
        param_names = self._parameter_names()
        param_values = self._parameter_values()

        # Get Parameter Bounds in FunmanModel (potentially wrapping an AMR model),
        # if they are overriden by the outer model.
        params = [
            Parameter(
                name=p,
                lb=self.parameter_bounds[p][0],
                ub=self.parameter_bounds[p][1],
            )
            for p in param_names
            if self.parameter_bounds
            # and p not in param_values
            and p in self.parameter_bounds and self.parameter_bounds[p]
        ]

        # Get values from wrapped model if not overridden by outer model
        params += [
            Parameter(
                name=p,
                lb=param_values[p],
                ub=param_values[p],
            )
            for p in param_names
            if p in param_values and p not in self.parameter_bounds
        ]

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
