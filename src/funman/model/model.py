"""
This module represents the abstract base classes for models.
"""
import copy
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List

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

    def _parameter_names(self) -> List[str]:
        return []

    def _state_var_names(self) -> List[str]:
        return []

    def _parameter_names(self):
        return []

    def _parameter_values(self):
        return {}

    def _parameters(self) -> List["Parameter"]:
        return []

    def _parameter_lb(self, param_name: str):
        return None

    def _parameter_ub(self, param_name: str):
        return None
