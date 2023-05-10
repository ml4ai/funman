"""
This module represents the abstract base classes for models.
"""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel
from pysmt.formula import FNode

from funman.representation.representation import Parameter


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    class Config:
        allow_inf_nan = True

    init_values: Dict[str, float] = {}
    parameter_bounds: Dict[str, List[float]] = {}
    structural_parameter_bounds: Dict[str, List[int]] = {}
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
        return self.init_values[var]

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
