"""
This module represents the abstract base classes for models.
"""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    init_values: Dict[str, float] = {}
    parameter_bounds: Dict[str, List[float]] = {}

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
