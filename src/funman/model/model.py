"""
This module represents the abstract base classes for models.
"""
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
    def default_encoder(self) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        pass
