"""
This module represents the abstract base classes for models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel, validator
from pysmt.shortcuts import REAL, Symbol

from funman.constants import NEG_INFINITY, POS_INFINITY
from funman.model.query import *


class Model(ABC, BaseModel):
    """
    The abstract base class for Models.
    """

    init_values: Dict[str, float] = None
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
