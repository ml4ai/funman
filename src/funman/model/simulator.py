"""
This module defines classes that wrap a simulator to function as a Model.
"""

from typing import Callable

from funman.model.model import Model


class SimulatorModel(Model):
    def __init__(
        self, main_fn: Callable, init_values=None, parameter_bounds=None
    ) -> None:
        super().__init__(init_values, parameter_bounds)
        self.main_fn = main_fn

    def default_encoder(self) -> "Encoder":
        """
        Simulators do not rely on encodings, so return a generic encoder.

        Returns
        -------
        Encoder
            _description_
        """
        from funman.translate import DefaultEncoder

        return DefaultEncoder()
