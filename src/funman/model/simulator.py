"""
This module defines classes that wrap a simulator to function as a Model.
"""


from funman.model.model import Model


class SimulatorModel(Model):
    main_fn: str = ""

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
