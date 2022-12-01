from funman.examples.chime import CHIME
from funman.model import CannedModel

from pysmt.shortcuts import get_free_variables, And


class ChimeModel(CannedModel):
    def __init__(
        self,
        config=None,
        chime: CHIME = None,
        init_values=None,
        parameter_bounds=None,
    ) -> None:
        super().__init__(
            init_values=init_values, parameter_bounds=parameter_bounds
        )
        self.config = config
        self.chime = chime
