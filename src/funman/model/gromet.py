"""
This module represents GrometModel-related classes.
"""
from automates.model_assembly.gromet.model import GrometFNModule
from automates.program_analysis.JSON2GroMEt.json2gromet import json_to_gromet

from funman.model import Model


class GrometModel(Model):
    """
    The GrometModel class is a representation of an executable Gromet model.
    """

    def __init__(
        self, gromet: GrometFNModule, init_values=None, parameter_bounds=None
    ) -> None:
        super().__init__(init_values, parameter_bounds)
        self.gromet = gromet

    @staticmethod
    def from_gromet_file(gromet_path):
        return GrometModel(json_to_gromet(gromet_path))
