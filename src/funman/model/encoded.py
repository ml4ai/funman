from typing import List

from pydantic import ConfigDict, Extra
from pysmt.formula import FNode
from pysmt.shortcuts import TRUE

from funman.model import Model
from funman.representation.representation import ModelParameter


class EncodedModel(Model):
    """
    Model that holds an SMT formula encoding a model.  This class is meant to wrap hand-coded SMT formulas.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _formula: FNode = TRUE()
    parameters: List[ModelParameter] = []

    def default_encoder(
        self, config: "FUNMANConfig", scenario: "AnalysisScenario"
    ) -> "Encoder":
        """
        EncodedModel uses EncodedEncoder as the default.

        Returns
        -------
        Encoder
            the EncodedEncoder
        """
        from funman.translate.encoded import EncodedEncoder

        return EncodedEncoder(config=config, scenario=scenario)

    def _parameter_names(self) -> List[str]:
        return [p.name for p in self.parameters]
