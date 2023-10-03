from typing import List, Optional

from pydantic import ConfigDict, PrivateAttr
from pysmt.formula import FNode
from pysmt.shortcuts import TRUE

from funman.model import Model
from funman.representation.parameter import ModelParameter


class EncodedModel(Model):
    """
    Model that holds an SMT formula encoding a model.  This class is meant to wrap hand-coded SMT formulas.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    _formula: FNode = PrivateAttr(TRUE())
    parameters: Optional[List[ModelParameter]] = None

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
        if self.parameters:
            return [p.name for p in self.parameters]
        else:
            return None

    def _state_var_names(self) -> List[str]:
        return [str(x) for x in self._formula.get_free_variables()]
