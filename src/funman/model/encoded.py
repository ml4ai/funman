from pysmt.formula import FNode
from pysmt.shortcuts import TRUE

from funman.model import Model


class EncodedModel(Model):
    """
    Model that holds an SMT formula encoding a model.  This class is meant to wrap hand-coded SMT formulas.
    """

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    _formula: FNode = TRUE()

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        EncodedModel uses EncodedEncoder as the default.

        Returns
        -------
        Encoder
            the EncodedEncoder
        """
        from funman.translate.encoded import EncodedEncoder

        return EncodedEncoder(config)
