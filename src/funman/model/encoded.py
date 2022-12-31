from pysmt.formula import FNode

from funman.model import Model


class EncodedModel(Model):
    """
    Model that holds an SMT formula encoding a model.  This class is meant to wrap hand-coded SMT formulas.
    """

    def __init__(self, formula: FNode) -> None:
        """
        Create an EncodedModel from a pysmt formula.

        Parameters
        ----------
        formula : pysmt.formula.FNode
            formula object
        """
        self.formula = formula

    def default_encoder(self) -> "Encoder":
        """
        EncodedModel uses EncodedEncoder as the default.

        Returns
        -------
        Encoder
            the EncodedEncoder
        """
        from funman.translate.encoded import EncodedEncoder

        return EncodedEncoder()
