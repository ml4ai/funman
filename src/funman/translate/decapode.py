from pysmt.shortcuts import TRUE

from .translate import Encoder, Encoding


class DecapodeEncoder(Encoder):
    def encode_model(self, scenario: "AnalysisScenario") -> Encoding:
        """
        Encode a model into an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode

        Returns
        -------
        Encoding
            formula and symbols for the encoding
        """
        return Encoding(formula=TRUE(), symbols={})
