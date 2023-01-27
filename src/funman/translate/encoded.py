"""
This module defines enocoders for already encoded models.  (Technically, a
pass-through that helps make the encoder abstraction uniform.)
"""
from funman.model import Model
from funman.model.encoded import EncodedModel
from funman.translate import Encoder, Encoding, EncodingOptions


class EncodedEncoder(Encoder):
    """
    An EncodedEncoder assumes that a model has already been encoded as an
    EncodedModel, and acts as a noop to maintain consistency with other
    encoders.
    """

    def __init__(self, config: "FUNMANConfig") -> None:
        super().__init__(config=config)

    def encode_model(self, model: Model):
        """
        Encode the model by returning the already encoded formula.

        Parameters
        ----------
        model : Model
            Encoded model

        Returns
        -------
        FNode
            SMTLib formula encoding the model
        """
        if isinstance(model, EncodedModel):
            encoding = Encoding(
                formula=model._formula,
                symbols=list(model._formula.get_free_variables()),
            )
            return encoding
        else:
            raise Exception(
                f"An EncodedEncoder cannot encode models of type: {type(model)}"
            )
