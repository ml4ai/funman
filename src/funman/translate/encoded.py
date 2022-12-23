from funman.model import Model
from funman.translate import Encoder, EncodingOptions


class EncodedEncoder(Encoder):
    def __init__(self, config: EncodingOptions = EncodingOptions()) -> None:
        super().__init__(config)

    def encode_model(self, model: Model):
        return model.encoding
