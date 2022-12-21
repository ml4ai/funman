from funman.model import Model
from funman.model2smtlib.translate import Encoding


class EncodedModel(Model):
    def __init__(self, formula) -> None:
        self.encoding = Encoding(
            formula=formula, symbols=list(formula.get_free_variables())
        )
