from model2smtlib.translate import Encoding

from funman.model import Model


class EncodedModel(Model):
    def __init__(self, formula) -> None:
        self.encoding = Encoding(
            formula=formula, symbols=list(formula.get_free_variables())
        )
