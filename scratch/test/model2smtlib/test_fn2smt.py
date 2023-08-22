import os
import unittest

from pysmt.shortcuts import Symbol, get_model
from pysmt.typing import INT

from funman.translate.gromet.gromet import QueryableGromet

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestCompilation(unittest.TestCase):
    def test_gromet_to_smt1(self):
        """Encoding for `x = 2`"""
        gFile = os.path.join(RESOURCES, "gromet", "exp0--Gromet-FN-auto.json")

        fn = QueryableGromet.from_gromet_file(gFile)
        print(fn._gromet_fn)
        phi = fn.to_smtlib()
        model = get_model(phi)
        assert model  # Is phi satisfiable?
        assert (
            model.get_py_value(Symbol("exp0.fn.x", INT)) == 2
        )  # Did the model get the right assignment?


if __name__ == "__main__":
    unittest.main()
