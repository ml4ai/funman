import unittest

import pysmt
from pysmt.shortcuts import REAL, Minus, Plus, Real, Symbol, Times, get_env

from funman.translate.simplifier import FUNMANSimplifier


class TestUseCases(unittest.TestCase):
    def test_distribute_constant_over_times(self):
        #    (* (- c1 (* beta c2)) c3)
        # -> (- (* c1 c3) (* beta c2 c3))

        beta = Symbol("beta", REAL)
        c1 = Real(2.0)
        c2 = Real(3.0)
        c3 = Real(4.0)

        env = get_env()
        env._simplifier = FUNMANSimplifier(env)

        original = Times(Minus(c1, Times(beta, c2)), c3)
        simplified = original.simplify()

        original_str = original.to_smtlib(daggify=False)
        simplified_str = simplified.to_smtlib(daggify=False)
        assert (
            original_str != simplified_str
        ), "Simplification did not change formula"
        assert (
            simplified_str == "(- 8.0 (* beta 12.0))"
        ), "Simplification constructed wrong formula"

    def test_group_terms(self):
        #    (* (c1 - (beta * c2)) * (beta * c3))
        # -> (- (* beta c1 c3) (* beta c2 c3))
        beta = Symbol("beta", REAL)
        c1 = Real(2.0)
        c2 = Real(3.0)
        c3 = Real(4.0)

        env = get_env()
        env._simplifier = FUNMANSimplifier(env)

        original = Times(Minus(c1, Times(beta, c2)), Times(beta, c3))
        simplified = original.simplify()

        original_str = original.to_smtlib(daggify=False)
        simplified_str = simplified.to_smtlib(daggify=False)
        assert (
            original_str != simplified_str
        ), "Simplification did not change formula"
        assert (
            simplified_str == "(- 8.0 (* beta 12.0))"
        ), "Simplification constructed wrong formula"


if __name__ == "__main__":
    unittest.main()
