import unittest

import pysmt
import sympy

# from sympy import *
from sympy.printing.smtlib import smtlib_code

# from pysmt.shortcuts import REAL, Minus, Plus, Real, Symbol, Times, get_env


# from funman.translate.simplifier import FUNMANSimplifier


class TestUseCases(unittest.TestCase):
    def test_distribute_constant_over_times(self):
        #    (* (- c1 (* beta c2)) c3)
        # -> (- (* c1 c3) (* beta c2 c3))
        beta, c1, c2, c3 = symbols("beta c1 c2 c3")
        original = c3 * (c1 - (beta * c2))
        print("original:", original)
        simplified = simplify(original)
        print("simplified:", simplified)
        expanded = expand(original)
        print("expanded:", expanded)
        expanded_sub = expanded.subs([(c1, 2.0), (c2, 3.0), (c3, 4.0)])
        print("expanded with substituted values:", expanded_sub)
        expanded_sub_smt = smtlib_code(expanded_sub)
        print("expanded with substituted values in SMT:", expanded_sub_smt)
        assert (
            expanded_sub == 8.0 - 12.0 * beta
        ), "Simplification constructed wrong formula"

    def test_group_terms(self):
        #    (* (c1 - (beta * c2)) * (beta * c3))
        # -> (- (* beta c1 c3) (* beta c2 c3))
        beta, c1, c2, c3 = symbols("beta c1 c2 c3")
        original = (c1 - (beta * c2)) * (beta * c3)
        print("original:", original)
        simplified = simplify(original)
        print("simplified:", simplified)
        expanded = expand(original)
        print("expanded:", expanded)
        expanded_sub = expanded.subs([(c1, 2.0), (c2, 3.0), (c3, 4.0)])
        print("expanded with substituted values:", expanded_sub)


if __name__ == "__main__":
    unittest.main()
