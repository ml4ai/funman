import unittest

import funman_dreal  # Needed to use dreal with pysmt
from pysmt.logics import QF_NRA
from pysmt.shortcuts import (
    LE,
    REAL,
    And,
    Equals,
    Not,
    Plus,
    Pow,
    Real,
    Solver,
    Symbol,
    get_model,
)

# from pysmt.smtlib.script import smtlibscript_from_formula


class TestRunDrealNative(unittest.TestCase):
    def test_dreal(self):

        # from dreal import *

        # x = Variable("x")
        # y = Variable("y")
        # z = Variable("z")

        # f_sat = And(0 <= x, x <= 10, 0 <= y, y <= 10, 0 <= z, z <= 10, sin(x) + cos(y) == z)

        # result = CheckSatisfiability(f_sat, 0.001)
        # print(result)

        with Solver(name="dreal", logic=QF_NRA) as s:
            x = Symbol("x", REAL)
            y = Symbol("y", REAL)
            z = Symbol("z", REAL)

            f_sat = And(
                [
                    LE(Real(0.0), x),
                    LE(x, Real(10.0)),
                    LE(Real(0.0), y),
                    LE(y, Real(10.0)),
                    LE(Real(0.0), z),
                    LE(z, Real(10.0)),
                    Equals(
                        Plus(Pow(x, Real(2.0)), Pow(y, Real(2.0))),
                        # Plus(x, y),
                        # Real(1.0)
                        Pow(z, Real(2.0)),
                    ),
                ]
            )
            s.add_assertion(f_sat)

            result = None
            if s.solve():
                result = s.get_model()
            print(f"Result: {result}")

            s.push(1)

            a = Symbol("a", REAL)
            g_sat = And(Equals(Plus(a, z), Real(1.0)))

            s.add_assertion(g_sat)
            result = None
            if s.solve():
                result = s.get_model()
            print(f"Result: {result}")

            s.pop(1)

            q_sat = Not(Equals(Plus(a, z), Real(1.0)))
            s.add_assertion(q_sat)
            result = None
            if s.solve():
                result = s.get_model()

            print(f"Result: {result}")


if __name__ == "__main__":
    unittest.main()
