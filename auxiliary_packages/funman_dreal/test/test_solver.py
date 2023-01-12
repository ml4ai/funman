import unittest

import funman_dreal
from pysmt.logics import QF_NRA
from pysmt.shortcuts import (
    LE,
    LT,
    REAL,
    And,
    Equals,
    ForAll,
    Not,
    Or,
    Plus,
    Pow,
    Real,
    Solver,
    Symbol,
)

# from pysmt.smtlib.script import smtlibscript_from_formula


class TestRunDrealNative(unittest.TestCase):
    def test_dreal(self):
        # Needed to use dreal with pysmt
        funman_dreal.ensure_dreal_in_pysmt()

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

    def test_ea_dreal(self):
        # Needed to use dreal with pysmt
        funman_dreal.ensure_dreal_in_pysmt()

        with Solver(name="dreal", logic=QF_NRA) as s:
            x = Symbol("x", REAL)
            x_lb = Symbol("x_lb", REAL)
            x_ub = Symbol("x_ub", REAL)
            y = Symbol("y", REAL)
            z = Symbol("z", REAL)

            ea = ForAll(
                [x],
                And(
                    [
                        And(LE(Real(0.0), x_ub), LT(x_ub, Real(5.0))),
                        And(LE(Real(0.0), x_lb), LT(x_lb, Real(5.0))),
                        LT(x_lb, x_ub),
                        Or(
                            Not(And(LT(x, x_ub), LE(x_lb, x))),
                            Equals(x, Plus(y, z)),
                        ),
                    ]
                ),
            )
            s.push(1)
            s.add_assertion(ea)

            result = None
            if s.solve():
                result = s.get_model()
                print("Model:")
                for v in ea.get_free_variables():
                    print(f"{v} = {float(result.get_py_value(v))}")
                    if v == x_lb:
                        old_x_lb = result.get_py_value(v)
                    if v == x_ub:
                        old_x_ub = result.get_py_value(v)
            else:
                print("Unsat")

            # s.pop(1)

            # ea1 = ForAll(
            #     [x],
            #     And(
            #         [
            #             # convert interval [x_lb, x_ub] from previous call to no-good
            #             Not(
            #                 And(
            #                     LE(Real(1.2495), x_ub),
            #                     LE(x_ub, Real(3.7495000000000003)),
            #                 )
            #             ),
            #             Not(
            #                 And(
            #                     LE(Real(1.2495), x_lb),
            #                     LE(x_lb, Real(3.7495000000000003)),
            #                 )
            #             ),
            #             And(LE(Real(0.0), x_ub), LE(x_ub, Real(5.0))),
            #             And(LE(Real(0.0), x_lb), LE(x_lb, Real(5.0))),
            #             Or(
            #                 Not(And(LE(x, x_ub), LE(x_lb, x))),
            #                 Equals(x, Plus(y, z)),
            #             ),
            #         ]
            #     ),
            # )
            ea1 = And(
                [
                    # convert interval [x_lb, x_ub] from previous call to no-good
                    Not(
                        And(
                            LE(Real(old_x_lb), x_ub),
                            LT(x_ub, Real(old_x_ub)),
                        )
                    ),
                    Not(
                        And(
                            LE(Real(old_x_lb), x_lb),
                            LT(x_lb, Real(old_x_ub)),
                        )
                    ),
                ]
            )

            s.push(1)
            s.add_assertion(ea1)

            result = None
            if s.solve():
                result = s.get_model()
                print("Model:")
                for v in ea1.get_free_variables():
                    print(f"{v} = {float(result.get_py_value(v))}")
            else:
                print("Unsat")
            # print(f"Result: {result}")


if __name__ == "__main__":
    unittest.main()
