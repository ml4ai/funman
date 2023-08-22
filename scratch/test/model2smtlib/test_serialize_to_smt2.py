import unittest

from pysmt.shortcuts import GT, And, Int, Or, Solver, Symbol
from pysmt.solvers.z3 import Z3Solver
from pysmt.typing import BOOL, INT


class TestSerializeToSMT2(unittest.TestCase):
    """
    Test Serialization to SMT2
    """

    def test_serialize_to_smt2(self):
        """
        NOTE This only works with the Z3 solver for the moment.

        pysmt does not _seem_ to have a well defined 'to_smt2' function
        defined within its standard API so I had to delve into the Z3 API to
        get access to something that does a complete SMT2 serialization.

        There is at least a .serialize() behavior available at the pysmt level.
        Combining it will some processing of the current state of the solver
        may allow for a general SMT2 serialization function. For now I am just
        falling back to Z3.
        """
        solver: Z3Solver
        with Solver(name="z3") as solver:
            a = Symbol("A", BOOL)
            b = Symbol("B", BOOL)
            c = Symbol("C", INT)
            solver.add_assertion(Or([And([a, b]), GT(c, Int(5))]))
            print(solver.z3.to_smt2())


if __name__ == "__main__":
    unittest.main()
