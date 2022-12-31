import io
import unittest
from functools import partial

from funman_demo.example.chime import CHIME
from funman_dreal.solver import DReal, run_dreal
from pysmt.exceptions import SolverRedefinitionError
from pysmt.logics import QF_NRA
from pysmt.shortcuts import GT, Equals, Real, Solver, Symbol, get_env
from pysmt.typing import REAL

from funman.utils.smtlib_utils import (
    smtlibscript_from_formula,
    smtlibscript_from_formula_list,
)

# from pysmt.smtlib.script import smtlibscript_from_formula


class TestRunDreal(unittest.TestCase):
    def get_flat_encoding(self, num_timepoints):
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=False
        )
        phi = chime.encode_time_horizon(
            parameters, init, dynamics, query, num_timepoints
        )
        return phi

    def get_layered_encoding(self, num_timepoints):
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=True
        )
        phi = chime.encode_time_horizon_layered(
            parameters, init, dynamics, query, num_timepoints
        )
        return phi

    @unittest.skip
    def test_dreal(self):
        """
        This test runs dreal in docker while iteratively sending commands and keeping the docker container and dreal process alive.
        """
        # out_dir = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)), "out"
        # )
        # print(out_dir)
        # if not os.path.exists(out_dir):
        #     os.mkdir(out_dir)

        num_timepoints = 2

        phi = self.get_flat_encoding(num_timepoints)
        # phis = self.get_layered_encoding(num_timepoints)

        f = io.StringIO()
        # smtlibscript_from_formula_list(phis).serialize(f, daggify=False)
        smtlibscript_from_formula(phi).serialize(f, daggify=False)

        dreal = DReal()
        # for command in f.getvalue().split("\n"):
        #     print(f"sending {command}")
        #     output = dreal.send_input(command)
        #     print(f"received {output}")

        rval = dreal.send_input(f.getvalue())
        print(f"Received: \n{rval}")

        rval = dreal.send_input(
            """
        (push 1)
        (assert (and (<= i_0 100) (<= i_2 100) ))
        (check-sat)
        """
        )
        print(f"Received: \n{rval}")

        rval = dreal.send_input(
            """
        (push 1)
        (assert (and (<= beta_0 0.00007) (<= 0.00005 beta_0) ))
        (check-sat)
        """
        )
        print(f"Received: \n{rval}")

        rval = dreal.send_input(
            """
        (pop 1)
        (push 1)
        (assert (and (<= beta_0 0.00006) (<= 0.00005 beta_0) ))
        (check-sat)
        """
        )
        print(f"Received: \n{rval}")

        # smt2_flat_file = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)),
        #     f"../resources/smt2/chime_flat_{num_timepoints}.smt2",
        # )
        # smt2_layered_file = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)),
        #     f"../resources/smt2/chime_layered_{num_timepoints}.smt2",
        # )

        # with open(smt2_flat_file, "w") as f:
        #     smtlibscript_from_formula(phi).serialize(f, daggify=False)

        # print(smt2_file)
        # result = run_dreal(
        #     smt2_flat_file,
        #     out_dir=out_dir,
        #     solver_opts="",
        # )
        # docker run --rm -it -v `pwd`/code/funman/auxiliary_packages/funman_dreal/resources/smt2:/smt2  dreal/dreal4
        # assert result
        assert rval

    def test_pysmt_to_dreal(self):
        name = "dreal"
        # path = "dreal/dreal4"

        env = get_env()
        if name in env.factory._all_solvers:
            raise SolverRedefinitionError("Solver %s already defined" % name)
        solver = partial(DReal, "dreal/dreal4", LOGICS=DReal.LOGICS)
        solver.LOGICS = DReal.LOGICS
        env.factory._all_solvers[name] = solver
        env.factory._generic_solvers[name] = ("dreal/dreal4", DReal.LOGICS)
        env.factory.solver_preference_list.append(name)

        with Solver(name="dreal", logic=QF_NRA) as s:
            s.add_assertion(GT(Symbol("p", REAL), Symbol("q", REAL)))
            res1 = s.solve()
            s.push(1)
            s.add_assertion(GT(Symbol("r", REAL), Symbol("q", REAL)))
            res2 = s.solve()
            s.pop(1)
            s.add_assertion(Equals(Symbol("q", REAL), Real(1.0)))
            s.add_assertion(Equals(Symbol("p", REAL), Real(0.0)))
            res3 = s.solve()  # unsat
        assert res1 and res2 and not res3


if __name__ == "__main__":
    unittest.main()
