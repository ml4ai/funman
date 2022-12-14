import os
import unittest

from funman_demo.example.chime import CHIME
from funman_dreal.solver import run_dreal

from funman.utils.smtlib_utils import (
    smtlibscript_from_formula,
    smtlibscript_from_formula_list,
)

# from pysmt.smtlib.script import smtlibscript_from_formula


class TestRunDreal(unittest.TestCase):
    def get_flat_encoding(self, num_timepoints):
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=True
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

    def test_dreal(self):
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "out"
        )
        print(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        num_timepoints = 2

        phi = self.get_flat_encoding(num_timepoints)
        phis = self.get_layered_encoding(num_timepoints)

        smt2_flat_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../resources/smt2/chime_flat_{num_timepoints}.smt2",
        )
        smt2_layered_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../resources/smt2/chime_layered_{num_timepoints}.smt2",
        )

        with open(smt2_flat_file, "w") as f:
            smtlibscript_from_formula(phi).serialize(f, daggify=False)
        with open(smt2_layered_file, "w") as f:
            smtlibscript_from_formula_list(phis).serialize(f, daggify=False)

        # print(smt2_file)
        result = run_dreal(
            smt2_flat_file,
            out_dir=out_dir,
            solver_opts="",
        )
        # docker run --rm -it -v `pwd`/code/funman/auxiliary_packages/funman_dreal/resources/smt2:/smt2  dreal/dreal4
        # assert result


if __name__ == "__main__":
    unittest.main()
