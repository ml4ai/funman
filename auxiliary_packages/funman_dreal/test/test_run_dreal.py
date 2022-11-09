import os
import unittest
from funman_dreal.funman_dreal import run_dreal


class TestRunDreal(unittest.TestCase):
    def test_dreal(self):
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "out"
        )
        print(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        smt2_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../resources/smt2/chime_flat_30.smt2",
        )
        print(smt2_file)
        result = run_dreal(
            smt2_file,
            out_dir=out_dir,
            solver_opts="",
        )

        assert result
