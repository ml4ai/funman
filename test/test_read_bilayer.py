import os
import unittest

from pysmt.shortcuts import get_model

from funman.model import BilayerDynamics
from funman.translate import BilayerEncoder

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources/bilayer"
)


class TestCompilation(unittest.TestCase):
    def test_read_bilayer(self):
        bilayer_json_file = os.path.join(
            DATA, "CHIME_SIR_dynamics_BiLayer.json"
        )
        bilayer = BilayerDynamics.from_json(bilayer_json_file)
        assert bilayer

        #        encoding = bilayer.to_smtlib_timepoint(2) ## encoding at the single timepoint 2
        encoder = BilayerEncoder()
        encoding = encoder._encode_bilayer(
            bilayer, [2.5, 3, 4, 6]
        )  ## encoding at the list of timepoints [2,3]
        assert encoding
        model = get_model(encoding)
        assert model


if __name__ == "__main__":
    unittest.main()
