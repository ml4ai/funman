import os
import unittest

from funman.model.bilayer import BilayerDynamics

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources/bilayer"
)


class TestCompilation(unittest.TestCase):
    def initial_bilayer(self):
        bilayer_src1 = {
            "Wa": [
                {"influx": 1, "infusion": 2},  # (beta_1, E')
                {"influx": 2, "infusion": 4},  # (gamma, R')
                {"influx": 8, "infusion": 3},  # (epsilon, I')
                {"influx": 3, "infusion": 5},  # (mu_s, D')
                {"influx": 4, "infusion": 5},  # (mu_e, D')
                {"influx": 5, "infusion": 5},  # (mu_i, D')
                {"influx": 6, "infusion": 5},  # (mu_r, D')
                {"influx": 7, "infusion": 5},  # (alpha, D')
            ],
            "Win": [
                {"arg": 1, "call": 1},  # (S, beta_1)
                {"arg": 1, "call": 3},  # (S, mu_s)
                {"arg": 3, "call": 1},  # (I, beta_1)
                {"arg": 3, "call": 2},  # (I, gamma)
                {"arg": 2, "call": 4},  # (E, mu_e)
                {"arg": 2, "call": 8},  # (E, epsilon)
                {"arg": 3, "call": 7},  # (I, alpha)
                {"arg": 3, "call": 5},  # (I, mu_i)
                {"arg": 4, "call": 6},  # (R, mu_r)
            ],
            "Box": [
                {
                    "parameter": "beta_1",
                    "metadata": {
                        "ref": "http://34.230.33.149:8772/askemo:0000008",
                        "type": "float",
                        "lb": "0.0",
                    },
                },  # 1
                {"parameter": "gamma"},  # 2
                {"parameter": "mu_s"},  # 3
                {"parameter": "mu_e"},  # 4
                {"parameter": "mu_i"},  # 5
                {"parameter": "mu_r"},  # 6
                {"parameter": "alpha"},  # 7
                {"parameter": "epsilon"},  # 8
            ],
            "Qin": [
                {
                    "variable": "S",
                    "metadata": {
                        "ref": "http://34.230.33.149:8772/askemo:0000001"
                    },
                },
                {"variable": "E"},
                {"variable": "I"},
                {"variable": "R"},
                {"variable": "D"},
            ],
            "Qout": [
                {"tanvar": "S'"},
                {"tanvar": "E'"},
                {"tanvar": "I'"},
                {"tanvar": "R'"},
                {"tanvar": "D'"},
            ],
            "Wn": [
                {"efflux": 3, "effusion": 1},  # (mu, S')
                {"efflux": 1, "effusion": 1},  # (beta_1, S')
                {"efflux": 4, "effusion": 2},  # (mu_e, E')
                {"efflux": 8, "effusion": 2},  # (epsilon, E')
                {"efflux": 6, "effusion": 4},  # (mu_r, R')
                {"efflux": 2, "effusion": 3},  # (gamma, I')
                {"efflux": 5, "effusion": 3},  # (mu_i, I')
                {"efflux": 7, "effusion": 3},  # (mu_i, I')
            ],
        }
        return bilayer_src1

    def test_read_bilayer(self):
        bilayer = self.initial_bilayer()
        dynamics = BilayerDynamics(json_graph=bilayer)
        # quick and dirty grabs of the metadata for beta_1
        metadata = list(dynamics._node_incoming_edges)[0].metadata
        assert metadata.ref == "http://34.230.33.149:8772/askemo:0000008"
        assert metadata.type == "float"
        assert metadata.lb == 0.0
        assert metadata.ub is None


if __name__ == "__main__":
    unittest.main()
