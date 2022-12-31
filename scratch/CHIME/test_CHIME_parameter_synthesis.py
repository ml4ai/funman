import os
import unittest
from multiprocessing.heap import Arena

from funman.parameter_space import ParameterSpace
from funman.translate.gromet.gromet import QueryableGromet

RESOURCES = os.path.join("resources")
GROMET_FILE_1 = os.path.join(
    RESOURCES, "gromet", "CHIME_SIR_while_loop--Gromet-FN-auto.json"
)
GROMET_FILE_2 = os.path.join(
    RESOURCES, "gromet", "CHIME_SIR_while_loop--Gromet-FN-auto-one-epoch.json"
)


class Test_CHIME_SIR(unittest.TestCase):
    @unittest.expectedFailure
    def test_parameter_synthesis(self):
        """
        This test constructs two formulations of the CHIME model:
           - the original model where Beta is a epoch-dependent constant over three epochs (i.e., a triple of parameters)
           - a modified variant of the original model using a single constant Beta over the entire simulation (akin to a single epoch).

        It then compares the models by determining that the respective spaces of feasible parameter values intersect.
        """
        ############################ Prepare Models ####################################
        # read in the gromet files
        # GROMET_FILE_1 is the original GroMEt extracted from the simulator
        # It sets N_p = 3 and n_days = 20, resulting in three epochs of 0, 20, and 20 days
        gromet_org = QueryableGromet.from_gromet_file(GROMET_FILE_1)
        # GROMET_FILE_2 modifes sets N_p = 2 and n_days = 40, resulting in one epoch of 40 days
        gromet_sub = QueryableGromet.from_gromet_file(GROMET_FILE_2)
        # Scenario query threshold
        infected_threshold = 130

        ############################ Evaluate Models ###################################
        # get parameter space for the original (3 epochs)
        ps_b1_b2_b3 = gromet_org.synthesize_parameters(
            f"(forall ((t Int)) (<= (I t) {infected_threshold}))"
        )
        # get parameter space for the constant beta variant
        ps_bc = gromet_sub.synthesize_parameters(
            f"(forall ((t Int)) (<= (I t) {infected_threshold}))"
        )

        ############################ Compare Models ####################################
        # construct special parameter space where parameters are equal
        ps_eq = ParameterSpace.construct_all_equal(ps_b1_b2_b3)
        # intersect the original parameter space with the ps_eq to get the
        # valid parameter space where (b1 == b2 == b3)
        ps = ParameterSpace.intersect(ps_b1_b2_b3, ps_eq)
        # assert the parameters spaces for the original and the constant beta
        # variant are the same
        assert ParameterSpace.compare(ps_bc, ps)


if __name__ == "__main__":
    unittest.main()
