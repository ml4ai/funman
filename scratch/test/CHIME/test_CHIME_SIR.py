import os
import unittest

from funman_demo.sim.CHIME.CHIME_SIR import main as run_CHIME_SIR

from funman.translate.gromet.gromet import QueryableGromet

RESOURCES = os.path.join("resources")
GROMET_FILE = os.path.join(
    RESOURCES, "gromet", "CHIME_SIR_while_loop--Gromet-FN-auto.json"
)


class Test_CHIME_SIR(unittest.TestCase):
    def compare_results(self, infected_threshold):
        """This function compares the simulator and FUNMAN reasoning about the CHIME SIR model.  The query_simulator function executes the simulator main() as run_CHIME_SIR, and answers the does_not_cross_threshold() query using the simulation reults.  The QueryableGromet class constructs a model from the GroMEt file corresponding to the simulator, and answers a query with the model.  The query for both cases asks whether the number of infected at any time point exceed a specified threshold.  The test will succeed if the simulator and QueryableGromet class agree on the response to the query.

        Args:
            infected_threshold (int): The upper bound for the number of infected for any time point.
        Returns:
            bool: Do the simulator and QueryableGromet results match?
        """

        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)

        q_sim = does_not_cross_threshold(run_CHIME_SIR())

        # query the gromet file
        gromet = QueryableGromet.from_gromet_file(GROMET_FILE)
        q_gromet = gromet.query(
            f"(forall ((t Int)) (<= (I t) {infected_threshold}))"
        )

        # assert the both queries returned the same result
        return q_sim == q_gromet

    def test_query_true(self):
        """This test requires both methods to return True."""
        # threshold for infected population
        infected_threshold = 130
        assert self.compare_results(infected_threshold)

    @unittest.expectedFailure
    def test_query_false(self):
        """This test requires both methods to return False."""
        # threshold for infected population
        infected_threshold = 100
        assert self.compare_results(infected_threshold)


if __name__ == "__main__":
    unittest.main()
