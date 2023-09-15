import logging
import os
import unittest

from parameterized import parameterized

from funman.utils.run import Runner

logging.getLogger("funman.translate.translate").setLevel(logging.DEBUG)

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../resources"
)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "terrarium-tests")

cases = [
    (
        os.path.join(AMR_PETRI_DIR, "t01_request.json"),
        os.path.join(AMR_PETRI_DIR, "t01_model.json"),
        "Nelson's test case using ub query on I and parameter ranges.  Consistency Problem.",
    ),
]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestModels(unittest.TestCase):
    @parameterized.expand(cases)
    def test_request(self, request, model, desc):
        result = Runner().run(model, request, case_out_dir=out_dir)
        assert result


if __name__ == "__main__":
    unittest.main()
