import tempfile

import matplotlib.pyplot as plt
from funman_demo.handlers import (
    NotebookImageRefresher,
    RealtimeResultPlotter,
    ResultCacheWriter,
)

from funman import Funman
from funman.model import Parameter, QueryLE, QueryTrue
from funman.model.bilayer import (
    BilayerDynamics,
    BilayerMeasurement,
    BilayerModel,
)
from funman.representation.representation import (
    ResultCombinedHandler,
    SearchConfig,
)
from funman.scenario.consistency import ConsistencyScenario
from funman.translate.bilayer import BilayerEncoder, BilayerEncodingOptions


def run_chime_bilayer_example(output_path):
    # Define the dynamics with a bilayer
    chime_bilayer_src = {
        "Wa": [{"influx": 1, "infusion": 2}, {"influx": 2, "infusion": 3}],
        "Win": [
            {"arg": 1, "call": 1},
            {"arg": 2, "call": 1},
            {"arg": 2, "call": 2},
        ],
        "Box": [{"parameter": "beta"}, {"parameter": "gamma"}],
        "Qin": [{"variable": "S"}, {"variable": "I"}, {"variable": "R"}],
        "Qout": [{"tanvar": "S'"}, {"tanvar": "I'"}, {"tanvar": "R'"}],
        "Wn": [{"efflux": 1, "effusion": 1}, {"efflux": 2, "effusion": 2}],
    }

    chime_bilayer = BilayerDynamics.from_json(chime_bilayer_src)

    # Define the measurements made of the bilayer variables
    measurements = {
        "state": [{"variable": "I"}],
        "observable": [{"observable": "H"}],
        "rate": [{"parameter": "hr"}],
        "Din": [{"variable": 1, "parameter": 1}],
        "Dout": [{"parameter": 1, "observable": 1}],
    }
    hospital_measurements = BilayerMeasurement.from_json(measurements)

    # Model Setup for both Intervention 1
    # - Prescribed reduction in transmission

    transmission_reduction = 0.05
    duration = 120  # days
    model = BilayerModel(
        chime_bilayer,
        measurements=hospital_measurements,
        init_values={"S": 10000, "I": 1, "R": 1},
        parameter_bounds={
            # "beta": [0.00067*(1.0-transmission_reduction), 0.00067*(1.0-transmission_reduction)],
            "beta": [0.00005, 0.00007],
            "gamma": [1.0 / 14.0, 1.0 / 14.0],
            "hr": [0.05, 0.05],
        },
    )
    # query = QueryLE("I", 9000) # TODO change to H after incorporating measurement encoding
    query = QueryTrue()

    # Analyze Intervention 1 to check it will achieve goals of query
    tmp_dir_path = tempfile.mkdtemp(prefix="funman-")
    result = Funman().solve(
        ConsistencyScenario(
            model,
            query,
            _smt_encoder=BilayerEncoder(
                config=BilayerEncodingOptions(step_size=10, max_steps=duration)
            ),  # four months
        )
    )
    if result.consistent:
        result.plot()
        print(f"parameters = {result.parameters()}")
        print(result.dataframe())
        plt.savefig(output_path)
    else:
        print("Scenario Inconsistent")
