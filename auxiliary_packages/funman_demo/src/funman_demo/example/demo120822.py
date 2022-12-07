from funman.model.bilayer import Bilayer, BilayerMeasurement, BilayerModel
from funman.model import Parameter, QueryLE, QueryTrue
from IPython.display import Markdown as md
from IPython.display import Image
from model2smtlib.bilayer.translate import (
    BilayerEncoder,
    BilayerEncodingOptions,
)
from funman.scenario.consistency import ConsistencyScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman import Funman
from funman.model.bilayer import Bilayer, BilayerMeasurement, BilayerModel
from funman_demo.handlers import (
    ResultCacheWriter,
    RealtimeResultPlotter,
    NotebookImageRefresher,
)
from funman.search_utils import ResultCombinedHandler, SearchConfig
from funman.search_episode import DRealSearchEpisode
from funman.search import SMTCheck, BoxSearch
import os
import tempfile
import pandas as pd

# import funman_dreal # Needed to use dreal with pysmt


class Scenario1(object):
    def __init__(
        self,
        init_values={"S": 9998, "I": 1, "R": 1},
        query_threshold=10000,
        duration=1,
        step_size=1,
    ):
        # Define the dynamics with a bilayer

        self.chime_bilayer_src = {
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

        self.chime_bilayer = Bilayer.from_json(self.chime_bilayer_src)

        self.chime_sviivr_bilayer_src = {
            "Qin": [
                {"variable": "S"},
                {"variable": "V"},
                {"variable": "I"},
                {"variable": "I_v"},
                {"variable": "R"},
            ],
            "Box": [
                {"parameter": "beta_1"},
                {"parameter": "beta_2"},
                {"parameter": "v_r"},
                {"parameter": "v_s1"},
                {"parameter": "v_s2"},
                {"parameter": "gamma_1"},
                {"parameter": "gamma_2"},
            ],
            "Qout": [
                {"tanvar": "S'"},
                {"tanvar": "V'"},
                {"tanvar": "I'"},
                {"tanvar": "I_v'"},
                {"tanvar": "R'"},
            ],
            "Win": [
                {"arg": 1, "call": 1},
                {"arg": 1, "call": 2},
                {"arg": 1, "call": 3},
                {"arg": 2, "call": 4},
                {"arg": 2, "call": 5},
                {"arg": 3, "call": 1},
                {"arg": 3, "call": 4},
                {"arg": 3, "call": 6},
                {"arg": 4, "call": 2},
                {"arg": 4, "call": 5},
                {"arg": 4, "call": 7},
            ],
            "Wa": [
                {"influx": 1, "infusion": 3},
                {"influx": 2, "infusion": 3},
                {"influx": 3, "infusion": 2},
                {"influx": 4, "infusion": 4},
                {"influx": 5, "infusion": 4},
                {"influx": 6, "infusion": 5},
                {"influx": 7, "infusion": 5},
            ],
            "Wn": [
                {"efflux": 1, "effusion": 1},
                {"efflux": 2, "effusion": 1},
                {"efflux": 3, "effusion": 1},
                {"efflux": 4, "effusion": 2},
                {"efflux": 5, "effusion": 2},
                {"efflux": 6, "effusion": 3},
                {"efflux": 7, "effusion": 4},
            ],
        }

        self.chime_sviivr_bilayer = Bilayer.from_json(
            self.chime_sviivr_bilayer_src
        )

        # Define the measurements made of the bilayer variables
        # Hospitalizations (H) are a proportion (hr) of those infected (I)

        self.measurements = {
            "state": [{"variable": "I"}],
            "observable": [{"observable": "H"}],
            "rate": [{"parameter": "hr"}],
            "Din": [{"variable": 1, "parameter": 1}],
            "Dout": [{"parameter": 1, "observable": 1}],
        }
        self.hospital_measurements = BilayerMeasurement.from_json(
            self.measurements
        )

        self.measurements1 = {
            "state": [{"variable": "I"}, {"variable": "I_v"}],
            "observable": [{"observable": "H"}],
            "rate": [{"parameter": "hr"}],
            "Din": [
                {"variable": 1, "parameter": 1},
                {"variable": 2, "parameter": 1},
            ],
            "Dout": [{"parameter": 1, "observable": 1}],
        }
        self.hospital_measurements1 = BilayerMeasurement.from_json(
            self.measurements1
        )

        # Model Setup for both Intervention 1
        # - Prescribed reduction in transmission, i.e., beta' = (1-transmission_reduction)beta

        self.config = {
            # "transmission_reduction": 0.05,
            "duration": duration,  # 10
            "step_size": step_size,
            "query_variable": "H",
            "query_threshold": query_threshold,
        }

        self.models = {
            "intervention1": {
                "SIR+H": BilayerModel(
                    self.chime_bilayer,
                    measurements=self.hospital_measurements,
                    init_values=init_values,
                    parameter_bounds={
                        "beta": [
                            0.000067,
                            0.000067,
                        ],
                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                        "hr": [0.01, 0.01],
                    },
                ),
                "SVIIR": BilayerModel(
                    self.chime_sviivr_bilayer,
                    measurements=self.hospital_measurements1,
                    init_values={"S": 10000, "V": 1, "I": 1, "I_v": 1, "R": 1},
                    parameter_bounds={
                        "beta_1": [0.000067, 0.000067],
                        "beta_2": [0.000067, 0.000067],
                        "gamma_1": [1.0 / 14.0, 1.0 / 14.0],
                        "gamma_2": [1.0 / 14.0, 1.0 / 14.0],
                        "v_s1": [0.000067, 0.000067],
                        "v_s2": [0.000067, 0.000067],
                        "v_r": [0.05, 0.05],
                        "hr": [0.01, 0.01],
                    },
                ),
            },
            "intervention2": {
                "SIR+H": BilayerModel(
                    self.chime_bilayer,
                    measurements=self.hospital_measurements,
                    init_values=init_values,
                    parameter_bounds={
                        "beta": [0.000067, 0.000067],
                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                        "hr": [0.01, 0.01],
                    },
                ),
                "SVIIR": BilayerModel(
                    self.chime_sviivr_bilayer,
                    measurements=self.hospital_measurements,
                    init_values={"S": 10000, "V": 1, "I": 1, "I_v": 1, "R": 1},
                    parameter_bounds={
                        "beta_1": [0.000067, 0.000067],
                        "beta_2": [0.000067, 0.000067],
                        "gamma_1": [1.0 / 14.0, 1.0 / 14.0],
                        "gamma_2": [1.0 / 14.0, 1.0 / 14.0],
                        "v_s1": [0.000067, 0.000067],
                        "v_s2": [0.000067, 0.000067],
                        "v_r": [0.05, 0.05],
                        "hr": [0.01, 0.01],
                    },
                ),
            },
        }

        self.encoding_options = BilayerEncodingOptions(
            step_size=self.config["step_size"],
            max_steps=self.config["duration"],
        )

        if self.config["query_threshold"] is None:
            self.query = QueryTrue()
        else:
            self.query = QueryLE(
                self.config["query_variable"], self.config["query_threshold"]
            )

    def to_md(self, model):

        model.measurements.to_dot().render(filename="measurement", format="png")
        model.bilayer.to_dot().render(filename="bilayer", format="png")

        init_values_md = "\n".join(
            {f"- ## {k}: {v}" for k, v in model.init_values.items()}
        )
        parameters_bounds_md = "\n".join(
            {f"- ## {k}: {v}" for k, v in model.parameter_bounds.items()}
        )
        q = (
            "\\bigwedge\\limits_{t \\in [0,"
            + str(self.config["duration"])
            + "]} "
            + self.config["query_variable"]
            + "_t \\leq"
            + str(self.config["query_threshold"])
        )
        config_md = "\n".join(
            {f"- ## {k}: {v}" for k, v in self.config.items()}
        )
        self.md = md(
            f"""# Bilayer and Measurement Model
SIR Bilayer (left), Hospitalized Measurement (right)

![](bilayer.png) ![](measurement.png)
# Initial State (population 10000)
{init_values_md}
# Parameter Bounds
{parameters_bounds_md}
# Scenario Configuration
{config_md}
# Query
- ## ${q}$
"""
        )
        return self.md

    def analyze_intervention_1(self, transmission_reduction, models=[]):
        results = {}
        for model_name, model in self.models["intervention1"].items():
            if model_name not in models:
                continue

            if model_name == "SIR+H":
                model.parameter_bounds["beta"] = [
                    model.parameter_bounds["beta"][0]
                    * (1.0 - transmission_reduction),
                    model.parameter_bounds["beta"][1]
                    * (1.0 - transmission_reduction),
                ]
            elif model_name == "SVIIR":
                for beta in ["beta_1", "beta_2"]:
                    model.parameter_bounds[beta] = [
                        model.parameter_bounds[beta][0]
                        * (1.0 - transmission_reduction),
                        model.parameter_bounds[beta][1]
                        * (1.0 - transmission_reduction),
                    ]
            result = Funman().solve(
                ConsistencyScenario(
                    model,
                    self.query,
                    smt_encoder=BilayerEncoder(config=self.encoding_options),
                ),
                config=SearchConfig(solver="dreal", search=SMTCheck),
            )
            if result.consistent:
                msg = f"{model_name}: Query Satisfied"
                variables = list(model.init_values.keys()) + [
                    v.parameter
                    for k, v in model.measurements.observable.items()
                ]

                plot = result.plot(
                    y=variables,
                    logy=True,
                    title=f"Scenario 1 with {transmission_reduction} reduction in transmissibiilty, {model_name}",
                    ylabel="Population",
                    xlabel="Day",
                )
                dataframe = result.dataframe()
            else:
                msg = f"{model_name}: Query Not Satisfied"
                plot = None
                dataframe = None
            results[model_name] = {
                "message": msg,
                "plot": plot,
                "dataframe": dataframe,
            }

        return results

    def compare_model_results(self, results):
        df = pd.DataFrame(
            {
                f"{name}": result["dataframe"]["H"]
                for name, result in results.items()
                if result["dataframe"] is not None
            }
        )
        df = df.apply(lambda x: self.config["query_threshold"] - x)
        if len(df) > 0:
            ax = df.boxplot()
            ax.set_title("Unused Hospital Capacity per Day")
            ax.set_xlabel("Model")
            ax.set_ylabel("Unused Hospital Capacity")

    def analyze_intervention_2(self, transmission_reduction, models=[]):
        if not isinstance(transmission_reduction, list):
            raise Exception(
                f"transmission_reduction must be a list of the form [lb, ub]"
            )

        results = []
        for model_name, model in self.models["intervention2"].items():
            if model_name not in models:
                continue

            lb = model.parameter_bounds["beta"][0] * (
                1.0 - transmission_reduction[1]
            )
            ub = model.parameter_bounds["beta"][1] * (
                1.0 - transmission_reduction[0]
            )
            model.parameter_bounds["beta"] = [lb, ub]

            parameters = [Parameter("beta", lb=lb, ub=ub)]
            tmp_dir_path = tempfile.mkdtemp(prefix="funman-")
            result = Funman().solve(
                ParameterSynthesisScenario(
                    parameters,
                    model,
                    self.query,
                    smt_encoder=BilayerEncoder(config=self.encoding_options),
                ),
                config=SearchConfig(
                    number_of_processes=1,
                    tolerance=1e-8,
                    solver="dreal",
                    search=BoxSearch,
                    # wait_action = NotebookImageRefresher(os.path.join(tmp_dir_path, "search.png"), sleep_for=1),
                    handler=ResultCombinedHandler(
                        [
                            ResultCacheWriter(
                                os.path.join(tmp_dir_path, "search.json")
                            ),
                            RealtimeResultPlotter(
                                parameters,
                                plot_points=True,
                                realtime_save_path=os.path.join(
                                    tmp_dir_path, "search.png"
                                ),
                            ),
                        ]
                    ),
                ),
            )
            msg = ""
            plot = result.plot()
            df = pd.DataFrame()
            results.append(
                {
                    "message": msg,
                    "plot": plot,
                    "dataframe": df,
                    "parameter_space": result.parameter_space,
                }
            )
        return results

    def analyze_intervention_3(self, transmission_reduction):
        pass
