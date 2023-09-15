import json
import os
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from funman_benchmarks.benchmark import Benchmark
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from interruptingcow import timeout
from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Or,
    Plus,
    Real,
    Symbol,
    Times,
)

# from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerGraph, BilayerModel
from funman.model.query import QueryEncoded, QueryTrue
from funman.representation.representation import (
    ModelParameter,
    StructureParameter,
)
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.scenario.scenario import AnalysisScenario, AnalysisScenarioResult
from funman.utils.handlers import ResultCombinedHandler


class TestUnitTests(Benchmark):
    results_df = pd.DataFrame()
    models = {
        "Mosaphir_petri_to_bilayer": "SIDARTHE_BiLayer_corrected.json",
        "UF_petri_to_bilayer": "SIDARTHE_petri_UF_bilayer.json",
        "Morrison_bilayer": "SIDARTHE_BiLayer-CTM-correction.json",
        "Skema_bilayer": "SKEMA_SIDARTHE_PN_BL_renamed_transitions.json",
    }

    def from_json(self, filename):
        with open(
            filename,
            "r",
        ) as f:
            j = json.load(f)
        return j

    def sidarthe_bilayer(self, model: str = "Morrison_bilayer"):
        return self.from_json(
            os.path.join(TestUnitTests.RESOURCES, "bilayer", model)
        )

    def initial_state_sidarthe(self):
        init = self.from_json(
            os.path.join(
                TestUnitTests.RESOURCES,
                "evaluation23",
                "SIDARTHE",
                "SIDARTHE-ic-unit1.json",
            )
        )
        init = {k: round(v, 10) for k, v in init.items()}
        return init

    def bounds_sidarthe(self):
        params = self.from_json(
            os.path.join(
                TestUnitTests.RESOURCES,
                "evaluation23",
                "SIDARTHE",
                "SIDARTHE-params-unit1.json",
            )
        )
        return {k: [v, v] for k, v in params.items()}

    def make_bounds(self, steps, init_values, tolerance=1e-5, step_size=1):
        return And(
            [
                And(
                    [
                        LE(
                            Plus(
                                [Symbol(f"{v}_{i}", REAL) for v in init_values]
                            ),
                            Real(1.0 + tolerance),
                        ),
                        GE(
                            Plus(
                                [Symbol(f"{v}_{i}", REAL) for v in init_values]
                            ),
                            Real(1.0 - tolerance),
                        ),
                        And(
                            [
                                GE(
                                    Symbol(f"{v}_{i}", REAL),
                                    Real(0.0 - tolerance),
                                )
                                for v in init_values
                            ]
                        ),
                    ]
                )
                for i in range(0, steps + 1, step_size)
            ]
        )

    def make_scenario(
        self,
        bilayer,
        init_values,
        parameter_bounds,
        identical_parameters,
        steps,
        step_size,
        query,
        extra_constraints=None,
    ):
        model = BilayerModel(
            bilayer=bilayer,
            init_values=init_values,
            parameter_bounds=parameter_bounds,
            identical_parameters=identical_parameters,
        )
        model._extra_constraints = extra_constraints
        parameters = [
            StructureParameter(name="num_steps", lb=steps, ub=steps),
            StructureParameter(name="step_size", lb=step_size, ub=step_size),
        ]

        scenario = ConsistencyScenario(
            model=model,
            query=query,
            parameters=[
                StructureParameter(name="num_steps", lb=steps, ub=steps),
                StructureParameter(
                    name="step_size", lb=step_size, ub=step_size
                ),
            ],
        )
        return scenario

    def make_ps_scenario(
        self,
        bilayer,
        init_values,
        parameter_bounds,
        identical_parameters,
        steps,
        step_size,
        query,
        params_to_synth=["inf_o_o", "rec_o_o"],
        extra_constraints=None,
    ):
        model = BilayerModel(
            bilayer=bilayer,
            init_values=init_values,
            parameter_bounds=parameter_bounds,
            identical_parameters=identical_parameters,
        )
        model._extra_constraints = extra_constraints
        parameters = [
            ModelParameter(name=k, lb=v[0], ub=v[1])
            for k, v in parameter_bounds.items()
            if k in params_to_synth
        ] + [
            StructureParameter(name="num_steps", lb=steps, ub=steps),
            StructureParameter(name="step_size", lb=step_size, ub=step_size),
        ]
        scenario = ParameterSynthesisScenario(
            parameters=parameters, model=model, query=query
        )
        return scenario

    def report(self, result: AnalysisScenarioResult, name):
        if result.consistent:
            parameters = result.scenario.parameters

            res = pd.Series(name=name, data=parameters).to_frame().T
            self.results_df = pd.concat([self.results_df, res])
            result.scenario.model.bilayer.to_dot(
                values=result.scenario.model.variables()
            ).render(f"{name}_bilayer")
            point = result.parameter_space.true_points[0]
            print(result.dataframe(point))
            ax = result.plot(
                point,
                variables=list(result.scenario.model.init_values.keys()),
                title="\n".join(textwrap.wrap(str(parameters), width=75)),
            )
            ax.set_xlabel("Day")
            ax.set_ylabel("Proportion Population")
            try:
                plt.savefig(f"{name}_test_case.png")
            except Exception as e:
                print(f"Model {name}: Exception while plotting: {e}")
            plt.clf()
        else:
            print(f"Model {name}: is inconsistent")
