from typing import Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from funman import Funman
from funman.funman import FUNMANConfig
from funman.scenario import AnalysisScenarioResultException
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.scenario.simulation import (
    SimulationScenario,
    SimulationScenarioResult,
)

app = FastAPI(title="funman_api")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put(
    "/solve/consistency",
    response_model=Union[
        ConsistencyScenarioResult, AnalysisScenarioResultException
    ],
)
async def solve_consistency(
    scenario: ConsistencyScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        # print(e)
        # raise e
        return AnalysisScenarioResultException(
            exception=f"Failed to solve scenario due to internal exception: {e}"
        )
    return result


@app.put(
    "/solve/parameter_synthesis",
    response_model=Union[
        ParameterSynthesisScenarioResult, AnalysisScenarioResultException
    ],
)
async def solve_parameter_synthesis(
    scenario: ParameterSynthesisScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        # print(e)
        # raise e
        return AnalysisScenarioResultException(
            exception=f"Failed to solve scenario due to internal exception: {e}"
        )
    return result


@app.put(
    "/solve/simulation",
    response_model=Union[
        SimulationScenarioResult, AnalysisScenarioResultException
    ],
)
async def solve_simulation(
    scenario: SimulationScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        # print(e)
        # raise e
        return AnalysisScenarioResultException(
            exception=f"Failed to solve scenario due to internal exception: {e}"
        )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
