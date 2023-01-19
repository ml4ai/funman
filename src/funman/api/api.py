from typing import Optional

import uvicorn
from fastapi import FastAPI

from funman import Funman
from funman.funman import FUNMANConfig
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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/solve/consistency")
async def solve_consistency(
    scenario: ConsistencyScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
) -> ConsistencyScenarioResult:
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        print(e)
        raise e
    return result


@app.put("/solve/parameter_synthesis")
async def solve_parameter_synthesis(
    scenario: ParameterSynthesisScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
) -> ParameterSynthesisScenarioResult:
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        print(e)
        raise e
    return result


@app.put("/solve/simulation")
async def solve_simulation(
    scenario: SimulationScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
) -> SimulationScenarioResult:
    try:
        f = Funman()
        result = f.solve(scenario, config=config)
    except Exception as e:
        print(e)
        raise e
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
