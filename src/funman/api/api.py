from typing import Optional, Union

import uvicorn
from fastapi import FastAPI

from funman import Funman
from funman.scenario import AnalysisScenario, AnalysisScenarioResult, Config
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.simulation import (
    SimulationScenario,
    SimulationScenarioResult,
)

app = FastAPI(title="funman_api")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/solve")
def solve(
    scenario: Union[ConsistencyScenario, SimulationScenario],
    config: Optional[Config] = None,
) -> Union[ConsistencyScenarioResult, SimulationScenarioResult]:
    try:
        f = Funman()
        result: AnalysisScenarioResult = f.solve(scenario, config=config)
    except Exception as e:
        print(e)
        raise e
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
