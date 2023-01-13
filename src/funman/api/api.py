from typing import Optional

import uvicorn
from fastapi import FastAPI

from funman import Funman
from funman.scenario import AnalysisScenario, AnalysisScenarioResult, Config
from funman.scenario.consistency import ConsistencyScenario

app = FastAPI(title="funman_api")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/solve")
def solve(
    scenario: ConsistencyScenario,
    config: Optional[Config],
) -> AnalysisScenarioResult:
    f = Funman()
    result: AnalysisScenarioResult = f.solve(scenario, config=config)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
