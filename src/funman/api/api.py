from typing import Optional, Union

import uvicorn
from fastapi import FastAPI

from funman import Funman
from funman.scenario import AnalysisScenario, AnalysisScenarioResult, Config

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/solve")
def solve(
    scenario: AnalysisScenario,
    config: Optional[Config],
) -> AnalysisScenarioResult:
    f = Funman()
    result: AnalysisScenarioResult = f.solve(scenario, config=config)
    return result


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
