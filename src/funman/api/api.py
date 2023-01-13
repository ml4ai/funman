from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from funman import Funman
from funman.scenario import AnalysisScenario as IAnalysisScenario
from funman.scenario import AnalysisScenarioResult as IAnalysisScenarioResult
from funman.scenario import Config as IConfig
from funman.scenario import ConsistencyScenario as IConsistencyScenario

app = FastAPI()


class Parameter(BaseModel):
    name: str
    lb: Union[float, str]
    ub: Union[float, str]


class AnalysisScenario(BaseModel):
    parameters: List[Parameter]


class ConsistencyScenario(AnalysisScenario):
    model: Dict


class AnalysisScenarioResult(BaseModel):
    result: str


class Config(BaseModel):
    pass


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.put("/solve")
def solve(
    scenario: AnalysisScenario,
    config: Optional[Config],
) -> AnalysisScenarioResult:
    i_scenario = IAnalysisScenario()
    i_scenario.parameters = scenario.parameters
    i_config = IConfig()
    f = Funman()
    i_result: IAnalysisScenarioResult = f.solve(i_scenario, config=i_config)
    result: AnalysisScenarioResult = AnalysisScenarioResult(result="bar")
    return result


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
