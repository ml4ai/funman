from typing import Union

from fastapi import FastAPI

from funman.funman import Funman
from funman.scenario.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    Config,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/solve")
def solve():
    f = Funman()
    f.solve()


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
