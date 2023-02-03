"""
Definitions for the REST API endpoints.  Running this file will start a uvicorn server that serves the API.

Raises
------
HTTPException
    HTTPException description
"""
import os
from typing import Optional, Union

import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyQuery

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

_FUNMAN_API_TOKEN = os.getenv("FUNMAN_API_TOKEN", None)
api_key_query = APIKeyQuery(name="token", auto_error=False)


def _api_key_auth(api_key: str = Security(api_key_query)):
    # bypass key auth if no token is provided
    if _FUNMAN_API_TOKEN is None:
        print("WARNING: Running without API token")
        return

    # ensure the token is a non-empty string
    if not isinstance(_FUNMAN_API_TOKEN, str) or _FUNMAN_API_TOKEN == "":
        print("ERROR: API token is either empty or not a string")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

    if api_key != _FUNMAN_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
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


@app.get("/", dependencies=[Depends(_api_key_auth)])
def read_root():
    """
    Root endpoint

    Returns
    -------
    Dict
        emtpy result
    """
    return {}


@app.put(
    "/solve/consistency",
    response_model=Union[
        ConsistencyScenarioResult, AnalysisScenarioResultException
    ],
    dependencies=[Depends(_api_key_auth)],
)
async def solve_consistency(
    scenario: ConsistencyScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    """
    Solve a consisistency scenario.

    Parameters
    ----------
    scenario : ConsistencyScenario
        the scenario to solve
    config : Optional[FUNMANConfig], optional
        solver configuration, by default FUNMANConfig()

    Returns
    -------
    ConsistencyScenarioResult
        the scenario result
    """
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
    dependencies=[Depends(_api_key_auth)],
)
async def solve_parameter_synthesis(
    scenario: ParameterSynthesisScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    """
    Solve a Parameter Synthesis Scenario

    Parameters
    ----------
    scenario : ParameterSynthesisScenario
        the scenario to solve
    config : Optional[FUNMANConfig], optional
        solver configuration, by default FUNMANConfig()

    Returns
    -------
        ParameterSynthesisScenarioResult
    """
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
    dependencies=[Depends(_api_key_auth)],
)
async def solve_simulation(
    scenario: SimulationScenario,
    config: Optional[FUNMANConfig] = FUNMANConfig(),
):
    """
    Solve a simulation scenario

    Parameters
    ----------
    scenario : SimulationScenario
        the scenario to solve
    config : Optional[FUNMANConfig], optional
        solver configuration, by default FUNMANConfig()

    Returns
    -------
    SimulationScenarioResult
    """
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
