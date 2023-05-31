"""
Definitions for the REST API endpoints.  Running this file will start a uvicorn server that serves the API.

Raises
------
HTTPException
    HTTPException description
"""
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing_extensions import Annotated

from funman import Funman
from funman.api.settings import Settings
from funman.funman import FUNMANConfig
from funman.model.ensemble import EnsembleModel
from funman.scenario.consistency import ConsistencyScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.server.exception import NotFoundFunmanException
from funman.server.query import QueryRequest, QueryResponse
from funman.server.storage import Storage

settings = Settings()

_storage = Storage()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await _storage.start(settings.data_path)
    yield
    await _storage.stop()


def get_storage():
    return _storage


app = FastAPI(title="funman_api", lifespan=lifespan)

api_key_header = APIKeyHeader(name="token", auto_error=False)


def _api_key_auth(api_key: str = Security(api_key_header)):
    # bypass key auth if no token is provided
    if settings.funman_api_token is None:
        print("WARNING: Running without API token")
        return

    # ensure the token is a non-empty string
    if (
        not isinstance(settings.funman_api_token, str)
        or settings.funman_api_token == ""
    ):
        print("ERROR: API token is either empty or not a string")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

    if api_key != settings.funman_api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )


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


@app.get(
    "/queries/{query_id}",
    response_model=QueryResponse,
    dependencies=[Depends(_api_key_auth)],
)
async def get_queries(
    query_id: str,
    storage: Annotated[Storage, Depends(get_storage)],
):
    eid = uuid.uuid4()
    try:
        response = await storage.get_result(query_id)
        return response
    except NotFoundFunmanException:
        raise HTTPException(404)
    except Exception as e:
        print(f"Internal Server Error ({eid}): {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {eid}"
        )


@app.post(
    "/queries",
    response_model=QueryResponse,
    dependencies=[Depends(_api_key_auth)],
)
async def post_queries(
    request: QueryRequest,
    storage: Annotated[Storage, Depends(get_storage)],
    config: Optional[FUNMANConfig] = None,
):
    eid = uuid.uuid4()
    try:
        if config is None:
            config = FUNMANConfig()

        # convert to scenario
        if request.parameters is None or all(
            request.parameters[k] == "one" for k in request.parameters
        ):
            kind = "consistency"
            scenario = ConsistencyScenario(
                model=request.model, query=request.query
            )
        else:
            kind = "parameter_synthesis"
            if isinstance(request.model, EnsembleModel):
                raise Exception(
                    "TODO handle EnsembleModel for ParameterSynthesisScenario"
                )
            scenario = ParameterSynthesisScenario(
                model=request.model, parameters=[]
            )

        f = Funman()
        id = await storage.claim_id()
        result = f.solve(scenario, config=config)
        response = QueryResponse(id=id, scenario=kind, result=result)
        await storage.add_result(response)
        return response
    except Exception as e:
        print(f"Internal Server Error ({eid}): {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {eid}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
