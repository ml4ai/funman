"""
Definitions for the REST API endpoints.  Running this file will start a uvicorn server that serves the API.

Raises
------
HTTPException
    HTTPException description
"""
import sys
import traceback
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from typing_extensions import Annotated

from funman import __version__ as FunmanVersion
from funman.api.settings import Settings
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.model.generated_models.regnet import Model as GeneratedRegNet
from funman.model.petrinet import GeneratedPetriNetModel, PetrinetModel
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.server.exception import NotFoundFunmanException
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

settings = Settings()
_storage = Storage()
_worker = FunmanWorker(_storage)


# Rig some services to run while the API is online
@asynccontextmanager
async def lifespan(_: FastAPI):
    _storage.start(settings.data_path)
    _worker.start()
    yield
    _worker.stop()
    _storage.stop()


def get_storage():
    return _storage


def get_worker():
    return _worker


app = FastAPI(title="funman_api", lifespan=lifespan)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="FUNMAN API",
        version=FunmanVersion,
        description="Functional Model Analysis",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

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


@contextmanager
def internal_error_handler():
    eid = uuid.uuid4()
    try:
        yield
    except HTTPException:
        raise
    except Exception:
        print(f"Internal Server Error ({eid}):", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {eid}"
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
    "/queries/{query_id}/halt",
    response_model=str,
    dependencies=[Depends(_api_key_auth)],
)
async def halt(
    query_id: str, worker: Annotated[FunmanWorker, Depends(get_worker)]
):
    with internal_error_handler():
        worker.halt(query_id)
        return "Success"


@app.get(
    "/queries/current",
    response_model=Optional[str],
    dependencies=[Depends(_api_key_auth)],
)
async def get_current(worker: Annotated[FunmanWorker, Depends(get_worker)]):
    with internal_error_handler():
        return worker.get_current()


@app.get(
    "/queries/{query_id}",
    response_model=FunmanResults,
    response_model_exclude_defaults=True,
    dependencies=[Depends(_api_key_auth)],
)
async def get_queries(
    query_id: str, worker: Annotated[FunmanWorker, Depends(get_worker)]
):
    with internal_error_handler():
        try:
            return worker.get_results(query_id)
        except NotFoundFunmanException:
            raise HTTPException(404)


@app.post(
    "/queries",
    response_model=FunmanWorkUnit,
    response_model_exclude_defaults=True,
    dependencies=[Depends(_api_key_auth)],
)
async def post_queries(
    model: Union[
        GeneratedPetriNet,
        GeneratedRegNet,
        RegnetModel,
        PetrinetModel,
        # DecapodeModel,
        # BilayerModel,
    ],
    request: FunmanWorkRequest,
    worker: Annotated[FunmanWorker, Depends(get_worker)],
):
    with internal_error_handler():
        return worker.enqueue_work(_wrap_with_internal_model(model), request)


def _wrap_with_internal_model(
    model: Union[
        GeneratedPetriNet,
        GeneratedRegNet,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
    ]
) -> Union[
    GeneratedPetriNetModel,
    GeneratedRegnetModel,
    RegnetModel,
    PetrinetModel,
    DecapodeModel,
    BilayerModel,
]:
    if isinstance(model, GeneratedPetriNet):
        return GeneratedPetriNetModel(petrinet=model)
    elif isinstance(model, GeneratedRegNet):
        return GeneratedRegnetModel(regnet=model)
    else:
        return model


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8190)
