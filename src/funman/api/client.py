"""
Client API Generation and usage functionality.
"""
import sys
from os import chdir, path

import openapi_python_client
from openapi_python_client import Config, MetaType


def make_client(
    install_path: str,
    client_name: str = "funman-api-client",
    openapi_url: str = "http://0.0.0.0:8190/openapi.json",
):
    """
    Use openapi-python-client generator to create client.  Adds client package to sys.path.
    """
    client_path = path.join(install_path, client_name)
    chdir(install_path)
    if path.exists(client_path):
        openapi_python_client.update_existing_client(
            url=openapi_url,
            path=None,
            meta=MetaType.POETRY,
            config=Config(),
        )
    else:
        openapi_python_client.create_new_client(
            url=openapi_url,
            path=None,
            meta=MetaType.POETRY,
            config=Config(),
        )
    sys.path.append(path.join(install_path, "funman-api-client"))
