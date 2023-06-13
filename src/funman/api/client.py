"""
Client API Generation and usage functionality.
"""
import os
import sys
from pathlib import Path

import openapi_python_client as opc
from openapi_python_client import MetaType
from openapi_python_client.config import Config


def make_client(
    install_path: str,
    client_name: str = "funman-api-client",
    openapi_url: str = "http://0.0.0.0:8190/openapi.json",
):
    """
    Use openapi-python-client generator to create client.  Adds client package to sys.path.
    """
    install_dir = Path(install_path).resolve()
    client_path = install_dir / Path(client_name)
    prev_cwd = Path.cwd()
    try:
        os.chdir(install_dir)
        if client_path.exists():
            print(
                f"Updating existing funman client at {install_dir} from {openapi_url}"
            )
            opc.update_existing_client(
                url=openapi_url,
                path=None,
                meta=MetaType.POETRY,
                config=Config(),
            )
        else:
            print(
                f"Creating new funman client at {install_dir} from {openapi_url}"
            )
            opc.create_new_client(
                url=openapi_url,
                path=None,
                meta=MetaType.POETRY,
                config=Config(),
            )
        sys.path.append(str(client_path))
    finally:
        os.chdir(prev_cwd)
