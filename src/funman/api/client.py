"""
Client API Generation and usage functionality.
"""
import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import openapi_python_client as opc
from openapi_python_client import MetaType
from openapi_python_client.cli import handle_errors
from openapi_python_client.config import Config


def get_patched_schema(url: str, output: Path):
    data = opc._get_document(url=url, path=None, timeout=5)
    for k, v in data["components"]["schemas"].items():
        m = re.match(r"^funman__model__generated_models__(.+)__.*$", k)
        if m:
            t = "title"
            v[t] = f"internal_generated_{m.group(1)}_{v[t]}"
    output.write_text(json.dumps(data))


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

    config = Config(use_path_prefixes_for_title_model_names=True)
    try:
        os.chdir(install_dir)

        with TemporaryDirectory() as tmpdir:
            openapi_file = Path(tmpdir) / Path("openapi.json")
            api = get_patched_schema(openapi_url, openapi_file)

            if client_path.exists():
                print(
                    f"Updating existing funman client at {install_dir} from {openapi_url}"
                )
                errors = opc.update_existing_client(
                    url=None,
                    path=openapi_file,
                    meta=MetaType.POETRY,
                    config=config,
                )
                handle_errors(errors, True)
            else:
                print(
                    f"Creating new funman client at {install_dir} from {openapi_url}"
                )
                errors = opc.create_new_client(
                    url=None,
                    path=openapi_file,
                    meta=MetaType.POETRY,
                    config=config,
                )
                handle_errors(errors, True)
        sys.path.append(str(client_path))
    finally:
        os.chdir(prev_cwd)
