import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from .exception import *
from .query import FunmanResults


# Basic placeholder storage utility for query results
class Storage:
    def __init__(self):
        self.started = False
        self.lock = Lock()
        self.path = Path(".").resolve()
        self.results = {}

    def _check_start(self):
        if not self.started:
            raise Exception("Storage not started")

    def start(self, path: Optional[str] = None):
        with self.lock:
            if path is not None and isinstance(path, str):
                self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.results = {}
            for path_object in self.path.glob("*.json"):
                if not path_object.is_file():
                    continue
                self.results[path_object.stem] = None
            self.started = True

    def stop(self):
        with self.lock:
            self.started = False

    def claim_id(self) -> str:
        with self.lock:
            self._check_start()
            result = None
            while True:
                result = str(uuid.uuid4())
                if result not in self.results:
                    break
            self.results[result] = None
            return result

    def add_result(self, result: FunmanResults):
        with self.lock:
            self._check_start()
            if result is None or not isinstance(result, FunmanResults):
                raise FunmanException(f"Result is invalid object")
            id = result.id
            if not isinstance(id, str) or not id:
                raise FunmanException(f"Result id is invalid")
            if id not in self.results:
                raise FunmanException(
                    f"Id {id} does not exist in results (is unclaimed)."
                )
            if self.results[id] is not None:
                raise FunmanException(f"Id {id} was already set to a value.")
            self.results[id] = result
            with open(self.path / f"{id}.json", "w") as f:
                f.write(result.model_dump_json())

    def get_result(self, id: str) -> FunmanResults:
        with self.lock:
            self._check_start()
            if id in self.results and self.results[id] is not None:
                return self.results[id]
            path = self.path / f"{id}.json"
            if not path.is_file():
                raise NotFoundFunmanException("Result for id '{id}' not found")
            result = FunmanResults.parse_file(path)
            self.results[id] = result
            return result
