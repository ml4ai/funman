import uuid
from asyncio import Lock
from pathlib import Path
from typing import Optional

from .exception import *
from .query import QueryResponse


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

    async def start(self, path: Optional[str] = None):
        async with self.lock:
            if path is not None and isinstance(path, str):
                self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.results = {}
            for path_object in self.path.glob("*.json"):
                if not path_object.is_file():
                    continue
                self.results[path_object.stem] = None
            self.started = True

    async def stop(self):
        async with self.lock:
            self.started = False

    async def claim_id(self) -> str:
        async with self.lock:
            self._check_start()
            result = None
            while True:
                result = str(uuid.uuid4())
                if result not in self.results:
                    break
            self.results[result] = None
            return result

    async def add_result(self, result: QueryResponse):
        async with self.lock:
            self._check_start()
            if result is None or not isinstance(result, QueryResponse):
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
                f.write(result.json())

    async def get_result(self, id: str) -> QueryResponse:
        async with self.lock:
            self._check_start()
            if id in self.results and self.results[id] is not None:
                return self.results[id]
            path = self.path / f"{id}.json"
            if not path.is_file():
                raise NotFoundFunmanException("Result for id '{id}' not found")
            result = QueryResponse.parse_file(path)
            self.results[id] = result
            return result
