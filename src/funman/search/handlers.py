import traceback
from abc import ABC, abstractmethod
from typing import List


class WaitAction(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ResultHandler(ABC):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> "ResultHandler":
        self.open()
        return self

    def __exit__(self) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def process(self, result: dict) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class NoopResultHandler(ResultHandler):
    def open(self) -> None:
        pass

    def process(self, result: dict) -> None:
        pass

    def close(self) -> None:
        pass


class ResultCombinedHandler(ResultHandler):
    def __init__(self, handlers: List[ResultHandler]) -> None:
        self.handlers = handlers if handlers is not None else []

    def open(self) -> None:
        for h in self.handlers:
            try:
                h.open()
            except Exception as e:
                l.error(traceback.format_exc())

    def process(self, result: dict) -> None:
        for h in self.handlers:
            try:
                h.process(result)
            except Exception as e:
                l.error(traceback.format_exc())

    def close(self) -> None:
        for h in self.handlers:
            try:
                h.close()
            except Exception as e:
                l.error(traceback.format_exc())
