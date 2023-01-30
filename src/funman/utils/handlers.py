"""
The handlers module includes several search handlers that deal with search episode data, such as plotting and writing results to a file.
"""
import logging
import traceback
from abc import ABC, abstractmethod
from typing import List

LOG_LEVEL = logging.INFO

l = logging.getLogger(__file__)
l.setLevel(LOG_LEVEL)


class WaitAction(ABC):
    """
    The WaitAction abstract class allows search processes to handle waiting for additional work.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ResultHandler(ABC):
    """
    The ResultHandler abstract class handles data produced by search processes.
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "ResultHandler":
        self.open()
        return self

    def __exit__(self) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        """
        Listener interface for starting handling.
        """
        pass

    @abstractmethod
    def process(self, result: dict) -> None:
        """
        Listener interface for processing data in result.

        Parameters
        ----------
        result : dict
            data to process
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Listener interface for ending handling.
        """
        pass


class NoopResultHandler(ResultHandler):
    """
    The NoopResultHandler is used in the absence of other handlers and performs no (i.e., noop) handling.
    """

    def open(self) -> None:
        pass

    def process(self, result: dict) -> None:
        pass

    def close(self) -> None:
        pass


class ResultCombinedHandler(ResultHandler):
    """
    The ResultCombinedHandler combines multiple sub-handlers by iteratively invoking the sub-handler listener interfaces.
    """

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
