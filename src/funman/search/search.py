import datetime
from abc import ABC, abstractmethod
from multiprocessing import Array, Queue, Value
from multiprocessing.managers import SyncManager
from queue import Queue as SQueue
from typing import List, Optional, Union

import multiprocess as mp
import pysmt
from pydantic import BaseModel

from funman.scenario import AnalysisScenario
from funman.search.handlers import NoopResultHandler, ResultHandler


class SearchStatistics(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    _multiprocessing: bool = False
    _num_true: Union[int, Value] = 0
    _num_false: Union[int, Value] = 0
    _num_unknown: Union[int, Value] = 0
    _residuals: Union[Queue, SQueue] = None
    _current_residual: Union[float, Value] = 0.0
    _last_time: Union[List[datetime.datetime], Array] = None
    _iteration_time: Union[SQueue, Queue] = None
    _iteration_operation: Union[SQueue, Queue] = None


class SearchStaticsMP(SearchStatistics):
    @staticmethod
    def from_manager(manager: SyncManager) -> "SearchStatistics":
        ss = SearchStatistics()

        ss._multiprocessing = manager is not None
        ss._num_true = manager.Value("i", 0) if ss._multiprocessing else 0
        ss._num_false = manager.Value("i", 0) if ss._multiprocessing else 0
        ss._num_unknown = manager.Value("i", 0) if ss._multiprocessing else 0
        ss._residuals = manager.Queue() if ss._multiprocessing else SQueue()
        ss._current_residual = (
            manager.Value("d", 0.0) if ss._multiprocessing else 0.0
        )
        ss._last_time = manager.Array("u", "") if ss._multiprocessing else []
        ss._iteration_time = (
            manager.Queue() if ss._multiprocessing else SQueue()
        )
        ss._iteration_operation = (
            manager.Queue() if ss._multiprocessing else SQueue()
        )
        return ss


class SearchConfig(ABC):
    def __init__(
        self,
        *,  # non-positional keywords
        tolerance=1e-1,
        queue_timeout=1,
        number_of_processes=mp.cpu_count(),
        handler: ResultHandler = NoopResultHandler(),
        wait_timeout=None,
        wait_action=None,
        wait_action_timeout=0.05,
        read_cache=None,
        episode_type=None,
        search=None,
        solver="z3",
    ) -> None:
        self.tolerance = tolerance
        self.queue_timeout = queue_timeout
        self.number_of_processes = number_of_processes
        self.handler: ResultHandler = handler
        self.wait_timeout = wait_timeout
        self.wait_action = wait_action
        self.wait_action_timeout = wait_action_timeout
        self.read_cache = read_cache
        self.episode_type = episode_type
        self.search = search
        self.solver = solver
        if self.solver == "dreal":
            try:
                import funman_dreal
            except:
                raise Exception(
                    "The funman_dreal package failed to import. Do you have it installed?"
                )
            else:
                funman_dreal.ensure_dreal_in_pysmt()


class SearchEpisode(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    problem: AnalysisScenario
    config: SearchConfig
    statistics: SearchStatistics = SearchStatistics()
    _model: pysmt.solvers.solver.Model

    def num_parameters(self):
        return len(self.problem.parameters)


class Search(ABC):
    def __init__(self) -> None:
        self.episodes = []

    @abstractmethod
    def search(
        self, problem: AnalysisScenario, config: Optional[SearchConfig] = None
    ) -> SearchEpisode:
        pass
