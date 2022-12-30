from abc import ABC, abstractmethod
from multiprocessing.managers import SyncManager
from queue import Queue as SQueue
from typing import Optional

import multiprocess as mp

from funman.scenario import AnalysisScenario
from funman.search.handlers import NoopResultHandler, ResultHandler


class SearchStatistics(object):
    def __init__(self, manager: Optional[SyncManager] = None):
        self.multiprocessing = manager is not None
        self.num_true = manager.Value("i", 0) if self.multiprocessing else 0
        self.num_false = manager.Value("i", 0) if self.multiprocessing else 0
        self.num_unknown = manager.Value("i", 0) if self.multiprocessing else 0
        self.residuals = manager.Queue() if self.multiprocessing else SQueue()
        self.current_residual = (
            manager.Value("d", 0.0) if self.multiprocessing else 0.0
        )
        self.last_time = manager.Array("u", "") if self.multiprocessing else []
        self.iteration_time = (
            manager.Queue() if self.multiprocessing else SQueue()
        )
        self.iteration_operation = (
            manager.Queue() if self.multiprocessing else SQueue()
        )


class SearchConfig(ABC):
    def __init__(
        self,
        *,  # non-positional keywords
        tolerance=1e-2,
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


class SearchEpisode(object):
    def __init__(
        self, config: SearchConfig, problem: "AnalysisScenario"
    ) -> None:
        self.config: SearchConfig = config
        self.problem = problem
        self.num_parameters: int = len(self.problem.parameters)
        self.statistics = SearchStatistics()


class Search(ABC):
    def __init__(self) -> None:
        self.episodes = []

    @abstractmethod
    def search(
        self, problem: AnalysisScenario, config: Optional[SearchConfig] = None
    ) -> SearchEpisode:
        pass
