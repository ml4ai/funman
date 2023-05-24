import datetime
from abc import ABC, abstractmethod
from multiprocessing import Array, Queue, Value
from multiprocessing.managers import SyncManager
from queue import Queue as SQueue
from typing import List, Optional

import pysmt
from pydantic import BaseModel
from pysmt.shortcuts import And, Not, Solver, get_model

from funman.funman import FUNMANConfig
from funman.scenario.scenario import AnalysisScenario


class SearchStatistics(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    _multiprocessing: bool = False
    _num_true: int = 0
    _num_false: int = 0
    _num_unknown: int = 0
    _residuals: SQueue = None
    _current_residual: float = 0.0
    _last_time: List[datetime.datetime] = []
    _iteration_time: SQueue = None
    _iteration_operation: SQueue = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self._residuals = SQueue()
        self._iteration_time = SQueue()
        self._iteration_operation = SQueue()


class SearchStaticsMP(SearchStatistics):
    _multiprocessing: bool = True
    _num_true: Value = 0
    _num_false: Value = 0
    _num_unknown: Value = 0
    _residuals: Queue = None
    _current_residual: Value = 0.0
    _last_time: Array = None
    _iteration_time: Queue = None
    _iteration_operation: Queue = None

    def __init__(self, **kw):
        super().__init__(**kw)

        manager = kw["manager"]
        self._multiprocessing = manager is not None
        self._num_true = manager.Value("i", 0)
        self._num_false = manager.Value("i", 0)
        self._num_unknown = manager.Value("i", 0)
        self._residuals = manager.Queue()
        self._current_residual = manager.Value("d", 0.0)
        self._last_time = manager.Array("u", "")
        self._iteration_time = manager.Queue()
        self._iteration_operation = manager.Queue()


class Search(ABC):
    def __init__(self) -> None:
        self.episodes = []

    @abstractmethod
    def search(
        self,
        problem: "AnalysisScenario",
        config: Optional["FUNMANConfig"] = None,
    ) -> SearchEpisode:
        pass

    def _initialize_encoding(self, solver: Solver, episode: BoxSearchEpisode):
        """
        The formula encoding the model M is of the form:

        AM <=> M

        where AM is a symbol denoting whether we assume M is true.  With this formula we can push/pop AM or Not(AM) to assert M or Not(M) without popping M.  Similarly we also assert the query as:

        AQ <==> Q

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        solver.push(1)
        formula = And(
            episode.problem._model_encoding._formula,
            episode.problem._query_encoding._formula,
        )
        episode._formula_stack.append(formula)
        solver.add_assertion(formula)
