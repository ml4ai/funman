"""
This submodule contains the search algorithms used to run FUNMAN.
"""
from abc import abstractmethod
from typing import Optional

from funman.search_episode import SearchEpisode
from funman.utils.search_utils import SearchConfig


class Search(object):
    def __init__(self) -> None:
        self.episodes = []

    @abstractmethod
    def search(
        self, problem, config: Optional[SearchConfig] = None
    ) -> SearchEpisode:
        pass
