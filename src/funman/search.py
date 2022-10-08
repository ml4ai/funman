from typing import Dict, List, Union
from funman.plotting import BoxPlotter
from funman.search_episode import BoxSearchEpisode
from funman.search_utils import Box, SearchConfig
from pysmt.shortcuts import get_model, And, Not


from multiprocessing import Queue, Process, Value
from queue import Empty
import logging

from funman.model import Parameter

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class BoxSearch(object):
    def __init__(self) -> None:

        self.episodes = []

    def split(self, box: Box, episode: BoxSearchEpisode):
        b1, b2 = box.split()
        episode.add_unknown([b1, b2])
        episode.statistics.iteration_operation.put("s")

    def initialize(self, episode: BoxSearchEpisode):
        initial_boxes = Queue()
        initial_boxes.put(episode.initial_box())
        num_boxes = 1
        while num_boxes < episode.config.number_of_processes:
            b1, b2 = initial_boxes.get().split()
            initial_boxes.put(b1)
            initial_boxes.put(b2)
            num_boxes += 1
        for i in range(num_boxes):
            b = initial_boxes.get()
            episode.add_unknown(b)

    def expand(self, rval: Queue, episode: BoxSearchEpisode) -> None:
        """
        A single search process will evaluate and expand the boxes in the episode.unknown_boxes queue.  The processes exit when the queue is empty.  For each box, the algorithm checks whether the box contains a false (infeasible) point.  If it contains a false point, then it checks if the box contains a true point.  If a box contains both a false and true point, then the box is split into two boxes and both are added to the unknown_boxes queue.  If a box contains no false points, then it is a true_box (all points are feasible).  If a box contains no true points, then it is a false_box (all points are infeasible).

        The return value is pushed onto the rval queue to end the process's work within the method.  The return value is a Dict[str, List[Box]] type that maps the "true_boxes" and "false_boxes" to a list of boxes in each set.  Each box in these sets is unique by design.

        Parameters
        ----------
        rval : Queue
            Return value shared queue
        episode : BoxSearchEpisode
            Shared search data and statistics.
        """
        if episode.internal_process_id == 0:
            self.initialize(episode)

        while True:
            try:
                box = episode.get_unknown()
            except Empty:
                break
            else:
                # Check whether box intersects f (false region)
                phi = And(box.to_smt(), Not(episode.problem.model.formula))
                res = get_model(phi)
                if res:
                    # box intersects f (false region)

                    # Check whether box intersects t (true region)
                    phi1 = And(box.to_smt(), episode.problem.model.formula)
                    res1 = get_model(phi1)
                    if res1:
                        # box intersects both t and f, so it must be split
                        self.split(box, episode)
                    else:
                        # box is a subset of f (intersects f but not t)
                        episode.add_false(box)  # TODO consider merging lists of boxes

                else:
                    # box does not intersect f, so it is in t (true region)
                    episode.add_true(box)
                episode.on_iteration()
        episode.close()
        rval.put({"true_boxes": episode.true_boxes, "false_boxes": episode.false_boxes})

    def search(
        self, problem, config: SearchConfig = SearchConfig()
    ) -> BoxSearchEpisode:
        """
        The BoxSearch.search() creates a BoxSearchEpisode object that stores the
        search progress.  This method is the entry point to the search that
        spawns several processes to parallelize the evaluation of boxes in the
        BoxSearch.expand() method.  It treats the zeroth process as a special
        process that is allowed to initialize the search and plot the progress
        of the search.

        Parameters
        ----------
        problem : ParameterSynthesisScenario
            Model and parameters to synthesize
        config : SearchConfig, optional
            BoxSearch configuration, by default SearchConfig()

        Returns
        -------
        BoxSearchEpisode
            Final search results (parameter space) and statistics.
        """

        episode = BoxSearchEpisode(config, problem)
        self.episodes.append(episode)
        episode.on_start()

        processes = []

        rval = Queue()

        # create a plotting process
        plotter = BoxPlotter(problem.parameters)
        p = Process(
            target=plotter.run,
            args=(
                rval,
                episode,
            ),
        )
        processes.append(p)
        p.start()

        # creating processes
        for w in range(episode.config.number_of_processes - 1):
            p = Process(
                target=self.expand,
                args=(
                    rval,
                    episode,
                ),
            )
            processes.append(p)
            p.start()
            episode.internal_process_id = w

        results = [rval.get() for p in processes]
        rval.close()

        for p in processes:
            p.terminate()

        # completing process
        for p in processes:
            p.join()

        episode.false_boxes = [b for r in results for b in r["false_boxes"]]
        episode.true_boxes = [b for r in results for b in r["true_boxes"]]

        return episode
