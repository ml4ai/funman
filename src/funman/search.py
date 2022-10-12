from typing import Dict, List, Union
from funman.plotting import BoxPlotter
from funman.search_episode import BoxSearchEpisode
from funman.search_utils import Box, Point, SearchConfig
from pysmt.shortcuts import get_model, And, Not


from multiprocessing import Queue, Process, Value
from queue import Empty
import logging
import json

from funman.model import Parameter

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


class BoxSearch(object):
    def __init__(self) -> None:
        self.episodes = []
        self.real_time_plotting = True
        self.write_cache_parameter_space = None,
        self.read_cache_parameter_space = None

    def split(self, box: Box, episode: BoxSearchEpisode):
        b1, b2 = box.split()
        episode.add_unknown([b1, b2])
        episode.statistics.iteration_operation.put("s")

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

        while True:
            try:
                box = episode.get_unknown()
            except Empty:
                break
            else:
                # Check whether box intersects f (false region)
                false_point = None
                try:
                    # First see if a cached false point exists in the box
                    false_point = next(fp for fp in episode.false_points if box.contains_point(fp))
                except StopIteration:
                    # If no cached point, then attempt to generate one
                    phi = And(box.to_smt(), episode.problem.model.formula, Not(episode.problem.query.formula))
                    res = get_model(phi)
                    # Record the false point
                    if res:
                        false_point = episode.extract_point(res)
                        episode.add_false_point(false_point)

                if false_point:
                    # box intersects f (false region)                   

                    # Check whether box intersects t (true region)
                    true_point = None
                    try:
                        # First see if a cached false point exists in the box
                        true_point = next(tp for tp in episode.true_points if box.contains_point(tp))
                    except StopIteration:
                        # If no cached point, then attempt to generate one
                        phi1 = And(box.to_smt(), episode.problem.model.formula, episode.problem.query.formula)
                        res1 = get_model(phi1)
                        # Record the true point
                        if res1:
                            true_point = episode.extract_point(res1)
                            episode.add_true_point(true_point)                   
                
                    if true_point:
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

    def _write_to_cache(self, cache, region):
        if "box" in region:
            box = region['box']
            if region["label"] == "true":
                cache.write(json.dumps({
                    "label": "true",
                    "type": "box",
                    "value": box.to_dict(),
                }))
                cache.write("\n")
            elif region["label"] == "false":
                cache.write(json.dumps({
                    "label": "false",
                    "type": "box",
                    "value": box.to_dict(),
                }))
                cache.write("\n")
            elif region["label"] == "unknown":
                cache.write(json.dumps({
                    "label": "unknown",
                    "type": "box",
                    "value": box.to_dict(),
                }))
                cache.write("\n")
            else:
                raise Exception("Invalid label")
        elif "point" in region:
            point = region['point']
            if region["label"] == "true":
                cache.write(json.dumps({
                    "label": "true",
                    "type": "point",
                    "value": point.to_dict(),
                }))
                cache.write("\n")
            elif region["label"] == "false":
                cache.write(json.dumps({
                    "label": "false",
                    "type": "point",
                    "value": point.to_dict(),
                }))
                cache.write("\n")
            else:
                raise Exception("Invalid label")
        else:
            raise Exception("Unknown type")

    def _load_from_cache(self, cache_path, episode: BoxSearchEpisode):
        with open(cache_path) as f:
            for line in f.readlines():
                if len(line) == 0:
                    continue
                region = json.loads(line)
                if region["type"] == "box":
                    box = Box.from_dict(region["value"])
                    if region["label"] == "true":
                        episode.add_true(box)
                    elif region["label"] == "false":
                        episode.add_false(box)
                    elif region["label"] == "unknown":
                        episode.add_unknown(box)
                    else:
                        raise Exception("Invalid label")
                elif region["type"] == "point":
                    point = Point.from_dict(region["value"])
                    if region["label"] == "true":
                        episode.add_true_point(point)
                    elif region["label"] == "false":
                        episode.add_false_point(point)
                    else:
                        raise Exception("Invalid label")
                else:
                    raise Exception("Unknown type")


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

        if self.read_cache_parameter_space is not None:
            self._load_from_cache(self.read_cache_parameter_space, episode)

        write_region_to_cache = None
        out_cache = None
        if self.write_cache_parameter_space is not None:
            out_cache = open(self.write_cache_parameter_space, 'w')
            write_region_to_cache = lambda x: self._write_to_cache(out_cache, x)

        # create a plotting process
        plotter = BoxPlotter(problem.parameters,
            real_time_plotting=self.real_time_plotting,
            write_region_to_cache=write_region_to_cache)
        p = Process(
            target=plotter.run,
            args=(
                rval,
                episode,
            ),
        )
        processes.append(p)
        p.start()

        # short circuit since reading from cache
        if self.read_cache_parameter_space is not None:
            rval.close()
            p.terminate()
            p.join()
            # results = {"true_boxes": episode.true_boxes, "false_boxes": episode.false_boxes}
            return episode

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

        episode.close() # Main process needs to close queues used to initialize

        results = [rval.get() for p in processes]
        rval.close()

        

        for p in processes:
            p.terminate()

        # completing process
        for p in processes:
            p.join()

        episode.false_boxes = [b for r in results for b in r["false_boxes"]]
        episode.true_boxes = [b for r in results for b in r["true_boxes"]]

        if out_cache is not None:
            out_cache.close()

        return episode
