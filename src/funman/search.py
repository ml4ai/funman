"""
This submodule contains the search algorithms used to run FUNMAN.
"""
import traceback
from typing import List
from funman.search_episode import BoxSearchEpisode, SearchEpisode
from funman.search_utils import Box, Point, SearchConfig, ResultHandler, \
    decode_labeled_object, \
    encode_true_box, \
    encode_false_box, \
    encode_true_point, \
    encode_false_point

from pysmt.shortcuts import get_model, And, Not

import multiprocessing as mp
from multiprocessing.synchronize import Condition, Event, Lock

from queue import Empty
import logging
import os

LOG_LEVEL = logging.WARN

class BoxSearch(object):
    def __init__(self) -> None:
        self.episodes = []

    def split(self, box: Box, episode: BoxSearchEpisode, points=None):
        b1, b2 = box.split(points=points)
        episode.statistics.iteration_operation.put("s")
        return episode.add_unknown([b1, b2])

    def expand(self, rval: mp.Queue, episode: BoxSearchEpisode, id: int, more_work: Condition, idle_mutex: Lock, idle_flags: List[Event]):
        """
        A single search process will evaluate and expand the boxes in the
        episode.unknown_boxes queue.  The processes exit when the queue is
        empty.  For each box, the algorithm checks whether the box contains a
        false (infeasible) point.  If it contains a false point, then it checks
        if the box contains a true point.  If a box contains both a false and
        true point, then the box is split into two boxes and both are added to
        the unknown_boxes queue.  If a box contains no false points, then it is
        a true_box (all points are feasible).  If a box contains no true points,
        then it is a false_box (all points are infeasible).
        
        The return value is pushed onto the rval queue to end the process's work
        within the method.  The return value is a Dict[str, List[Box]] type that
        maps the "true_boxes" and "false_boxes" to a list of boxes in each set.
        Each box in these sets is unique by design.

        Parameters
        ----------
        rval : Queue
            Return value shared queue
        episode : BoxSearchEpisode
            Shared search data and statistics.
        """
        process_name = f"Expander_{id}_p{os.getpid()}"
        l = mp.log_to_stderr()
        l.name = process_name
        l.setLevel(LOG_LEVEL)
        
        try:
            l.info(f"{process_name} entering process loop")
            while True:
                try:
                    box: Box = episode.get_unknown()
                    l.info(f"{process_name} claimed work")
                except Empty:
                    # set this processes idle flag and check all the other process idle flags
                    # doing so under the idle_mutex to ensure the flags are not under active change
                    with idle_mutex:
                        idle_flags[id].set()
                        should_exit = all(f.is_set() for f in idle_flags)

                    # all worker processes appear to be idle
                    if should_exit:
                        # one last check to see if there is work to be done
                        # which would be an error at this point in the code
                        if episode.unknown_boxes.qsize() != 0:
                            l.error(f"{process_name} found more work while preparing to exit")
                            continue
                        l.info(f"{process_name} is exiting")
                        # tell other worker processing to check again with the expectation
                        # that they will also resolve to exit
                        with more_work:
                            more_work.notify()
                        # break of the while True and allow the process to exit
                        break

                    # wait for notification of more work
                    l.info(f"{process_name} is awaiting work")
                    with more_work:
                        more_work.wait()

                    # clear the current processes idle flag under the idle_mutex
                    with idle_mutex:
                        idle_flags[id].clear()

                    continue
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
                            rval.put(encode_false_point(false_point))

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
                                rval.put(encode_true_point(true_point))
                    
                        if true_point:
                            # box intersects both t and f, so it must be split
                            # use the true and false points to compute a midpoint
                            if self.split(box, episode, points=[true_point, false_point]):
                                l.info(f"{process_name} produced work")
                            with more_work:
                                more_work.notify_all()
                        else:
                            # box is a subset of f (intersects f but not t)
                            episode.add_false(box)  # TODO consider merging lists of boxes
                            rval.put(encode_false_box(box))

                    else:
                        # box does not intersect f, so it is in t (true region)
                        episode.add_true(box)
                        rval.put(encode_true_box(box))
                    episode.on_iteration()
                    l.info(f"{process_name} finished work")
        except KeyboardInterrupt:
            l.info(f"{process_name} Keyboard Interrupt")
        except Exception:
            l.error(traceback.format_exc())

    def _run_handler(self, rval: mp.Queue, config: SearchConfig):
        """
        Execute the process that does final processing of the results of expand()
        """
        l = mp.log_to_stderr()
        l.name = f"search_process_result_handler"
        l.setLevel(LOG_LEVEL)

        handler: ResultHandler = config.handler
        true_boxes = []
        false_boxes = []
        true_points = []
        false_points = []
        break_on_interrupt = False
        try:
            handler.open()
            while True:
                try:
                    result: dict = rval.get(timeout=config.queue_timeout)
                except Empty:
                    continue
                except KeyboardInterrupt:
                    if break_on_interrupt:
                        break
                    break_on_interrupt = True
                else:
                    if result is None:
                        break

                    # TODO this is a bit of a mess and can likely be cleaned up
                    ((inst, label), typ) = decode_labeled_object(result)
                    if typ is Box:
                        if label == "true":
                            true_boxes.append(inst)
                        elif label == "false":
                            false_boxes.append(inst)
                        else:
                            l.warn(f"Skipping Box with label: {label}")
                    elif typ is Point:
                        if label == "true":
                            true_points.append(inst)
                        elif label == "false":
                            false_points.append(inst)
                        else:
                            l.warn(f"Skipping Point with label: {label}")
                    else:
                        l.error(f"Skipping invalid object type: {typ}")

                    try:
                        handler.process(result)
                    except Exception:
                        l.error(traceback.format_exc())

        except Exception as error:
            l.error(error)
        finally:
            handler.close()
        return {
            "true_boxes": true_boxes,
            "false_boxes": false_boxes,
            "true_points": true_points,
            "false_points": false_points 
        }

    def search(
        self, problem, config: SearchConfig = None
    ) -> SearchEpisode:
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
        l = mp.get_logger()
        l.setLevel(LOG_LEVEL)

        if config is None:
            config = SearchConfig()

        processes = config.number_of_processes
        with mp.Manager() as manager:
            rval = manager.Queue()
            episode = BoxSearchEpisode(config, problem, manager)
            
            expand_count = processes - 1
            episode.initialize_boxes(expand_count)
            idle_mutex = manager.Lock()
            idle_flags = [manager.Event() for _ in range(expand_count)]

            with mp.Pool(processes=processes) as pool:
                more_work_condition = manager.Condition()

                # start the result handler process
                l.info("Starting result handler process")
                rval_handler_process = pool.apply_async(self._run_handler, args=(rval, config))
                # blocking exec of the expansion processes
                l.info(f"Starting {expand_count} expand processes")
                starmap_result = pool.starmap_async(self.expand, [(rval, episode, id, more_work_condition, idle_mutex, idle_flags) for id in range(expand_count)])

                # tell the result handler process we are done with the expansion processes
                try:
                    if config.wait_action is not None:
                        while not starmap_result.ready():
                            config.wait_action.run()
                    starmap_result.wait()
                except KeyboardInterrupt:
                    l.warning("--- Received Keyboard Interrupt ---")

                rval.put(None)
                l.info("Waiting for result handler process")
                # wait for the result handler to finish
                rval_handler_process.wait(timeout=config.wait_timeout)

                if not rval_handler_process.successful():
                    l.error("Result handler failed to exit")
                all_results = rval_handler_process.get()

                episode.true_boxes = all_results.get("true_boxes")
                episode.false_boxes = all_results.get("false_boxes")
                episode.true_points = all_results.get("true_points")
                episode.false_points = all_results.get("false_points")
                return episode
 