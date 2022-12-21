import logging
import multiprocessing as mp
import os
import traceback
from multiprocessing.synchronize import Condition, Event, Lock
from queue import Empty
from queue import Queue as QueueSP
from typing import List, Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Not, Solver, get_model

from funman.search_episode import BoxSearchEpisode
from funman.utils.search_utils import (
    Box,
    Point,
    ResultHandler,
    SearchConfig,
    decode_labeled_object,
    encode_false_box,
    encode_false_point,
    encode_true_box,
    encode_true_point,
    encode_unknown_box,
)

from .search import Search

LOG_LEVEL = logging.WARN


class BoxSearch(Search):
    def split(self, box: Box, episode: BoxSearchEpisode, points=None):
        b1, b2 = box.split(points=points)
        episode.statistics.iteration_operation.put("s")
        return episode.add_unknown([b1, b2])

    def _logger(self, config, process_name=None):
        if config.number_of_processes > 1:
            l = mp.log_to_stderr()
            if process_name:
                l.name = process_name
            l.setLevel(LOG_LEVEL)
        else:
            if not process_name:
                process_name = "BoxSearch"
            l = logging.Logger(process_name)
        return l

    def _handle_empty_queue(
        self, process_name, episode, more_work, idle_mutex, idle_flags
    ):
        if episode.config.number_of_processes > 1:
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
                    l.error(
                        f"{process_name} found more work while preparing to exit"
                    )
                    return False
                l.info(f"{process_name} is exiting")
                # tell other worker processing to check again with the expectation
                # that they will also resolve to exit
                with more_work:
                    more_work.notify()
                # break of the while True and allow the process to exit
                return True

            # wait for notification of more work
            l.info(f"{process_name} is awaiting work")
            with more_work:
                more_work.wait()

            # clear the current processes idle flag under the idle_mutex
            with idle_mutex:
                idle_flags[id].clear()

            return False
        else:
            return True

    def initialize_encoding(self, solver, episode):
        solver.push(1)
        solver.add_assertion(episode.problem.model_encoding.formula)

    def initialize_box(self, solver, box):
        solver.push(1)
        solver.add_assertion(box.to_smt())

    def setup_false_query(self, solver, episode):
        solver.push(1)
        solver.add_assertion(Not(episode.problem.query_encoding.formula))

    def setup_true_query(self, solver, episode):
        solver.push(1)
        solver.add_assertion(episode.problem.query_encoding.formula)

    def get_false_points(self, solver, episode, box, rval):
        false_points = [
            fp for fp in episode.false_points if box.contains_point(fp)
        ]
        if len(false_points) == 0:
            # If no cached point, then attempt to generate one
            # print("Checking false query")
            self.setup_false_query(solver, episode)
            if solver.solve():
                # Record the false point
                res = solver.get_model()
                false_points = [episode.extract_point(res)]
                map(episode.add_false_point, false_points)
                map(
                    lambda x: rval.put(encode_false_point(x)),
                    false_points,
                )
            solver.pop(1)  # Remove false query
        return false_points

    def get_true_points(self, solver, episode, box, rval):
        true_points = [
            tp for tp in episode.true_points if box.contains_point(tp)
        ]
        if len(true_points) == 0:
            # If no cached point, then attempt to generate one
            # print("Checking true query")
            self.setup_true_query(solver, episode)
            if solver.solve():
                # Record the true point
                res1 = solver.get_model()
                true_points = [episode.extract_point(res1)]
                map(episode.add_true_point, true_points)
                map(
                    lambda x: rval.put(encode_true_point(x)),
                    true_points,
                )
            solver.pop(1)  # Remove true query
        return true_points

    def expand(
        self,
        rval,
        episode: BoxSearchEpisode,
        idx: Optional[int] = None,
        more_work: Optional[Condition] = None,
        idle_mutex: Optional[Lock] = None,
        idle_flags: Optional[List[Event]] = None,
        handler: Optional[ResultHandler] = None,
        all_results=None,
    ):
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
        process_name = f"Expander_{idx}_p{os.getpid()}"
        l = self._logger(episode.config, process_name=process_name)

        try:
            with Solver(name=episode.config.solver, logic=QF_NRA) as solver:
                l.info(f"{process_name} entering process loop")
                print("Starting initializing dynamics of model")
                self.initialize_encoding(solver, episode)
                print("Initialized dynamics of model")
                while True:
                    try:
                        box: Box = episode.get_unknown()
                        rval.put(encode_unknown_box(box))
                        l.info(f"{process_name} claimed work")
                    except Empty:
                        exit = self._handle_empty_queue(
                            process_name,
                            episode,
                            more_work,
                            idle_mutex,
                            idle_flags,
                        )
                        if exit:
                            break
                        else:
                            continue
                    else:
                        self.initialize_box(solver, box)

                        # Check whether box intersects t (true region)
                        # First see if a cached false point exists in the box
                        true_points = self.get_true_points(
                            solver, episode, box, rval
                        )

                        if len(true_points) > 0:
                            # box intersects f (true region)

                            # Check whether box intersects f (false region)
                            # First see if a cached false point exists in the box
                            false_points = self.get_false_points(
                                solver, episode, box, rval
                            )

                            if len(false_points) > 0:
                                # box intersects f (false region)

                                # box intersects both t and f, so it must be split
                                # use the true and false points to compute a midpoint
                                if self.split(
                                    box,
                                    episode,
                                    points=true_points + false_points,
                                ):
                                    l.info(f"{process_name} produced work")
                                if episode.config.number_of_processes > 1:
                                    # FIXME This would only be none when
                                    # the number of processes is 1. This
                                    # can be done more cleanly.
                                    if more_work:
                                        with more_work:
                                            more_work.notify_all()
                                print(f"Split({box})")
                            else:
                                # box does not intersect f, so it is in t (true region)
                                episode.add_true(box)
                                rval.put(encode_true_box(box))
                                print(f"+++ True({box})")
                        else:
                            # box is a subset of f (intersects f but not t)
                            episode.add_false(
                                box
                            )  # TODO consider merging lists of boxes
                            rval.put(encode_false_box(box))
                            print(f"--- False({box})")
                        solver.pop(1)  # Remove box from solver
                        episode.on_iteration()
                        if handler:
                            all_results = handler(
                                rval, episode.config, all_results
                            )
                        l.info(f"{process_name} finished work")
                solver.pop(1)  # Remove the dynamics from the solver
        except KeyboardInterrupt:
            l.info(f"{process_name} Keyboard Interrupt")
        except Exception:
            l.error(traceback.format_exc())

    def _run_handler(self, rval, config: SearchConfig):
        """
        Execute the process that does final processing of the results of expand()
        """
        l = self._logger(config, process_name=f"search_process_result_handler")

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
            "false_points": false_points,
        }

    def _run_handler_step(self, rval, config: SearchConfig, all_results):
        """
        Execute one step of processing the results of expand()
        """
        l = self._logger(config, process_name=f"search_process_result_handler")

        handler: ResultHandler = config.handler
        # true_boxes = []
        # false_boxes = []
        # true_points = []
        # false_points = []
        break_on_interrupt = False
        try:
            # handler.open()
            while True:
                try:
                    result: dict = rval.get(timeout=config.queue_timeout)
                except Empty:
                    break
                except KeyboardInterrupt:
                    if break_on_interrupt:
                        break
                    break_on_interrupt = True
                else:
                    if result is None:
                        break

                    # TODO this is a bit of a mess and can likely be cleaned up
                    try:
                        ((inst, label), typ) = decode_labeled_object(result)
                    except:
                        l.error(f"Skipping invalid object")
                        continue

                    if typ is Box:
                        if label == "true":
                            all_results["true_boxes"].append(inst)
                        elif label == "false":
                            all_results["false_boxes"].append(inst)
                        elif label == "unknown":
                            pass  # Allow unknown boxes for plotting
                        else:
                            l.warn(f"Skipping Box with label: {label}")
                    elif typ is Point:
                        if label == "true":
                            all_results["true_points"].append(inst)
                        elif label == "false":
                            all_results["false_points"].append(inst)
                        else:
                            l.warn(f"Skipping Point with label: {label}")
                    else:
                        l.error(f"Skipping invalid object type: {typ}")
                        continue

                    try:
                        handler.process(result)
                    except Exception:
                        l.error(traceback.format_exc())

        except Exception as error:
            l.error(error)
        finally:
            if config.wait_action is not None:
                config.wait_action.run()
            # handler.close()
        return all_results

    def search(self, problem, config: SearchConfig = None) -> BoxSearchEpisode:
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
        if config is None:
            config = SearchConfig()

        problem.encode()

        if config.number_of_processes > 1:
            return self._search_mp(problem, config)
        else:
            return self._search_sp(problem, config)

    def _search_sp(self, problem, config: SearchConfig):
        episode = BoxSearchEpisode(config, problem)
        episode.initialize_boxes(1)
        rval = QueueSP()
        all_results = {
            "true_boxes": [],
            "false_boxes": [],
            "true_points": [],
            "false_points": [],
        }
        config.handler.open()
        self.expand(
            rval,
            episode,
            handler=self._run_handler_step,
            all_results=all_results,
        )
        config.handler.close()
        # rval.put(None)

        # all_results = self._run_handler(rval, config)
        episode.true_boxes = all_results.get("true_boxes")
        episode.false_boxes = all_results.get("false_boxes")
        episode.true_points = all_results.get("true_points")
        episode.false_points = all_results.get("false_points")
        return episode

    def _search_mp(self, problem, config: SearchConfig):
        l = mp.get_logger()
        l.setLevel(LOG_LEVEL)
        processes = config.number_of_processes
        with mp.Manager() as manager:
            rval = manager.Queue()
            episode = BoxSearchEpisode(config, problem, manager=manager)

            expand_count = processes - 1
            episode.initialize_boxes(expand_count)
            idle_mutex = manager.Lock()
            idle_flags = [manager.Event() for _ in range(expand_count)]

            with mp.Pool(processes=processes) as pool:
                more_work_condition = manager.Condition()

                # start the result handler process
                l.info("Starting result handler process")
                rval_handler_process = pool.apply_async(
                    self._run_handler, args=(rval, config)
                )
                # blocking exec of the expansion processes
                l.info(f"Starting {expand_count} expand processes")

                starmap_result = pool.starmap_async(
                    self.expand,
                    [
                        (
                            rval,
                            episode,
                            {
                                "idx": idx,
                                "more_work": more_work_condition,
                                "idle_mutex": idle_mutex,
                                "idle_flags": idle_flags,
                            },
                        )
                        for idx in range(expand_count)
                    ],
                )

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


class DrealBoxSearch(BoxSearch):
    def __init__(self) -> None:
        super().__init__()

    def _push_formula(self, formula):
        episode.problem.model_encoding.formula

    def expand(
        self,
        rval,
        episode: BoxSearchEpisode,
        idx: int = None,
        more_work: Condition = None,
        idle_mutex: Lock = None,
        idle_flags: List[Event] = None,
        handler: ResultHandler = None,
    ):
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
        process_name = f"Expander_{idx}_p{os.getpid()}"
        l = self._logger(episode.config, process_name=process_name)

        try:
            l.info(f"{process_name} entering process loop")
            while True:
                try:
                    box: Box = episode.get_unknown()
                    rval.put(encode_unknown_box(box))
                    l.info(f"{process_name} claimed work")
                except Empty:
                    exit = self._handle_empty_queue(
                        process_name,
                        episode,
                        more_work,
                        idle_mutex,
                        idle_flags,
                    )
                    if exit:
                        break
                    else:
                        continue
                else:
                    # Check whether box intersects f (false region)
                    # First see if a cached false point exists in the box
                    false_points = [
                        fp
                        for fp in episode.false_points
                        if box.contains_point(fp)
                    ]
                    if len(false_points) == 0:
                        # If no cached point, then attempt to generate one
                        phi = And(
                            box.to_smt(),
                            episode.problem.model_encoding.formula,
                            Not(episode.problem.query_encoding.formula),
                        )
                        res = get_model(phi)
                        # Record the false point
                        if res:
                            false_points = [episode.extract_point(res)]
                            map(episode.add_false_point, false_points)
                            map(
                                lambda x: rval.put(encode_false_point(x)),
                                false_points,
                            )

                    if len(false_points) > 0:
                        # box intersects f (false region)

                        # Check whether box intersects t (true region)
                        # First see if a cached false point exists in the box
                        true_points = [
                            tp
                            for tp in episode.true_points
                            if box.contains_point(tp)
                        ]
                        if len(true_points) == 0:
                            # If no cached point, then attempt to generate one
                            phi1 = And(
                                box.to_smt(),
                                episode.problem.model_encoding.formula,
                                episode.problem.query_encoding.formula,
                            )
                            res1 = get_model(phi1)
                            # Record the true point
                            if res1:
                                true_points = [episode.extract_point(res1)]
                                map(episode.add_true_point, true_points)
                                map(
                                    lambda x: rval.put(encode_true_point(x)),
                                    true_points,
                                )

                        if len(true_points) > 0:
                            # box intersects both t and f, so it must be split
                            # use the true and false points to compute a midpoint
                            if self.split(
                                box, episode, points=true_points + false_points
                            ):
                                l.info(f"{process_name} produced work")
                            if episode.config.number_of_processes > 1:
                                with more_work:
                                    more_work.notify_all()
                        else:
                            # box is a subset of f (intersects f but not t)
                            episode.add_false(
                                box
                            )  # TODO consider merging lists of boxes
                            rval.put(encode_false_box(box))

                    else:
                        # box does not intersect f, so it is in t (true region)
                        episode.add_true(box)
                        rval.put(encode_true_box(box))
                    episode.on_iteration()
                    if handler:
                        handler(rval, episode.config)
                    l.info(f"{process_name} finished work")
        except KeyboardInterrupt:
            l.info(f"{process_name} Keyboard Interrupt")
        except Exception:
            l.error(traceback.format_exc())