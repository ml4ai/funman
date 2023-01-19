"""
This module defines the BoxSearch class and supporting classes.

"""
import logging
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Condition, Event, Lock
from queue import Empty
from queue import Queue as QueueSP
from queue import Queue as SQueue
from typing import List, Optional, Set, Union

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Not, Solver, get_model

from funman.scenario import AnalysisScenario
from funman.search import (
    Box,
    ParameterSpace,
    Point,
    ResultHandler,
    Search,
    SearchConfig,
    SearchEpisode,
    SearchStatistics,
)

LOG_LEVEL = logging.INFO

l = logging.getLogger(__file__)
l.setLevel(LOG_LEVEL)


class BoxSearchEpisode(SearchEpisode):
    """
    A BoxSearchEpisode stores the data required to organize a BoxSearch, including intermediate data and results. It takes as input:

    * config: a SearchConfig object that defines the configuration

    * problem: a ParameterSynthesisScenario to solve

    * manager: a SyncManager for coordinating multiple search processes that is used to declare several shared data

    A BoxSearchEpisode mainly tracks the true, false, and unknown boxes generated by the BoxSearch, along with statistics on the search.
    """

    def __init__(
        self,
        config: SearchConfig,
        problem: "ParameterSynthesisScenario",
        manager: Optional[SyncManager] = None,
    ) -> None:
        super(BoxSearchEpisode, self).__init__(config, problem)
        self.statistics = SearchStatistics.from_manager(manager)
        self.unknown_boxes = manager.Queue() if manager else SQueue()
        self.true_boxes: List[Box] = []
        self.false_boxes: List[Box] = []
        self.true_points: Set[Point] = set({})
        self.false_points: Set[Point] = set({})

        self.iteration = manager.Value("i", 0) if manager else 0

    def _initialize_boxes(self, expander_count):
        initial_box = self._initial_box()
        # if not self.add_unknown(initial_box):
        #     l.exception(
        #         f"Did not add an initial box (of width {initial_box.width()}), try reducing config.tolerance, currently {self.config.tolerance}"
        #     )
        initial_boxes = SQueue()
        initial_boxes.put(self._initial_box())
        num_boxes = 1
        while num_boxes < expander_count:
            b1, b2 = initial_boxes.get().split()
            initial_boxes.put(b1)
            initial_boxes.put(b2)
            num_boxes += 1
        for i in range(num_boxes):
            b = initial_boxes.get()
            if not self._add_unknown(b):
                l.exception(
                    f"Did not add an initial box (of width {initial_box.width()}), try reducing config.tolerance, currently {self.config.tolerance}"
                )
            l.debug(f"Initial box: {b}")

    def _initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def _on_start(self):
        if self.config.number_of_processes > 1:
            self.statistics._last_time.value = str(datetime.now())
        else:
            self.statistics._last_time = str(datetime.now())

    # def close(self):
    #     if self.multiprocessing:
    #         self.unknown_boxes.close()
    #         self.statistics.close()
    #         self.boxes_to_plot.close()

    def _on_iteration(self):
        if self.config.number_of_processes > 1:
            self.iteration.value = self.iteration.value + 1
        else:
            self.iteration = self.iteration + 1

    def _add_unknown_box(self, box: Box) -> bool:
        if box.width() > self.config.tolerance:
            self.unknown_boxes.put(box)
            if self.config.number_of_processes > 1:
                self.statistics._num_unknown.value += 1
            else:
                self.statistics._num_unknown += 1
            return True
        return False

    def _add_unknown(self, box: Union[Box, List[Box]]):
        did_add = False
        if isinstance(box, list):
            for b in box:
                did_add |= self._add_unknown_box(b)
        else:
            did_add = self._add_unknown_box(box)
        return did_add

    def _add_false(self, box: Box):
        self.false_boxes.append(box)
        # with self.statistics.num_false.get_lock():
        #     self.statistics.num_false.value += 1
        # self.statistics.iteration_operation.put("f")

    def _add_false_point(self, point: Point):
        if point in self.true_points:
            l.error(
                f"Point: {point} is marked false, but already marked true."
            )
        self.false_points.add(point)

    def _add_true(self, box: Box):
        self.true_boxes.append(box)
        # with self.statistics.num_true.get_lock():
        #     self.statistics.num_true.value += 1
        # self.statistics.iteration_operation.put("t")

    def _add_true_point(self, point: Point):
        if point in self.false_points:
            l.error(
                f"Point: {point} is marked true, but already marked false."
            )
        self.true_points.add(point)

    def _get_unknown(self):
        box = self.unknown_boxes.get(timeout=self.config.queue_timeout)
        if self.config.number_of_processes > 1:
            self.statistics._num_unknown.value = (
                self.statistics._num_unknown.value - 1
            )
            self.statistics._current_residual.value = box.width()
        else:
            self.statistics._num_unknown += 1
            self.statistics._current_residual = box.width()
        self.statistics._residuals.put(box.width())
        this_time = datetime.now()
        # FIXME self.statistics.iteration_time.put(this_time - self.statistics.last_time.value)
        # FIXME self.statistics.last_time[:] = str(this_time)
        return box

    def _get_box_to_plot(self):
        return self.boxes_to_plot.get(timeout=self.config.queue_timeout)

    def _extract_point(self, model):
        point = Point(self.problem.parameters)
        for p in self.problem.parameters:
            point.values[p] = float(model[p._symbol()].constant_value())
        return point


class BoxSearch(Search):
    """
    Box search algorithm.
    """

    def _split(self, box: Box, episode: BoxSearchEpisode, points=None):
        b1, b2 = box.split(points=points)
        episode.statistics._iteration_operation.put("s")
        return episode._add_unknown([b1, b2])

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
        solver.add_assertion(episode.problem.model_encoding.formula)
        solver.add_assertion(episode.problem.query_encoding.formula)

    def _initialize_box(self, solver, box):
        solver.push(1)
        solver.add_assertion(box.to_smt())

    def _setup_false_query(self, solver, episode):
        """
        Setup the assumptions so that satisfying the formulas requires that  either the model or the query is false

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        solver.push(1)
        solver.add_assertion(
            Not(
                And(episode.problem.assume_model, episode.problem.assume_query)
            )
        )

    def _setup_true_query(self, solver, episode):
        """
        Setup the assumptions so that satisfying the formulas requires that both the model and the query are true

        Parameters
        ----------
        solver : Solver
            pysmt solver object
        episode : episode
            data for the current search
        """
        solver.push(1)
        solver.add_assertion(
            And(episode.problem.assume_model, episode.problem.assume_query)
        )

    def _get_false_points(self, solver, episode, box, rval):
        false_points = [
            fp for fp in episode.false_points if box.contains_point(fp)
        ]
        if len(false_points) == 0:
            # If no cached point, then attempt to generate one
            # print("Checking false query")
            self._setup_false_query(solver, episode)
            if solver.solve():
                # Record the false point
                res = solver.get_model()
                false_points = [episode._extract_point(res)]
                for point in false_points:
                    episode._add_false_point(point)
                    rval.put(Point.encode_false_point(point))
            solver.pop(1)  # Remove false query
        return false_points

    def _get_true_points(self, solver, episode, box, rval):
        true_points = [
            tp for tp in episode.true_points if box.contains_point(tp)
        ]
        if len(true_points) == 0:
            # If no cached point, then attempt to generate one
            # print("Checking true query")
            self._setup_true_query(solver, episode)
            if solver.solve():
                # Record the true point
                res1 = solver.get_model()
                true_points = [episode._extract_point(res1)]
                for point in true_points:
                    episode._add_true_point(point)
                    rval.put(Point.encode_true_point(point))
                solver.pop(1)  # Remove true query
        return true_points

    def _expand(
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
                self._initialize_encoding(solver, episode)
                print("Initialized dynamics of model")
                while True:
                    try:
                        box: Box = episode._get_unknown()
                        rval.put(Box._encode_unknown_box(box))
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
                        self._initialize_box(solver, box)

                        # Check whether box intersects t (true region)
                        # First see if a cached false point exists in the box
                        true_points = self._get_true_points(
                            solver, episode, box, rval
                        )

                        if len(true_points) > 0:
                            # box intersects f (true region)

                            # Check whether box intersects f (false region)
                            # First see if a cached false point exists in the box
                            false_points = self._get_false_points(
                                solver, episode, box, rval
                            )

                            if len(false_points) > 0:
                                # box intersects f (false region)

                                # box intersects both t and f, so it must be split
                                # use the true and false points to compute a midpoint
                                if self._split(
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
                                episode._add_true(box)
                                rval.put(Box._encode_true_box(box))
                                print(f"+++ True({box})")
                        else:
                            # box is a subset of f (intersects f but not t)
                            episode._add_false(
                                box
                            )  # TODO consider merging lists of boxes
                            rval.put(Box._encode_false_box(box))
                            print(f"--- False({box})")
                        solver.pop(1)  # Remove box from solver
                        episode._on_iteration()
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
                    (
                        (inst, label),
                        typ,
                    ) = ParameterSpace.decode_labeled_object(result)
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
                        (
                            (inst, label),
                            typ,
                        ) = ParameterSpace.decode_labeled_object(result)
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

    def search(
        self, problem: AnalysisScenario, config: Optional[SearchConfig] = None
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
        if config is None:
            config = SearchConfig()

        # problem.encode()

        if config.number_of_processes > 1:
            return self._search_mp(problem, config)
        else:
            return self._search_sp(problem, config)

    def _search_sp(self, problem, config: SearchConfig):
        episode = BoxSearchEpisode(config, problem)
        episode._initialize_boxes(1)
        rval = QueueSP()
        all_results = {
            "true_boxes": [],
            "false_boxes": [],
            "true_points": [],
            "false_points": [],
        }
        config.handler.open()
        self._expand(
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
            episode._initialize_boxes(expand_count)
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
                    self._expand,
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
