import copy
import queue
import sys
import threading
import traceback
from enum import Enum
from typing import Optional

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model.model import Model
from funman.representation.representation import ParameterSpace
from funman.scenario.scenario import AnalysisScenario
from funman.server.exception import FunmanWorkerException
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)


class WorkerState(Enum):
    """
    States that FunmanWorker can be in
    """

    UNINITIALIZED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4
    ERRORED = 5


class FunmanWorker:
    _state: WorkerState = WorkerState.UNINITIALIZED

    def __init__(self, storage):
        self._halt_event = threading.Event()
        self._stop_event = None
        self._thread = None
        self._id_lock = threading.Lock()
        self._results_lock = threading.Lock()
        self._set_lock = threading.Lock()

        self.storage = storage
        self.queue = queue.Queue()
        self.queued_ids = set()
        self.current_id = None
        self.current_results = None

        # TODO consider changing to more robust state machine
        # instead of basic state field (if complexity increases)
        self._state_lock = threading.Lock()
        self._state = WorkerState.STOPPED

    def get_state(self) -> WorkerState:
        """
        Return the current state of the worker
        """
        with self._state_lock:
            return WorkerState(self._state)

    def in_state(self, state: WorkerState) -> bool:
        """
        Return true if in the provided state else false
        """
        with self._state_lock:
            return self._state == state

    def enqueue_work(
        self, model: Model, request: FunmanWorkRequest
    ) -> FunmanWorkUnit:
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException(
                f"FunmanWorker must be starting or running to enqueue work: {self.get_state()}"
            )
        id = self.storage.claim_id()
        work = FunmanWorkUnit(id=id, model=model, request=request)
        self.queue.put(work)
        with self._set_lock:
            self.queued_ids.add(work.id)
        return work

    def start(self):
        if not self.in_state(WorkerState.STOPPED):
            raise FunmanWorkerException(
                f"FunmanWorker must be stopped to start: {self.get_state()}"
            )
        with self._state_lock:
            self._state = WorkerState.STARTING
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=[self._stop_event], daemon=True
            )
            self._thread.start()
            self._state = WorkerState.RUNNING

    def stop(self, timeout=None):
        if not (
            self.in_state(WorkerState.RUNNING)
            or self.in_state(WorkerState.ERRORED)
        ):
            raise FunmanWorkerException(
                f"FunmanWorker be running to stop: {self.get_state()}"
            )
        # Grab the state lock for the entire process of stopping.
        with self._state_lock:
            # The worker is stopping
            self._state = WorkerState.STOPPING
            # Tell the work thread to stop
            self._stop_event.set()
            # Wait for the work thread to stop
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # TODO If the thread is still alive then it is likely the solver
                # is still processing and the thread will exit if/when the solver
                # returns. Ideally we could kill it here and abandon any state it
                # holds.
                print("Thread did not close")
            # Reset state
            self._thread = None
            self._stop_event = None
            # The worker is stopped
            self._state = WorkerState.STOPPED

    def is_processing_id(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException(
                f"FunmanWorker must be running to check processing id: {self.get_state()}"
            )
        with self._id_lock:
            return self.current_id == id

    def get_results(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException(
                f"FunmanWorker must be running to get results: {self.get_state()}"
            )
        with self._id_lock:
            if self.current_id == id:
                return copy.copy(self.current_results)
            return self.storage.get_result(id)

    def halt(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException(
                f"FunmanWorker must be running to halt request: {self.get_state()}"
            )
        with self._id_lock:
            if id == self.current_id:
                print(f"Halting {id}")
                self._halt_event.set()
                return
            with self._set_lock:
                if id in self.queued_ids:
                    self.queued_ids.remove(id)
                return

    def get_current(self) -> Optional[str]:
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException(
                f"FunmanWorker must be running to check currently processing request: {self.get_state()}"
            )
        with self._id_lock:
            return self.current_id

    def _update_current_results(
        self, scenario: AnalysisScenario, results: ParameterSpace
    ):
        with self._results_lock:
            if self.current_results is None:
                print(
                    "WARNING: Attempted to update results while current_results was None"
                )
                return

            if self.current_results.is_final():
                raise Exception(
                    "Cannot update current_results as it is already finalized"
                )
            self.current_results.update_parameter_space(scenario, results)

    def _run(self, stop_event: threading.Event):
        print("FunmanWorker running...")
        try:
            while True:
                if stop_event.is_set():
                    break
                try:
                    work: FunmanWorkUnit = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # skip work that is no longer in the queued_ids set
                # since that likely indicated it has been halted
                # before starting
                with self._set_lock:
                    if work.id not in self.queued_ids:
                        continue

                with self._id_lock:
                    self.current_id = work.id
                    self.current_results = FunmanResults(
                        id=work.id,
                        model=work.model,
                        request=work.request,
                        parameter_space=ParameterSpace(),
                    )

                print(f"Starting work on: {work.id}")
                try:
                    # convert to scenario
                    scenario = work.to_scenario()

                    config = (
                        FUNMANConfig()
                        if work.request.config is None
                        else work.request.config
                    )
                    f = Funman()
                    self._halt_event.clear()
                    result = f.solve(
                        scenario,
                        config=config,
                        haltEvent=self._halt_event,
                        resultsCallback=lambda results: self._update_current_results(
                            scenario, results
                        ),
                    )
                    with self._results_lock:
                        self.current_results.finalize_result(scenario, result)
                    print(f"Completed work on: {work.id}")
                except Exception as e:
                    print(
                        f"Internal Server Error ({work.id}):", file=sys.stderr
                    )
                    traceback.print_exc()
                    with self._results_lock:
                        self.current_results.finalize_result_as_error()
                    print(f"Aborting work on: {work.id}")
                finally:
                    self.storage.add_result(self.current_results)
                    self.queue.task_done()
                    with self._id_lock:
                        self.current_id = None
                        self.current_results = None
        except Exception:
            print("Fatal error in worker!", file=sys.stderr)
            traceback.print_exc()
            # Only mark the state as errored if the thread
            # has not yet been told to stop
            if not stop_event.is_set():
                with self._state_lock:
                    self._state = WorkerState.ERRORED
        print("FunmanWorker exiting...")
