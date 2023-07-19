import queue
import sys
import threading
import traceback
from enum import Enum
from typing import Optional

from funman import Funman
from funman.server.exception import FunmanWorkerException
from funman.funman import FUNMANConfig
from funman.model.model import Model
from funman.representation.representation import ParameterSpace
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
    EXITED = 3
    STOPPING = 4
    STOPPED = 5

class FunmanWorker:
    _state: WorkerState = WorkerState.UNINITIALIZED

    def __init__(self, storage):
        self._halt_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._id_lock = threading.Lock()
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

    def in_state(self, state: WorkerState) -> bool:
        """
        Return true if in the provided state else false
        """
        with self._state_lock:
            return self._state == state

    def set_state(self, state: WorkerState):
        """
        Change state
        """
        with self._state_lock:
            self._state = state

    def enqueue_work(
        self, model: Model, request: FunmanWorkRequest
    ) -> FunmanWorkUnit:
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException("FunmanWorker must be running to enqueue work")
        id = self.storage.claim_id()
        work = FunmanWorkUnit(id=id, model=model, request=request)
        self.queue.put(work)
        with self._set_lock:
            self.queued_ids.add(work.id)
        return work

    def start(self):
        if not self.in_state(WorkerState.STOPPED):
            raise FunmanWorkerException("FunmanWorker must be stopped to start")
        self.set_state(WorkerState.STARTING)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event.clear()
        self._thread.start()

    def stop(self, timeout=None):
        if not (self.in_state(WorkerState.RUNNING) or self.in_state(WorkerState.EXITED)):
            raise FunmanWorkerException("FunmanWorker be running to stop")
        self.set_state(WorkerState.STOPPING)
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            # TODO kill thread?
            print("Thread did not close")
        self._thread = None
        self.set_state(WorkerState.STOPPED)

    def is_processing_id(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException("FunmanWorker must be running to check processing id")
        with self._id_lock:
            return self.current_id == id

    def get_results(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException("FunmanWorker must be running to get results")
        if self.is_processing_id(id):
            return self.current_results
        return self.storage.get_result(id)

    def halt(self, id: str):
        if not self.in_state(WorkerState.RUNNING):
            raise FunmanWorkerException("FunmanWorker must be running to halt request")
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
            raise FunmanWorkerException("FunmanWorker must be running to check currently processing request")
        with self._id_lock:
            return self.current_id

    def _update_current_results(self, results: ParameterSpace):
        if self.current_results is None:
            print(
                "WARNING: Attempted to update results while results was None"
            )
            return
        # TODO handle copy?
        self.current_results.parameter_space = results

    def _run(self):
        print("FunmanWorker starting...")
        try:
            self.set_state(WorkerState.RUNNING)
            while True:
                if self._stop_event.is_set():
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
                        resultsCallback=self._update_current_results,
                    )
                    self.current_results.finalize_result(result)
                    print(f"Completed work on: {work.id}")
                except Exception:
                    print(f"Internal Server Error ({work.id}):", file=sys.stderr)
                    traceback.print_exc()
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
        finally:
            self.set_state(WorkerState.EXITED)
        print("FunmanWorker exiting...")
