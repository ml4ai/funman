import queue
import threading
from typing import Optional

from funman import Funman, model
from funman.funman import FUNMANConfig
from funman.model.model import Model
from funman.representation.representation import ParameterSpace
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)


class FunmanWorker:
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

    def enqueue_work(
        self, model: Model, request: FunmanWorkRequest
    ) -> FunmanWorkUnit:
        id = self.storage.claim_id()
        work = FunmanWorkUnit(id=id, model=model, request=request)
        self.queue.put(work)
        with self._set_lock:
            self.queued_ids.add(work.id)
        return work

    def start(self):
        if self._thread is not None:
            raise Exception("FunmanWorker already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event.clear()
        self._thread.start()

    def stop(self, timeout=None):
        if self._thread is None:
            raise Exception("FunmanWorker already stopped")
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            # TODO kill thread?
            print("Thread did not close")
        self._thread = None

    def is_processing_id(self, id: str):
        with self._id_lock:
            return self.current_id == id

    def get_results(self, id: str):
        if self.is_processing_id(id):
            return self.current_results
        return self.storage.get_result(id)

    def halt(self, id: str):
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
                    done=False,
                    parameter_space=ParameterSpace(),
                )

            print(f"Starting work on: {work.id}")
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
            self.storage.add_result(self.current_results)
            self.queue.task_done()
            print(f"Completed work on: {work.id}")
            with self._id_lock:
                self.current_id = None
                self.current_results = None
        print("FunmanWorker exiting...")
