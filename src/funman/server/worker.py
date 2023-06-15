import queue
import threading
from typing import Optional

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model.model import Model
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)


class FunmanWorker:
    def __init__(self, storage):
        self._stop_event = threading.Event()
        self._thread = None
        self._id_lock = threading.Lock()
        self._set_lock = threading.Lock()

        self.storage = storage
        self.queue = queue.Queue()
        self.queued_ids = set()
        self.current_id = None

    def enqueue_work(
        self, model: Model, request: FunmanWorkRequest
    ) -> FunmanWorkUnit:
        id = self.storage.claim_id()
        work = FunmanWorkUnit(id=id, model=model, request=request)
        self.queue.put(work)
        with self._set_lock:
            self.queued_ids.add(work.id)
        return work

    def _get_work(self):
        return self.queue.get()

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
            # TODO kill thread
            pass

    def is_processing_id(self, id: str):
        with self._id_lock:
            return self.current_id == id

    def get_results(self, id: str):
        if self.is_processing_id(id):
            print("TODO: Get running results")
            return None
        return self.storage.get_result(id)

    def halt(self, id: str):
        with self._id_lock:
            if id == self.current_id:
                print(f"TODO: Halt {id}")
                return
            with self._set_lock:
                if id in self.queued_ids:
                    self.queued_ids.remove(id)
                return

    def get_current(self) -> Optional[str]:
        with self._id_lock:
            return self.current_id

    def _run(self):
        print("FunmanWorker starting...")
        while True:
            if self._stop_event.is_set():
                break
            work: FunmanWorkUnit = self._get_work()

            # skip work that is no longer in the queued_ids set
            # since that likely indicated it has been halted
            # before starting
            with self._set_lock:
                if work.id not in self.queued_ids:
                    continue

            with self._id_lock:
                self.current_id = work.id
            print(f"Starting work on: {work.id}")
            # convert to scenario
            scenario = work.to_scenario()

            config = (
                FUNMANConfig()
                if work.request.config is None
                else work.request.config
            )

            f = Funman()
            result = f.solve(scenario, config=config)
            response = FunmanResults.from_result(work, result)
            self.storage.add_result(response)
            self.queue.task_done()
            print(f"Completed work on: {work.id}")
            with self._id_lock:
                self.current_id = None
        print("FunmanWorker exiting...")
