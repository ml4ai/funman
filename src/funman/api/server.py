"""
Server for API 
"""
import contextlib
import threading
import time

import uvicorn


class ServerConfig(uvicorn.Config):
    """
    Server configuration
    """

    def __init__(
        self,
        app,
        host: str = "0.0.0.0",
        port: int = 8190,
        log_level: str = "info",
    ):
        super().__init__(app, host=host, port=port, log_level=log_level)


class Server(uvicorn.Server):
    """
    Uvicorn server object
    """

    @contextlib.contextmanager
    def run_in_thread(self):
        """
        Override the uvicorn method to allow running server in a thread, for example in a notebook.
        """
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()
