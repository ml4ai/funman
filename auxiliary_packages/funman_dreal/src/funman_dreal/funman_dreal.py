from functools import partial
import os
from queue import Queue
import docker
from shutil import copyfile, rmtree
import time
import io
from funman.util import FUNMANSmtPrinter
from pysmt.smtlib.solver import SmtLibSolver, SmtLibOptions
from pysmt.solvers.solver import Solver, SolverOptions
from pysmt.logics import QF_NRA
from pysmt.solvers.solver import Solver
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import SmtLibCommand
import pysmt.smtlib.commands as smtcmd
from pysmt.exceptions import (
    SolverReturnedUnknownResultError,
    UnknownSolverAnswerError,
)
from pysmt.decorators import clear_pending_pop
from tenacity import retry
from pysmt.shortcuts import get_env, GT, Symbol
from pysmt.logics import QF_NRA
from pysmt.exceptions import SolverRedefinitionError
from functools import partial
import pysmt.smtlib.commands as smtcmd

# def setup(smt2_file, benchmark_path, out_dir):
#     print("setting up docker")
#     if os.path.exists(out_dir):
#         rmtree(out_dir)
#     os.makedirs(out_dir)
#     copyfile(
#         os.path.join(benchmark_path, smt2_file),
#         os.path.join(out_dir, smt2_file),
#     )


# TODO find a better way to determine if solver was successful
def run_dreal(
    smt2_file,
    # benchmark_path=os.path.join(os.path.abspath("../../resources/smt2")),
    out_dir="out",
    solver_opts=" --precision 1 --verbose debug",
):
    # setup(smt2_file, benchmark_path, out_dir)

    smt2_file_path = os.path.dirname(smt2_file)
    smt2_filename = os.path.basename(smt2_file)

    volumes = {
        smt2_file_path: {"bind": "/smt2", "mode": "rw"},
        out_dir: {"bind": "/out", "mode": "rw"},
        "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
    }
    client = docker.from_env()
    container = client.containers.run(
        "dreal/dreal4",
        command=f"dreal /smt2/{smt2_filename} {solver_opts}",
        # command="/bin/bash",
        volumes=volumes,
        detach=True,
    )
    container.wait()
    result = container.logs()

    with open(os.path.join(out_dir, f"{smt2_filename}.stdout"), "w") as f:
        f.write(result.decode())
        f.close()

    if not os.path.exists(os.path.join(out_dir, f"{smt2_filename}.model")):
        container.stop()
        ll_client = docker.APIClient()
        ll_client.remove_container(container.id)
        raise Exception("Could not construct a model")

    container.stop()
    ll_client = docker.APIClient()
    ll_client.remove_container(container.id)

    with open(os.path.join(out_dir, f"{smt2_filename}.model")) as f:
        model = f.read()

    return model


def add_dreal_to_pysmt():
    name = "dreal"
    env = get_env()
    if name in env.factory._all_solvers:
        raise SolverRedefinitionError("Solver %s already defined" % name)
    solver = partial(DReal, "dreal/dreal4", LOGICS=DReal.LOGICS)
    solver.LOGICS = DReal.LOGICS
    env.factory._all_solvers[name] = solver
    env.factory._generic_solvers[name] = ("dreal/dreal4", DReal.LOGICS)
    env.factory.solver_preference_list.append(name)


class DReal(SmtLibSolver):
    LOGICS = [QF_NRA]
    OptionsClass = SmtLibOptions

    commands_to_enqueue = [
        smtcmd.SET_LOGIC,
        smtcmd.DEFINE_FUN,
        smtcmd.PUSH,
        smtcmd.POP,
    ]
    commands_to_flush = [smtcmd.CHECK_SAT, smtcmd.ASSERT]

    def __init__(
        self, args, environment, logic, LOGICS=None, **options
    ) -> None:
        Solver.__init__(self, environment, logic=logic, **options)
        self.to = self.environment.typeso
        self.LOGICS = DReal.LOGICS
        self.declared_vars = [set()]
        self.declared_sorts = [set()]
        self.parser = SmtLibParser(interactive=True)

        # solver_opts = kwargs["solver_opts"] if "solver_opts" in kwargs else ""
        self.batch = Queue()  # Batch of commands to send in one call
        self.sent_batch = (
            Queue()
        )  # Batch of commands already sent and needs to be checked
        self.wait = 1  # How long to wait on next socket recv
        self.last_sent = None  # Last message sent to container
        self.client = docker.APIClient()
        self.container = self.client.create_container(
            "dreal/dreal4",
            stdin_open=True,
            tty=True,
            entrypoint=f"/usr/bin/dreal --in",
        )
        self.client.start(self.container)

        self.stdout = self.client.attach_socket(
            self.container,
            params={"stdout": 1, "stderr": 1, "stream": 1},
        )
        self.stdin = self.client.attach_socket(
            self.container,
            params={"stdin": 1, "stream": 1},
        )

        # Initialize solver
        try:
            self.options(self)
        except Exception as e:
            pass

        self.set_logic(logic)

    def container_alive(func):
        def inner(*args):
            # time.sleep(1)
            status = args[0].client.inspect_container(args[0].container)[
                "State"
            ]["Status"]
            if status == "exited":
                raise Exception(
                    f"Cannot send command to container with status: {status}"
                )
            else:
                return func(*args)

        return inner

    # @container_alive
    def _send_command(self, cmd):
        """Sends a command to the socket."""
        self._debug("Sending: %s", cmd.serialize_to_string())

        # s._sock.send(input.encode("utf-8"))

        if cmd.name == smtcmd.SET_OPTION:
            # dreal does not support set-option command
            self.wait = 0
            # print(f"Skipped: {text.getvalue()}")
            return
        else:
            # if "check-sat" in text.getvalue():
            #     self.wait = 10
            # # elif "assert" in text.getvalue():
            # #     self.wait = 1
            # # elif "declare-fun" in text.getvalue():
            # #     self.wait = 1
            # else:
            self.wait = 1

            # self.last_sent = payload
            self.batch.put(cmd)
            if cmd.name in DReal.commands_to_flush:
                self._send_batch()

            # print(f"Sent: {payload}")

    def _get_value_answer(self):
        """Reads and parses an assignment from the STDOUT pipe"""
        # lst = self.parser.get_assignment_list(self.solver_stdout)
        lst = [[0.0, 0.0]]
        self._debug("Read: %s", lst)
        return lst

    def get_value(self, item):
        # self._send_command(SmtLibCommand(smtcmd.GET_VALUE, [item]))
        lst = self._get_value_answer()
        assert len(lst) == 1
        assert len(lst[0]) == 2
        return lst[0][1]

    def _send_batch(self):
        payload = b""
        while not self.batch.empty():
            cmd = self.batch.get()
            text = io.StringIO()
            cmd.serialize(text, daggify=False, printer=FUNMANSmtPrinter(text))
            text.write("\r\n")
            encoded_text = text.getvalue().encode("utf-8")
            payload += encoded_text
            self.sent_batch.put((cmd, encoded_text))
        self.stdin._sock.sendall(payload)
        self.stdin.flush()

    @retry
    def _read_socket(self):
        return self.stdout._sock.recv(1)

    def _get_command_result(self, command, encoded_text):
        try:
            msg = []
            reading_result = False
            result = []
            encoded_text = encoded_text.replace(b"\r", b"")
            while True:
                b = self._read_socket()
                if b == b"\r":
                    continue
                msg.append(b)
                if command == smtcmd.CHECK_SAT:
                    # Keep reading to get result until "\n"
                    if reading_result:
                        if b == b"\n":
                            break
                        else:
                            result.append(b)
                    elif b"".join(msg) == encoded_text:
                        reading_result = True
                elif b"".join(msg) == encoded_text:
                    break
                elif b"".join(msg) == b"\n":
                    msg = []
                    continue

            if len(result) > 0:
                res = b"".join(result).decode()
            else:
                res = "success"
        except TimeoutError as e:
            print(f"Timeout waiting for socket receive.")
            res = "success"
        except Exception as e:
            print(f"Failed to read from socket: {e}")
        return res

    # @container_alive
    def _get_answer(self):
        """Reads a line from socket."""
        if self.wait > 0:
            # print("Reading from socket: ")
            # self.stdout._sock.settimeout(self.wait)
            res = "success"
            while not self.sent_batch.empty():
                res = self._get_command_result(*self.sent_batch.get())
        else:
            res = "success"
        # print(f"Read: {res}")
        return res

    @clear_pending_pop
    def solve(self, assumptions=None):
        assert assumptions is None
        self._send_command(SmtLibCommand(smtcmd.CHECK_SAT, []))
        ans = self._get_answer()
        if ans == "sat" or ans == "success" or "delta-sat" in ans:
            return True
        elif "unsat" in ans:
            return False
        elif ans == "unknown":
            raise SolverReturnedUnknownResultError
        else:
            raise UnknownSolverAnswerError("Solver returned: " + ans)

    def _exit(self):
        # self._send_command(SmtLibCommand(smtcmd.EXIT, []))

        self.stdin.close()
        self.stdout.close()
        print("Stopping dreal container ...")
        self.client.stop(self.container)
        self.client.wait(self.container)
        # raw_stream,status = client.get_archive(container,'/received.txt')
        # tar_archive = BytesIO(b"".join((i for i in raw_stream)))
        # t = tarfile.open(mode='r:', fileobj=tar_archive)
        # text_from_container_file = t.extractfile('received.txt').read().decode('utf-8')
        self.client.remove_container(self.container)
        print("Done stopping dreal container.")
        return

    def send_input(self, input):
        s = self.client.attach_socket(
            self.container,
            params={"stdout": 1, "stderr": 1, "stream": 1, "stdin": 1},
        )
        # pass
        s._sock.settimeout(1)
        s._sock.send(input.encode("utf-8"))
        rval = io.StringIO()
        while 1:
            try:
                msg = s._sock.recv(1024)
            except TimeoutError as e:
                # print(f"Timeout waiting for data from container: {e}")
                msg = None
            if not msg:
                break
            else:
                try:
                    for line in msg.split(b"\n"):
                        content = f"{line[8:].decode()}\n"
                        rval.write(content)
                except Exception as e:
                    print(f"Could not decode response: {msg} because {e}")
        s.close()
        return rval.getvalue()
