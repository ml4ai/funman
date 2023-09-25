import faulthandler
import io
import logging
import math
import os
from contextlib import contextmanager
from functools import partial
from queue import Queue
from timeit import default_timer
from typing import Dict, List

import dreal
import pysmt.smtlib.commands as smtcmd
from funman_dreal.converter import DRealConverter
from pysmt.decorators import clear_pending_pop
from pysmt.exceptions import (
    SolverRedefinitionError,
    SolverReturnedUnknownResultError,
    UnknownSolverAnswerError,
)
from pysmt.logics import QF_NRA
from pysmt.shortcuts import Real, get_env
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import SmtLibCommand
from pysmt.smtlib.solver import SmtLibOptions, SmtLibSolver
from pysmt.solvers.eager import EagerModel
from pysmt.solvers.smtlib import SmtLibBasicSolver, SmtLibIgnoreMixin
from pysmt.solvers.solver import Solver, SolverOptions, UnsatCoreSolver
from tenacity import retry
from pysmt.formula import FNode

import docker
from funman.utils.smtlib_utils import FUNMANSmtPrinter

# def setup(smt2_file, benchmark_path, out_dir):
#     print("setting up docker")
#     if os.path.exists(out_dir):
#         rmtree(out_dir)
#     os.makedirs(out_dir)
#     copyfile(
#         os.path.join(benchmark_path, smt2_file),
#         os.path.join(out_dir, smt2_file),
#     )


l = logging.getLogger(__name__)
l.setLevel(logging.INFO)


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
    solver = partial(DRealNative, "dreal/dreal4", LOGICS=DRealNative.LOGICS)
    solver.LOGICS = DRealNative.LOGICS
    env.factory._all_solvers[name] = solver
    env.factory._generic_solvers[name] = ("dreal/dreal4", DRealNative.LOGICS)
    env.factory.solver_preference_list.append(name)


def ensure_dreal_in_pysmt():
    try:
        add_dreal_to_pysmt()
    except SolverRedefinitionError:
        pass


class DReal(SmtLibSolver):
    LOGICS = [QF_NRA]
    OptionsClass = SmtLibOptions

    commands_to_enqueue = [
        smtcmd.SET_LOGIC,
        smtcmd.PUSH,
        smtcmd.POP,
        smtcmd.DEFINE_FUN,
        smtcmd.ASSERT,
    ]
    commands_to_flush = [
        smtcmd.CHECK_SAT,
    ]

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
            entrypoint=f"/usr/bin/dreal --in --model --smtlib2-compliant",
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

        self.models: List[Dict[str, (float, float)]] = []
        self.current_model = None
        self.symbols = set([])

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
            self.wait = 0
            return
        else:
            self.wait = 1

            if cmd.name == smtcmd.CHECK_SAT:
                self.current_model = {}
                self.models.append(self.current_model)
            elif cmd.name == smtcmd.DECLARE_FUN:
                symbol = cmd.args[0]
                self.symbols.add(symbol)

            # self.last_sent = payload
            self.batch.put(cmd)
            if cmd.name in DReal.commands_to_flush:
                self._send_batch()

            # print(f"Sent: {payload}")

    def _get_value_answer(self):
        """Reads and parses an assignment from the STDOUT pipe"""
        value = self.solver_stdout
        lst = self.parser.get_assignment_list(value)
        # lst = [[0.0, 0.0]]
        self._debug("Read: %s", lst)
        return lst

    def get_value(self, item):
        return Real(
            self.current_model[item.symbol_name()][0]
        )  # return lower bound

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
        # payload += b"\r\n"
        self.stdin._sock.send(payload)
        # self.stdin.flush()

    @retry
    def _read_socket(self):
        return self.stdout._sock.recv(1)

    def _read_line(self):
        # Read a line of content, skipping empty lines with only "\n"
        msg = []
        while True:
            b = self._read_socket()
            if b == b"\r" or (b == b"\n" and len(msg) == 0):
                continue
            msg.append(b)
            if b == b"\n":
                line = b"".join(msg)
                # print(f"Read: {line}")
                return line

    def _get_echo_result(self, encoded_text):
        """
        Read back the stdout from dreal that echo's the encoded_text

        Parameters
        ----------
        command : str
            SMTLib Command
        encoded_text : str
            Serialized command text
        """

        encoded_text = encoded_text.replace(b"\r", b"")
        line = self._read_line()  # .replace(b"\n", b"")
        if encoded_text == line:
            return "success"
        else:
            raise Exception(
                f"Could not read back echoed command from dreal: {encoded_text}, Got: {line}"
            )

    def get_check_sat_value_result(self):
        line = self._read_line().decode()
        if "delta-sat" in line:
            return "delta-sat"
        elif "unsat" in line:
            return "unsat"
        elif "sat" in line:
            return "sat"

    def _get_model(self):
        """
        Read list of "symbol: [lb, ub]\n" and store as model
        """
        while len(self.current_model) < len(self.symbols):
            line = self._read_line().decode()
            [symbol_str, bounds] = line.split(":")
            [lb, ub] = (
                bounds.replace("[", "").replace("]", "").strip().split(",")
            )
            self.current_model[symbol_str.strip()] = [float(lb), float(ub)]

    def _get_check_sat_result(self, encoded_text):
        """
        Read back the stdout from dreal that echo's the encoded_text, the result, and possibly a model.

        Parameters
        ----------
        command : str
            SMTLib Command
        encoded_text : str
            Serialized command text
        """
        if self._get_echo_result(encoded_text):  # read back "(check-sat)"
            result = self.get_check_sat_value_result()
            if result == "delta-sat" or result == "sat":
                self._get_model()
            return result

    def _get_command_result(self, command, encoded_text):
        handlers = {
            smtcmd.SET_LOGIC: self._get_echo_result,
            smtcmd.DECLARE_FUN: self._get_echo_result,
            smtcmd.ASSERT: self._get_echo_result,
            smtcmd.CHECK_SAT: self._get_check_sat_result,
        }
        try:
            return handlers[command.name](encoded_text)
        except TimeoutError as e:
            print(f"Timeout waiting for socket receive.")
            res = "success"
        except Exception as e:
            print(f"Failed to read from socket: {e}")

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


class DRealNative(
    Solver, UnsatCoreSolver, SmtLibBasicSolver, SmtLibIgnoreMixin
):
    LOGICS = [QF_NRA]
    OptionsClass = SolverOptions

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
        faulthandler.enable()

        Solver.__init__(self, environment, logic=logic, **options)

        # Setup super class attributes
        self.converter = DRealConverter(environment)
        # self.options(self)
        # self.options.set_params(self)
        self.mgr = environment.formula_manager
        self.model = None

        # dreal specific attributes
        self.config = dreal.Config()

        # self.context.config.use_polytope = True
        # self.context.config.use_worklist_fixpoint = True
        self.model = None
        if "solver_options" in options:
            if "dreal_precision" in options["solver_options"]:
                self.config.precision = options["solver_options"][
                    "dreal_precision"
                ]
            if "dreal_log_level" in options["solver_options"]:
                if options["solver_options"]["dreal_log_level"] == "debug":
                    dreal.set_log_level(dreal.LogLevel.DEBUG)
                elif options["solver_options"]["dreal_log_level"] == "trace":
                    dreal.set_log_level(dreal.LogLevel.TRACE)
                elif options["solver_options"]["dreal_log_level"] == "info":
                    dreal.set_log_level(dreal.LogLevel.INFO)

            if (
                "dreal_mcts" in options["solver_options"]
                and options["solver_options"]["dreal_mcts"]
            ):
                self.config.mcts = True

        self.config.unsat_core = True

        self.context = dreal.Context(self.config)
        self.context.SetLogic(dreal.Logic.QF_NRA)

        # self.to = self.environment.typeso
        self.LOGICS = DReal.LOGICS
        self.symbols = {}
        l.debug("Created new Solver ...")

    @contextmanager
    def elapsed_timer(self):
        start = default_timer()
        elapser = lambda: default_timer() - start
        try:
            yield elapser
        finally:
            elapser = None

    def __del__(self):
        # print("Exit()")
        self.context.Exit()  # Exit() only logs within dreal
        self.context = None

    @clear_pending_pop
    def add_assertion(self, formula, named=None):
        # print(f"Assert({formula.serialize()})")

        f = self.converter.convert(formula)

        # Convert Variable to a Formula
        if (
            isinstance(f, dreal.Variable)
            and f.get_type() == dreal.Variable.Bool
        ):
            f = dreal.And(f, f)

        deps = formula.get_free_variables()
        # Declare all variables
        for symbol in deps:
            assert symbol.is_symbol()
            self.cmd_declare_fun(
                SmtLibCommand(name=smtcmd.DECLARE_FUN, args=[symbol])
            )
        code = self.context.Assert(f)

    def cmd_set_option(self, cmd):
        pass

    def cmd_set_logic(self, cmd):
        pass

    def cmd_declare_fun(self, cmd):
        # print(f"DeclareVariable({cmd.args[0]})")
        if cmd.args[0] in self.converter.symbol_to_decl:
            v = self.converter.symbol_to_decl[cmd.args[0]]
        else:
            v = dreal.Variable(cmd.args[0].symbol_name(), dreal.Variable.Real)
        self.context.DeclareVariable(v)
        self.symbols[cmd.args[0].symbol_name()] = (cmd.args[0], v)

    def cmd_assert(self, cmd: SmtLibCommand):
        self.add_assertion(cmd.args[0])

    def cmd_push(self, cmd):
        self.push(cmd.args[0])

    def push(self, levels):
        # print("Push()")
        self.context.Push(levels)

    def cmd_pop(self, cmd):
        self.pop(cmd.args[0])

    def pop(self, levels):
        # print("Pop()")
        self.context.Pop(levels)

    def cmd_check_sat(self, cmd):
        return self.check_sat()

    def check_sat(self):
        # print("CheckSat()")
        with self.elapsed_timer() as t:
            result = self.context.CheckSat()
            elapsed_base_dreal = t()
        l.debug(
            f"{('delta-sat' if result else 'unsat' )} took {elapsed_base_dreal}s"
        )
        # result = dreal.CheckSatisfiability(self.assertion, 0.001)
        self.model = result
        return result

    def get_unsat_core(self)->FNode:
        unsat_core:dreal.Formula = self.context.get_unsat_core()
        f = self.converter.back(unsat_core)
        return f

    def get_named_unsat_core(self):
        """Returns the unsat core as a dict of names to formulae.

        After a call to solve() yielding UNSAT, returns the unsat core as a
        dict of names to formulae
        """
        raise NotImplementedError

    def _send_command(self, cmd):
        handlers = {
            smtcmd.SET_LOGIC: self.cmd_set_logic,
            smtcmd.SET_OPTION: self.cmd_set_option,
            smtcmd.DECLARE_FUN: self.cmd_declare_fun,
            smtcmd.ASSERT: self.cmd_assert,
            smtcmd.PUSH: self.cmd_push,
            smtcmd.POP: self.cmd_pop,
            smtcmd.CHECK_SAT: self.cmd_check_sat,
        }
        return handlers[cmd.name](cmd)

    def get_model(self):
        assignment = {}
        for sn in self.symbols:
            s = self.symbols[sn][0]
            if s.is_term():
                v = self.get_value(self.symbols[sn][1])
                assignment[s] = v
        return EagerModel(assignment=assignment, environment=self.environment)

    def get_value(self, item):
        # print(f"get_value() {item}: {self.model[item]}")
        ub = self.model[item].ub()
        lb = self.model[item].lb()
        mid = (ub - lb) / 2.0
        mid = mid + lb
        if not mid.is_integer() and (ub.is_integer() or lb.is_integer()):
            return Real(lb) if lb.is_integer() else Real(ub)
        elif not math.isinf(mid):
            return Real(mid)
        else:
            return Real(self.model[item].lb())

    @clear_pending_pop
    def solve(self, assumptions=None):
        assert assumptions is None
        try:
            ans = self.check_sat()
        except Exception as e:
            raise e
        return ans is not None

    def _exit(self):
        pass
