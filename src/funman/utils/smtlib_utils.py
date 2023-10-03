import logging
import warnings

import pysmt.smtlib.commands as smtcmd
from pysmt.constants import Numeral
from pysmt.environment import get_env
from pysmt.exceptions import NoLogicAvailableError, UndefinedLogicError
from pysmt.logics import SMTLIB2_LOGICS, Logic, get_closer_smtlib_logic
from pysmt.oracles import get_logic
from pysmt.smtlib.printers import (
    SmtDagPrinter,
    SmtPrinter,
    write_annotations,
    write_annotations_dag,
)
from pysmt.smtlib.script import SmtLibCommand, SmtLibScript
from pysmt.solvers.solver import Model

l = logging.Logger(__file__)
l.setLevel(logging.WARNING)


class FUNMANSmtLibScript(SmtLibScript):
    def serialize(self, outstream, daggify=True):
        """Serializes the SmtLibScript expanding commands"""
        if daggify:
            printer = FUNMANSmtDagPrinter(
                outstream, annotations=self.annotations
            )
        else:
            printer = FUNMANSmtPrinter(outstream, annotations=self.annotations)

        for cmd in self.commands:
            cmd.serialize(printer=printer)
            outstream.write("\n")


class FUNMANSmtPrinter(SmtPrinter):
    @write_annotations
    def walk_real_constant(self, formula):
        (n, d) = (
            abs(formula.constant_value().numerator),
            formula.constant_value().denominator,
        )
        if formula.constant_value() < 0:
            if d != 1:
                # res = f"(- {(n / d):20f})"
                res = f"(- {(n / d)})"
            else:
                res = f"(- {n})"
        else:
            if d != 1:
                # res = f"{(n / d):20f}"
                res = f"{(n / d)}"
            else:
                res = f"{n}"

        self.write(res)


class FUNMANSmtDagPrinter(SmtDagPrinter):
    @write_annotations_dag
    def walk_real_constant(self, formula, **kwargs):
        if formula.constant_value() < 0:
            template = "(- %s)"
        else:
            template = "%s"

        (n, d) = (
            abs(formula.constant_value().numerator),
            formula.constant_value().denominator,
        )
        if d != 1:
            return template % str(n / d)
        else:
            return template % (str(n) + ".0")


def smtlibscript_from_formula_list(formulas, logic=None):
    script = FUNMANSmtLibScript()

    if logic is None:
        # Get the simplest SmtLib logic that contains the formula
        f_logic = [get_logic(formula) for formula in formulas][0]

        smt_logic = None
        try:
            smt_logic = get_closer_smtlib_logic(f_logic)
        except NoLogicAvailableError:
            warnings.warn(
                "The logic %s is not reducible to any SMTLib2 "
                "standard logic. Proceeding with non-standard "
                "logic '%s'" % (f_logic, f_logic),
                stacklevel=3,
            )
            smt_logic = f_logic
    elif not (isinstance(logic, Logic) or isinstance(logic, str)):
        raise UndefinedLogicError(str(logic))
    else:
        if logic not in SMTLIB2_LOGICS:
            warnings.warn(
                "The logic %s is not reducible to any SMTLib2 "
                "standard logic. Proceeding with non-standard "
                "logic '%s'" % (logic, logic),
                stacklevel=3,
            )
        smt_logic = logic

    script.add(name=smtcmd.SET_LOGIC, args=[smt_logic])

    # Declare all types
    types = set(
        [
            t
            for formula in formulas
            for t in get_env().typeso.get_types(formula, custom_only=True)
        ]
    )

    # print(types)
    for type_ in types:
        script.add(name=smtcmd.DECLARE_SORT, args=[type_.decl])

    prev_deps = set([])
    # Assert formula
    for i, formula in enumerate(formulas):
        deps = set(
            [d for d in formula.get_free_variables() if d not in prev_deps]
        )
        # Declare all variables
        for symbol in deps:
            prev_deps.add(symbol)
            assert symbol.is_symbol()
            script.add(name=smtcmd.DECLARE_FUN, args=[symbol])
        script.add_command(SmtLibCommand(name=smtcmd.ASSERT, args=[formula]))
        script.add_command(SmtLibCommand(name=smtcmd.PUSH, args=[1]))

    # check-sat
    script.add_command(SmtLibCommand(name=smtcmd.CHECK_SAT, args=[]))
    return script


def smtlibscript_from_formula(formula, logic=None):
    script = FUNMANSmtLibScript()

    if logic is None:
        # Get the simplest SmtLib logic that contains the formula
        f_logic = get_logic(formula)

        smt_logic = None
        try:
            smt_logic = get_closer_smtlib_logic(f_logic)
        except NoLogicAvailableError:
            warnings.warn(
                "The logic %s is not reducible to any SMTLib2 "
                "standard logic. Proceeding with non-standard "
                "logic '%s'" % (f_logic, f_logic),
                stacklevel=3,
            )
            smt_logic = f_logic
    elif not (isinstance(logic, Logic) or isinstance(logic, str)):
        raise UndefinedLogicError(str(logic))
    else:
        if logic not in SMTLIB2_LOGICS:
            warnings.warn(
                "The logic %s is not reducible to any SMTLib2 "
                "standard logic. Proceeding with non-standard "
                "logic '%s'" % (logic, logic),
                stacklevel=3,
            )
        smt_logic = logic

    script.add(name=smtcmd.SET_LOGIC, args=[smt_logic])

    # Declare all types
    types = get_env().typeso.get_types(formula, custom_only=True)
    for type_ in types:
        script.add(name=smtcmd.DECLARE_SORT, args=[type_.decl])

    deps = formula.get_free_variables()
    # Declare all variables
    for symbol in deps:
        assert symbol.is_symbol()
        script.add(name=smtcmd.DECLARE_FUN, args=[symbol])

    # Assert formula
    script.add_command(SmtLibCommand(name=smtcmd.ASSERT, args=[formula]))
    # check-sat
    script.add_command(SmtLibCommand(name=smtcmd.CHECK_SAT, args=[]))
    return script


def model_to_dict(self):
    d = {}
    for var in self:
        try:
            value = (
                var[1].constant_value()
                if var[1].is_constant()
                else self.get_py_value(var[0], model_completion=False)
            )
            if isinstance(value, bool):
                pass
            elif isinstance(value, Numeral):
                l.warning(
                    f"Setting value of Numeral {var} to 0.0 because cannot convert to float"
                )
                value = 0.0
            else:
                value = float(value)
            d[var[0].symbol_name()] = value
        except OverflowError as e:
            raise e
    return d


Model.to_dict = model_to_dict
