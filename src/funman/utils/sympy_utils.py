import math
from typing import Dict, Union

import pysmt.operators as op
import pysmt.typing as types
import sympy
from pysmt.formula import FNode, FormulaManager
from pysmt.shortcuts import (
    REAL,
    Div,
    Minus,
    Plus,
    Pow,
    Real,
    Symbol,
    Times,
    get_env,
)


class FUNMANFormulaManager(FormulaManager):
    """FormulaManager is responsible for the creation of all formulae."""

    def __init__(self, formula_manager, env=None):
        # Copy formula_manager, and don't call super().__init__()
        self.env = formula_manager.env
        self.formulae = formula_manager.formulae
        self.symbols = formula_manager.symbols
        self._fresh_guess = formula_manager._fresh_guess
        self.get_type = formula_manager.get_type
        self._next_free_id = formula_manager._next_free_id

        self.int_constants = formula_manager.int_constants
        self.real_constants = formula_manager.real_constants
        self.string_constants = formula_manager.string_constants

        self.true_formula = formula_manager.true_formula
        self.false_formula = formula_manager.false_formula

    def Pow(self, base, exponent):
        """Creates the n-th power of the base.

        The exponent must be a constant.
        """

        # Funman allows expressions in base, thus the override of the super Pow
        # if not exponent.is_constant():
        #     raise PysmtValueError("The exponent of POW must be a constant.", exponent)

        if base.is_constant() and exponent.is_constant():
            val = base.constant_value() ** exponent.constant_value()
            if base.is_constant(types.REAL):
                return self.Real(val)
            else:
                assert base.is_constant(types.INT)
                return self.Int(val)
        return self.create_node(node_type=op.POW, args=(base, exponent))


def substitute(str_expr: str, values: Dict[str, Union[float, str]]):
    # Set which substrings are symbols
    symbols = {s: sympy.Symbol(s) for s in values}

    # Get expression
    expr = str_expr_to_expr(str_expr, symbols)

    # Get variable values
    values_syms = {s: values[str(s)] for s in symbols}
    subs = [(sym, val) for sym, val in values_syms.items() if val is not None]

    # substitute
    sub_expr = expr.subs(subs)
    return sub_expr


def str_expr_to_expr(sexpr, symbols):
    f = sympy.sympify(sexpr, symbols)
    return f


def rate_expr_to_pysmt(expr, state=None):
    env_symbols = get_env().formula_manager.symbols
    symbols = {s: sympy.Symbol(s) for s in env_symbols}
    f = str_expr_to_expr(expr, symbols)
    p: FNode = sympy_to_pysmt(f)

    if state:  # Map symbols in p to state indexed versions (e.g., I to I_5)
        symbol_to_state_var = {
            env_symbols[s]: state[str(s)] for s in symbols if str(s) in state
        }
        p_sub = p.substitute(symbol_to_state_var)
        return p_sub
    else:
        return p


def sympy_to_pysmt(expr):
    func = expr.func
    if func.is_Mul:
        return sympy_to_pysmt_op(Times, expr)
    elif func.is_Add:
        return sympy_to_pysmt_op(Plus, expr)
    elif func.is_Symbol:
        return sympy_to_pysmt_symbol(Symbol, expr, op_type=REAL)
    elif func.is_Pow:
        return sympy_to_pysmt_pow(expr)
    elif isinstance(expr, sympy.exp):
        return Pow(sympy_to_pysmt_real(math.e), sympy_to_pysmt(expr.exp))
    elif expr.is_constant():
        return sympy_to_pysmt_real(expr)
    else:
        raise Exception(f"Could not convert expression: {expr}")


def sympy_to_pysmt_op(op, expr):
    terms = [sympy_to_pysmt(arg) for arg in expr.args]
    return op(terms)


def sympy_to_pysmt_pow(expr):
    base = expr.args[0]
    exponent = expr.args[1]
    return Pow(sympy_to_pysmt(base), sympy_to_pysmt(exponent))


def sympy_to_pysmt_real(expr):
    return Real(float(expr))


def sympy_to_pysmt_symbol(op, expr, op_type=None):
    return op(str(expr), op_type) if op_type else op(str(expr))


if __name__ == "__main__":
    symbols = {
        "I": sympy.Symbol("I"),
        "S": sympy.Symbol("S"),
        "N": sympy.Symbol("N"),
    }
    exprs = [
        "I*S",
        "I*S*kappa*(beta_c + (-beta_c + beta_s)/(1 + exp(-k*(-t + t_0))))/N",
    ]
    for e in exprs:
        f = sympy.sympify(e, symbols)
        p: FNode = sympy_to_pysmt(f)
        print(f"Read: {e}")
        print(f"Sympy parsed: {f}")
        print(f"Pysmt parse: {p.to_smtlib(daggify=False)}")
        print("*" * 80)
