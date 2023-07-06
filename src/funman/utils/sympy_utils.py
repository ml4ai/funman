import math

import sympy
from pysmt.formula import FNode
from pysmt.shortcuts import REAL, Div, Minus, Plus, Pow, Real, Symbol, Times


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
