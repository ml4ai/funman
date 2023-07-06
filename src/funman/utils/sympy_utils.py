from pysmt.shortcuts import Symbol, REAL, Real, Pow, Plus, Minus, Times, Div


def sympy_to_pysmt(expr):
    func = expr.func
    if func.is_Mul:
        return sympy_to_pysmt_op(Times, expr)
    elif func.is_Symbol:
        return sympy_to_pysmt_unary(Symbol, expr, op_type=REAL)


def sympy_to_pysmt_op(op, expr):
    terms = []
    for arg in expr.args:
        terms.append(sympy_to_pysmt(arg))
    return op(terms)


def sympy_to_pysmt_unary(op, expr, op_type=None):
    return op(str(expr), op_type) if op_type else op(str(expr))
