from typing import Dict, List

import pysmt
from pysmt.fnode import FNode
from pysmt.shortcuts import get_env
from sympy import (
    Abs,
    Add,
    Expr,
    Float,
    Max,
    N,
    cancel,
    expand,
    factor,
    lambdify,
    nsimplify,
    series,
    symbols,
    sympify,
)

from funman.constants import POS_INFINITY
from funman.representation.representation import ModelParameter
from funman.utils.sympy_utils import (
    replace_reserved,
    series_approx,
    sympy_to_pysmt,
    to_sympy,
)

import logging

l = logging.getLogger(__name__)
l.setLevel(logging.INFO)

class FUNMANSimplifier(pysmt.simplifier.Simplifier):
    def __init__(self, env=None):
        super().__init__(env=env)
        self.manager = self.env.formula_manager

    def value_of(expr: Expr, subs: Dict[str, float] = {}):
        arg_values = [subs[str(v)] for v in expr.free_symbols]
        lfn = lambdify(list(expr.free_symbols), expr, 'numpy')
        if len(arg_values) > 0:
            try:
                value = lfn(*arg_values)
            except OverflowError as e:
                val = N(expr, subs=subs)
                if val > 1:
                    value = sys.float_info.max
                elif val < -1:
                    value = sys.float_info.min
                else:
                    value = 0.0 
                l.debug(f"Convert lambdify overflow of {expr} with {subs} from {val} to {value}")
            except:
                pass
            # except UnderflowError as e:
            #     value = 0.0

        else:
            value = lfn()
        return value

    def approximate(formula, parameters: List[ModelParameter], threshold=1e-4):
        if len(formula.free_symbols) == 0:
            return formula

        ub_values = {replace_reserved(p.name): p.ub for p in parameters}
        lb_values = {replace_reserved(p.name): p.lb for p in parameters}
        original_size = len(formula.args)

        # def calc_mag(g):
        #     free_syms = list(g.free_symbols)
        #     f = lambdify(list(g.free_symbols), Abs(g), "numpy")
        #     ub_vals = [ub_values[str(x)] for x in free_syms]
        #     ub_mag =f(*ub_vals)
        #     if ub_mag > threshold:
        #         return Float(ub_mag)
        #     else:
        #         lb_vals = [lb_values[str(x)] for x in free_syms]
        #         lb_mag =f(*lb_vals)
        #         return Float(Max(ub_mag, lb_mag))

        # term_magnitude = {}
        # for arg in formula.args:
        #     try:
        #         mag = calc_mag(arg)
        #     except Exception as e:
        #         mag = N(arg, subs=lb_values)
        #     term_magnitude[arg] = mag

        if formula.func.is_Add:
            args = formula.args
        else:
            args = [formula]

        to_drop = {
            arg: 0
            for arg in args
            if (
                abs(FUNMANSimplifier.value_of(arg, subs=lb_values)) < threshold
                and abs(FUNMANSimplifier.value_of(arg, subs=ub_values))
                < threshold
            )
        }
        # minimum_term_value = min(tm for arg, tm in term_magnitude.items()) if len(term_magnitude) > 0 else None
        # maximum_term_value = max(tm for arg, tm in term_magnitude.items()) if len(term_magnitude) > 0 else None

        # print("**** args:")
        # for arg in formula.args:
        #     status = f"({max(abs(arg.subs(ub_values)), abs(arg.subs(lb_values)))})" if (arg in to_drop) else None
        #     if status:
        #         print(f"{status} {arg}")

        # if len(to_drop) > 0:
        #     print("*" * 80)
        #     print(f"Drop\n {to_drop}")
        #     print(f"From\n {formula}")

        # for drop in to_drop:
        # subbed_formula = formula.subs(to_drop)
        if len(to_drop) > 0:
            subbed_formula = Add(
                *[t for t in args if t not in to_drop]
            )
        else:
            subbed_formula = formula
        print(
            f"*** {original_size}->{len(subbed_formula.args)}\t|{len(to_drop)}|"
        )
        # if len(to_drop) > 0:
        #     print(f"Result\n {formula}")
        #     pass

        return subbed_formula

    def sympy_simplify(
        formula: Expr,
        parameters: List[ModelParameter] = [],
        substitutions: Dict[FNode, FNode] = {},
        threshold: float = 1e-4,
        taylor_series_order=None,
    ):
        # substitutions are FNodes
        # transition terms are sympy.Expr
        # convert relevant substitutions to sympy.Expr
        # sympy subs transition term with converted subs
        # simplify/approximate substituted formula
        # convert to pysmt formula
        # TODO store substitutions as both FNode and pysmt.Expr to avoid extra conversion

        # if formula.is_real_constant():
        #     return formula

        # simplified_formula = formula.simplify()

        # if simplified_formula.is_real_constant():
        #     return simplified_formula

        # print(formula.serialize())
        # vars = formula.get_free_variables()
        # forumla_symbols = formula.free_symbols
        # var_map = {str(v): symbols(str(v)) for v in formula.free_symbols}
        # sympy_symbols = list(var_map.values())
        # var_map = {}
        psymbols = [replace_reserved(p.name) for p in parameters]
        sympy_subs = {
            symbols(s.symbol_name()): to_sympy(v, psymbols)
            for s, v in substitutions.items()
            if symbols(s.symbol_name()) in formula.free_symbols
        }
        # series_vars = [
        #     symbols(str(v)) for v in vars if symbols(str(v)) not in sympy_subs
        # ]

        # sympy_formula = sympify(simplified_formula.serialize(), var_map)
        # series_formula = reduce(
        #     lambda v1, v2: series(v1, v2).removeO(), series_vars, sympy_formula
        # )
        expanded_formula = expand(formula.subs(sympy_subs))

        f = expanded_formula
        if taylor_series_order is None:
            pass
        elif not f.is_polynomial(formula.free_symbols):
            f = series_approx(
                f,
                list(expanded_formula.free_symbols),
                order=taylor_series_order,
            )

        # expanded_formula = expand(sympy_formula)

        # print(expanded_formula)

        if threshold is not None and threshold > 0:
            f = FUNMANSimplifier.approximate(
                f, parameters, threshold=threshold
            )

        f = sympy_to_pysmt(f)

        # print(f.serialize())
        return f

    def walk_pow(self, formula, args, **kwargs):
        env = get_env()
        self.manager = env._formula_manager
        return self.manager.Pow(args[0], args[1])


class SympyToPysmt(object):
    def sympyToPysmt(sympy_formula):
        pass
