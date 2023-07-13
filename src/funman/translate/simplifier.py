from typing import List

from funman.representation.representation import Parameter
import pysmt
from pysmt.fnode import FNode
from pysmt.shortcuts import get_env
from sympy import cancel, expand, symbols, sympify, nsimplify, Float, Add, Abs, Max, lambdify, N

from funman.utils.sympy_utils import sympy_to_pysmt


class FUNMANSimplifier(pysmt.simplifier.Simplifier):
    def __init__(self, env=None):
        super().__init__(env=env)
        self.manager = self.env.formula_manager

    def approximate(formula, parameters: List[Parameter], threshold=1e-8):
        if len(formula.free_symbols)==0:
            return formula
        

        ub_values = {p.name: p.ub for p in parameters}
        lb_values = {p.name: p.lb for p in parameters}
        original_size = len(formula.args)

        def calc_mag(g):
            free_syms = list(g.free_symbols)
            f = lambdify(list(g.free_symbols), Abs(g), "numpy")
            ub_vals = [ub_values[str(x)] for x in free_syms]
            ub_mag =f(*ub_vals)
            if ub_mag > threshold:
                return Float(ub_mag)
            else:
                lb_vals = [lb_values[str(x)] for x in free_syms]
                lb_mag =f(*lb_vals)
                return Float(Max(ub_mag, lb_mag))

        term_magnitude = {}
        for arg in formula.args:
            try:
                mag = calc_mag(arg)
            except Exception as e:
                mag = N(arg, subs=lb_values) 
            term_magnitude[arg] = mag 

        to_drop = {
            arg: 0 for arg, tm in term_magnitude.items() if tm < threshold 
        }
        minimum_term_value = min(tm for arg, tm in term_magnitude.items()) if len(term_magnitude) > 0 else None
        maximum_term_value = max(tm for arg, tm in term_magnitude.items()) if len(term_magnitude) > 0 else None

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
        if len(to_drop)> 0:
            subbed_formula = Add(*[t for t in formula.args if t not in to_drop])
        else:
            subbed_formula = formula
        print(f"*** {original_size}->{len(subbed_formula.args)}\t [{minimum_term_value}, {maximum_term_value}] \t|{len(to_drop)}|")
        # if len(to_drop) > 0:
        #     print(f"Result\n {formula}")
        #     pass

        return subbed_formula

    def sympy_simplify(formula, parameters: List[Parameter] = []):
        if formula.is_real_constant():
            return formula
        
        # print(formula.serialize())
        vars = formula.get_free_variables()
        var_map = {str(v): symbols(str(v)) for v in vars}
        simplified_formula = formula.simplify()

        if simplified_formula.is_real_constant():
            return simplified_formula

        expanded_formula = expand(
            sympify(simplified_formula.serialize(), var_map)
        )
        # print(expanded_formula)
        approx_formula = FUNMANSimplifier.approximate(
            expanded_formula, parameters
        )
        # simp_approx_formula = simplify(approx_formula)
        # f = sympy_to_pysmt(simp_approx_formula)
        N_approx_formula = nsimplify(approx_formula, rational=True)
        f = sympy_to_pysmt(approx_formula)

        # print(f.serialize())
        return f

    # def walk_times(self, formula, args, **kwargs):
    #     new_args = []
    #     constant_mul = 1
    #     stack = list(args)
    #     ttype = self.env.stc.get_type(args[0])
    #     is_algebraic = False
    #     while len(stack) > 0:
    #         x = stack.pop()
    #         if x.is_constant():
    #             if x.is_algebraic_constant():
    #                 is_algebraic = True
    #             if x.is_zero():
    #                 constant_mul = 0
    #                 break
    #             else:
    #                 constant_mul *= x.constant_value()
    #         elif x.is_times():
    #             stack += x.args()
    #         else:
    #             new_args.append(x)

    #     const = None
    #     if is_algebraic:
    #         from pysmt.constants import Numeral

    #         const = self.manager._Algebraic(Numeral(constant_mul))
    #     elif ttype.is_real_type():
    #         const = self.manager.Real(constant_mul)
    #     else:
    #         assert ttype.is_int_type()
    #         const = self.manager.Int(constant_mul)

    #     if const.is_zero():
    #         return const
    #     else:
    #         if len(new_args) == 0:
    #             return const
    #         elif not const.is_one():
    #             new_args.append(const)

    #     # The block below handles the case of distributing a constant over the other terms in a product.  (* c t1 t2) --> (*+ (* c t1) (* c t2))
    #     new_new_args = []
    #     if not const.is_one():
    #         for arg in new_args:
    #             if arg != const:
    #                 if arg.is_plus() or arg.is_minus():
    #                     # Distribute constant over sum
    #                     arg_args = [
    #                         self.manager.Times(const, a).simplify()
    #                         for a in arg.args()
    #                     ]
    #                     new_arg = (
    #                         self.manager.Plus(arg_args)
    #                         if arg.is_plus()
    #                         else self.manager.Minus(*arg_args)
    #                     )
    #                 else:
    #                     # Fall through case
    #                     new_arg = self.manager.Times(arg, const)

    #                 # if not new_arg.is_symbol():
    #                 #     new_arg = new_arg.simplify()
    #                 new_new_args.append(new_arg)
    #                 # self.manager.Times(new_args)
    #     else:
    #         new_new_args = new_args

    #     ## FIXME conversion to sympy for simplification, needs conversion back to pysmt
    #     vars = formula.get_free_variables()
    #     var_map = {str(v): symbols(str(v)) for v in vars}
    #     expanded_formula = expand(sympify(str(formula), var_map))

    #     if len(new_new_args) > 1:
    #         new_new_args = sorted(new_new_args, key=FNode.node_id)
    #         return self.manager.Times(new_new_args)
    #     else:
    #         return new_new_args[0]

    #     # Handle combiing terms (* (b))

    #     # new_args = sorted(new_args, key=FNode.node_id)
    #     # return self.manager.Times(new_args)

    # def walk_plus(self, formula, args, **kwargs):
    #     to_sum = []
    #     to_sub = []
    #     constant_add = 0
    #     stack = list(args)
    #     ttype = self.env.stc.get_type(args[0])
    #     is_algebraic = False
    #     while len(stack) > 0:
    #         x = stack.pop()
    #         if x.is_constant():
    #             if x.is_algebraic_constant():
    #                 is_algebraic = True
    #             constant_add += x.constant_value()
    #         elif x.is_plus():
    #             stack += x.args()
    #         elif x.is_minus():
    #             to_sum.append(x.arg(0))
    #             to_sub.append(x.arg(1))
    #         elif x.is_times() and x.args()[-1].is_constant():
    #             const = x.args()[-1]
    #             const_val = const.constant_value()
    #             if const_val < 0:
    #                 new_times = list(x.args()[:-1])
    #                 if const_val != -1:
    #                     const_val = -const_val
    #                     if const.is_algebraic_constant():
    #                         from pysmt.constants import Numeral

    #                         const = self.manager._Algebraic(Numeral(const_val))
    #                     elif ttype.is_real_type():
    #                         const = self.manager.Real(const_val)
    #                     else:
    #                         assert ttype.is_int_type()
    #                         const = self.manager.Int(const_val)
    #                     new_times.append(const)
    #                 new_times = self.manager.Times(new_times)
    #                 to_sub.append(new_times)
    #             else:
    #                 to_sum.append(x)
    #         else:
    #             to_sum.append(x)

    #     const = None
    #     if is_algebraic:
    #         from pysmt.constants import Numeral

    #         const = self.manager._Algebraic(Numeral(constant_add))
    #     elif ttype.is_real_type():
    #         const = self.manager.Real(constant_add)
    #     else:
    #         assert ttype.is_int_type()
    #         const = self.manager.Int(constant_add)

    #     if len(to_sum) == 0 and len(to_sub) == 0:
    #         return const
    #     if not const.is_zero():
    #         to_sum.append(const)

    #     assert to_sum or to_sub

    #     res = self.manager.Plus(to_sum) if to_sum else None
    #     if all(x.is_constant() for x in to_sum):
    #         res = res.simplify()

    #     if to_sub:
    #         sub = self.manager.Plus(to_sub)
    #         if res:
    #             res = self.manager.Minus(res, sub)
    #         else:
    #             if ttype.is_int_type():
    #                 m_1 = self.manager.Int(-1)
    #             else:
    #                 assert ttype.is_real_type()
    #                 m_1 = self.manager.Real(-1)
    #             res = self.manager.Times(m_1, sub)
    #     return res

    def walk_pow(self, formula, args, **kwargs):
        # if args[0].is_real_constant():
        #     l = args[0].constant_value()
        #     r = args[1].constant_value()
        #     return self.manager.Real(l**r)

        # if args[0].is_int_constant():
        #     l = args[0].constant_value()
        #     r = args[1].constant_value()
        #     return self.manager.Int(l**r)

        # if args[0].is_algebraic_constant():
        #     from pysmt.constants import Numeral
        #     l = args[0].constant_value()
        #     r = args[1].constant_value()
        #     return self.manager._Algebraic(Numeral(l**r))

        env = get_env()
        self.manager = env._formula_manager

        return self.manager.Pow(args[0], args[1])


class SympyToPysmt(object):
    def sympyToPysmt(sympy_formula):
        pass
