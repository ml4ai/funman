import pysmt
from pysmt.fnode import FNode


class FUNMANSimplifier(pysmt.simplifier.Simplifier):
    def walk_times(self, formula, args, **kwargs):
        new_args = []
        constant_mul = 1
        stack = list(args)
        ttype = self.env.stc.get_type(args[0])
        is_algebraic = False
        while len(stack) > 0:
            x = stack.pop()
            if x.is_constant():
                if x.is_algebraic_constant():
                    is_algebraic = True
                if x.is_zero():
                    constant_mul = 0
                    break
                else:
                    constant_mul *= x.constant_value()
            elif x.is_times():
                stack += x.args()
            else:
                new_args.append(x)

        const = None
        if is_algebraic:
            from pysmt.constants import Numeral

            const = self.manager._Algebraic(Numeral(constant_mul))
        elif ttype.is_real_type():
            const = self.manager.Real(constant_mul)
        else:
            assert ttype.is_int_type()
            const = self.manager.Int(constant_mul)

        if const.is_zero():
            return const
        else:
            if len(new_args) == 0:
                return const
            elif not const.is_one():
                new_args.append(const)

        new_new_args = []
        if not const.is_one():
            for arg in new_args:
                if arg != const:
                    new_arg = self.manager.Times(arg, const)
                    if not arg.is_symbol():
                        arg = arg.simplify()
                    new_new_args.append(new_arg)
                    self.manager.Times(new_args)
        else:
            new_new_args = new_args
        if len(new_new_args) > 1:
            new_new_args = sorted(new_new_args, key=FNode.node_id)
            return self.manager.Times(new_new_args)
        else:
            return new_new_args[0]

        # new_args = sorted(new_args, key=FNode.node_id)
        # return self.manager.Times(new_args)

    def walk_plus(self, formula, args, **kwargs):
        to_sum = []
        to_sub = []
        constant_add = 0
        stack = list(args)
        ttype = self.env.stc.get_type(args[0])
        is_algebraic = False
        while len(stack) > 0:
            x = stack.pop()
            if x.is_constant():
                if x.is_algebraic_constant():
                    is_algebraic = True
                constant_add += x.constant_value()
            elif x.is_plus():
                stack += x.args()
            elif x.is_minus():
                to_sum.append(x.arg(0))
                to_sub.append(x.arg(1))
            elif x.is_times() and x.args()[-1].is_constant():
                const = x.args()[-1]
                const_val = const.constant_value()
                if const_val < 0:
                    new_times = list(x.args()[:-1])
                    if const_val != -1:
                        const_val = -const_val
                        if const.is_algebraic_constant():
                            from pysmt.constants import Numeral

                            const = self.manager._Algebraic(Numeral(const_val))
                        elif ttype.is_real_type():
                            const = self.manager.Real(const_val)
                        else:
                            assert ttype.is_int_type()
                            const = self.manager.Int(const_val)
                        new_times.append(const)
                    new_times = self.manager.Times(new_times)
                    to_sub.append(new_times)
                else:
                    to_sum.append(x)
            else:
                to_sum.append(x)

        const = None
        if is_algebraic:
            from pysmt.constants import Numeral

            const = self.manager._Algebraic(Numeral(constant_add))
        elif ttype.is_real_type():
            const = self.manager.Real(constant_add)
        else:
            assert ttype.is_int_type()
            const = self.manager.Int(constant_add)

        if len(to_sum) == 0 and len(to_sub) == 0:
            return const
        if not const.is_zero():
            to_sum.append(const)

        assert to_sum or to_sub

        res = self.manager.Plus(to_sum) if to_sum else None
        if all(x.is_constant() for x in to_sum):
            res = res.simplify()

        if to_sub:
            sub = self.manager.Plus(to_sub)
            if res:
                res = self.manager.Minus(res, sub)
            else:
                if ttype.is_int_type():
                    m_1 = self.manager.Int(-1)
                else:
                    assert ttype.is_real_type()
                    m_1 = self.manager.Real(-1)
                res = self.manager.Times(m_1, sub)
        return res
