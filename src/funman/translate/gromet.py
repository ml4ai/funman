"""
This module encodes Gromet models as SMTLib formulas.

"""
from automates.model_assembly.gromet.model.function_type import FunctionType
from automates.model_assembly.gromet.model.gromet_box_function import (
    GrometBoxFunction,
)
from automates.model_assembly.gromet.model.gromet_fn import GrometFN
from automates.model_assembly.gromet.model.gromet_fn_module import (
    GrometFNModule,
)
from automates.model_assembly.gromet.model.gromet_port import GrometPort
from automates.model_assembly.gromet.model.literal_value import LiteralValue
from automates.model_assembly.gromet.model.typed_value import TypedValue
from pysmt.shortcuts import And, Equals, Int, Symbol
from pysmt.typing import INT

from funman.model import Model
from funman.model.gromet import GrometModel
from funman.translate import Encoder, EncodingOptions


class GrometEncodingOptions(EncodingOptions):
    """
    Gromet encoding options.
    """


class GrometEncoder(Encoder):
    """
    Encodes Gromet models into SMTLib formulas.

    """

    def __init__(
        self,
        gromet_fn,
        config: GrometEncodingOptions = GrometEncodingOptions(
            num_steps=0, step_size=0
        ),
    ) -> None:
        super().__init__(config)
        self._gromet_fn = gromet_fn
        self.gromet_encoding_handlers = {
            str(GrometFNModule): self._gromet_fnmodule_to_smtlib,
            str(GrometFN): self._gromet_fn_to_smtlib,
            str(TypedValue): self._gromet_typed_value_to_smtlib,
            str(GrometPort): self._gromet_port_to_smtlib,
            str(GrometBoxFunction): self._gromet_box_function_to_smtlib,
            str(LiteralValue): self._gromet_literal_value_to_smtlib,
        }

    def encode_model(self, model: Model):
        """Convert the self._gromet_fn into a set of smtlib constraints.

        Returns:
            pysmt.Node: SMTLib object for constraints.
        """
        if isinstance(model, GrometModel):
            return self._to_smtlib(self._gromet_fn, stack=[])[0][1]
        else:
            raise Exception(
                f"GrometEncoder cannot encode a model of type: {type(model)}"
            )

    def _to_smtlib(self, node, stack=[]):
        """Convert the node into a set of smtlib constraints.

        Returns:
            pysmt.Node: SMTLib object for constraints.
        """
        return self.gromet_encoding_handlers[str(node.__class__)](
            node, stack=stack
        )

    def _get_stack_identifier(self, stack):
        return ".".join([name for (name, x) in stack])

    def _gromet_fnmodule_to_smtlib(self, node, stack=[]):
        stack.append((node.name, node))
        [(_, fn_constraints)] = self._to_smtlib(node.fn, stack=stack)
        # attr_constraints = And([self._to_smtlib(attr, stack=node) for attr in node.attributes])
        stack.pop()
        return [([Symbol(node.name, INT)], fn_constraints)]

    def _gromet_fn_to_smtlib(self, node, stack=[]):
        """Convert a fn node into constraints.  The constraints state that:
        - The function output (pof) is equal to the output of the box function (bf)

        Args:
            node (GrometFN): the function to encode
            stack (GrometFNModule, optional): GrometFNModule defining node. Defaults to None.

        Returns:
            pysmt.Node: Constraints encoding node.
        """
        # fn.pof[i] = fn.bf[fn.pof[i].box-1]

        stack.append(("fn", node))

        # Each iteration of this loop will generate a symbol
        # and an implementation for a box function output port (pof),
        # which appear in outputs as pairs

        outputs = []

        for j, bf_decl in enumerate(node.bf):
            # get all outputs for bf, store original index i
            bf_pofs = [
                (i, pof) for i, pof in enumerate(node.pof) if pof.box - 1 == j
            ]

            # get implementation for bf
            if hasattr(bf_decl, "contents") and bf_decl.contents:
                bf_impl = stack[0][1].attributes[bf_decl.contents - 1]
            else:
                bf_impl = bf_decl
            stack.append((f"bf[{j}]", bf_impl))
            [(bf_opo_symbols, phi_bf_impl)] = self._to_smtlib(
                bf_impl, stack=stack
            )
            stack.pop()

            # Bind the opo ports of the box function to the pofs of node
            port_bindings = []
            pof_symbols = []
            for i, (i_orig, pof) in enumerate(bf_pofs):
                # pof = bf.opo
                stack.append((f"pof[{i_orig}]", pof))
                [([pof_head], _)] = self._to_smtlib(pof, stack=stack)
                stack.pop()

                bf_opo = bf_opo_symbols[i]
                phi_bind_pof_opo = Equals(pof_head, bf_opo)
                port_bindings.append(phi_bind_pof_opo)

                if hasattr(pof, "name") and pof.name:
                    stack.append((f"{pof.name}", pof.name))
                    pof_name = Symbol(self._get_stack_identifier(stack), INT)
                    stack.pop()
                    pof_name_pof_binding = Equals(pof_name, pof_head)
                    port_bindings.append(pof_name_pof_binding)
                    pof_symbols.append(
                        pof_name
                    )  # Use pof name if present for outside reference
                else:
                    pof_symbols.append(
                        pof_head
                    )  # Otherwise use pof sybmol for outside reference

            phi = And(port_bindings + [phi_bf_impl])
            # symbol = Symbol(self._get_stack_identifier(stack), INT)
            outputs.append((pof_symbols, phi))

        # TODO implement the case for inputs
        inputs = []

        # Each iteration of the following loop will generate a
        # symbol and implementation for each outer output port (opo)
        # in terms of the wires in wfopo.

        opo_wires = []
        if node.wfopo:
            for i, wfopo in enumerate(node.wfopo):
                source_opo = node.opo[wfopo.src - 1]
                target_pof = node.pof[wfopo.tgt - 1]

                stack.append((f"opo[{wfopo.src-1}]", source_opo))
                source_symbol = Symbol(self._get_stack_identifier(stack), INT)
                stack.pop()
                stack.append((f"pof[{wfopo.tgt-1}]", target_pof))
                target_symbol = Symbol(self._get_stack_identifier(stack), INT)
                stack.pop()

                phi = Equals(source_symbol, target_symbol)
                opo_wires.append(([source_symbol], phi))

        # TODO implement case for input wires
        opi_wires = []

        stack.pop()

        # consolidate formulas for internals of function
        # need opo and opi symbols for binding and the
        # implementation of the internals
        out_symbols = [s for (ss, _) in opo_wires for s in ss]
        in_symbols = [s for (ss, _) in opi_wires for s in ss]
        impl = And(
            [imp for (_, imp) in outputs + inputs + opo_wires + opi_wires]
        )

        return [(out_symbols + in_symbols, impl)]

    def _gromet_port_to_smtlib(self, node, stack=[]):
        # stack.append((node.name, node))
        name = self._get_stack_identifier(stack)
        # stack.pop()
        return [([Symbol(f"{name}", INT)], None)]

    def _gromet_box_function_to_smtlib(self, node, stack=[]):
        phi = None
        if node.function_type == FunctionType.EXPRESSION:
            function_node = stack[0][1].attributes[node.contents - 1]
            stack.append((f"fn", function_node))
            [(symbols, phi)] = self._to_smtlib(function_node, stack=stack)
            stack.pop()
        elif node.function_type == FunctionType.LITERAL:
            function_node = node.value
            stack.append((f"value", function_node))
            [(symbols, phi)] = self._to_smtlib(function_node, stack=stack)
            stack.pop()
        else:
            raise ValueError(
                f"node.function_type = {node.function_type} is not supported."
            )

        return [(symbols, phi)]

    def _gromet_typed_value_to_smtlib(self, node, stack=[]):
        phi = None
        if node.type == "FN":  # FIXME find def for the string
            # define output ports
            # map i/o ports
            #  wfopo
            stack.append((f"value", node.value))
            [(symbols, phi)] = self._to_smtlib(node.value, stack=stack)
            stack.pop()

        return [(symbols, phi)]

    def _gromet_literal_value_to_smtlib(self, node, stack=[]):
        value_type = node.value_type
        value = None
        value_enum = None

        if value_type == "Integer":
            value = Int(node.value)
            value_enum = INT
        else:
            raise ValueError(
                f"literal_value of type {value_type} not supported."
            )

        literal = Symbol(f"{self._get_stack_identifier(stack)}", value_enum)
        phi = Equals(literal, value)

        return [([literal], phi)]
