"""
Feature selector class.

This file is a part of BdPy
"""


from typing import ClassVar, Dict, List, Optional, Tuple


class FeatureSelector(object):
    """Feature selector class.

    Parameters
    ----------
    expression : str
        Selection command

    Attributes
    ----------
    expression : str
        Selection command
    token : tuple
        Tokens
    rpn : tuple
        Tokens in reversed polish notation
    """

    # Class variables ##################
    signs: ClassVar[Tuple[str, ...]] = ('(', ')')
    operators: ClassVar[Tuple[str, ...]] = ('=', '|', '&', '+', '-', '@')

    __op_order: ClassVar[Dict[str, int]] = {
        '=': 10,
        '|': 5,
        '&': 5,
        '+': 5,
        '-': 5,
        '@': 3,
        '(': -1,
        ')': -1,
    }

    # Methods ##########################

    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.token = self.lexical_analysis(self.expression)
        self.rpn = self.parse(self.token)

        self.index: Optional[int] = None

    def lexical_analysis(self, expression: str) -> Tuple[str, ...]:
        """Tokenize selection command."""
        str_buf = ''
        output_buf: List[str] = []

        i = 0
        while i < len(expression):
            if expression[i] == ' ':
                # Ignore a white-space
                i += 1
                continue
            elif expression[i] == '"':
                i += 1
                while expression[i] != '"':
                    str_buf += expression[i]
                    i += 1
                i += 1
                continue
            elif expression[i] == "'":
                i += 1
                while expression[i] != "'":
                    str_buf += expression[i]
                    i += 1
                i += 1
                continue
            elif self.signs.count(expression[i]) or self.operators.count(expression[i]):
                if len(str_buf) > 0:
                    output_buf.append(str_buf)
                    str_buf = ''

                output_buf.append(expression[i])
            else:
                str_buf += expression[i]

            i += 1

        if len(str_buf) > 0:
            output_buf.append(str_buf)
            str_buf = ''

        # Convert '+' to '|'
        output_buf = ['|' if a == '+' else a for a in output_buf]

        return tuple(output_buf)

    def parse(self, token_list: Tuple[str, ...]) -> Tuple[str, ...]:
        """Parse selection command."""
        out_que: List[str] = []
        op_stack: List[str] = []

        for token in token_list:

            if self.operators.count(token):
                while op_stack:
                    if self.__op_order[token] > self.__op_order[op_stack[-1]]:
                        break
                    out_que.append(op_stack.pop())

                op_stack.append(token)
            elif token == '(':
                op_stack.append('(')
            elif token == ')':
                while op_stack:
                    if op_stack[-1] == '(':
                        op_stack.pop()
                    else:
                        out_que.append(op_stack.pop())
            else:
                out_que.append(token)

        while op_stack:
            out_que.append(op_stack.pop())

        return tuple(out_que)
