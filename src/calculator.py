import ast
import json
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation, localcontext


MAX_CHARS = 2000
MAX_LINES = 24
MAX_VALUE = Decimal("1e15")
TOOL = {
    "name": "calculate",
    "description": (
        "Exact decimal calculator. Send a program with one assignment per line, for example "
        "contribution = round(min(4210.55, 4500.00) * 0.06, 2). Use decimal numbers, names assigned on "
        "earlier lines, + - * /, parentheses, min(...), max(...), and round(x, 2), which rounds half up to "
        "the cent as R2 requires. Returns the value of every assigned name."
    ),
    "parameters": {
        "type": "object",
        "properties": {"program": {"type": "string", "description": "One assignment per line"}},
        "required": ["program"],
        "additionalProperties": False,
    },
}


class CalculatorError(ValueError):
    pass


def value_of(node, names, source):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        text = ast.get_source_segment(source, node)
        if text is None:
            raise CalculatorError("Unreadable number")
        return Decimal(text)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise CalculatorError(f"Unknown name: {node.id}")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = value_of(node.operand, names, source)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = value_of(node.left, names, source), value_of(node.right, names, source)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name, arguments = node.func.id, node.args
        if name in ("min", "max") and arguments:
            values = [value_of(argument, names, source) for argument in arguments]
            return min(values) if name == "min" else max(values)
        if name == "round" and len(arguments) == 2:
            digits = arguments[1]
            if not (isinstance(digits, ast.Constant) and type(digits.value) is int and 0 <= digits.value <= 4):
                raise CalculatorError("round needs a whole number of digits from 0 to 4")
            return value_of(arguments[0], names, source).quantize(Decimal(1).scaleb(-digits.value),
                                                                  rounding=ROUND_HALF_UP)
    raise CalculatorError("Unsupported expression; use numbers, names, + - * /, min, max, round(x, 2)")


def run_program(program):
    if not isinstance(program, str) or not program.strip() or len(program) > MAX_CHARS:
        raise CalculatorError(f"Program must be 1 to {MAX_CHARS} characters")
    try:
        tree = ast.parse(program.strip(), mode="exec")
    except SyntaxError:
        raise CalculatorError("Syntax error; write one assignment per line, such as total = 1 + 2") from None
    if not tree.body or len(tree.body) > MAX_LINES:
        raise CalculatorError(f"Use 1 to {MAX_LINES} assignments")
    names = {}
    source = program.strip()
    with localcontext() as context:
        context.prec = 34
        for statement in tree.body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                raise CalculatorError("Each line must assign one name, such as total = 1 + 2")
            try:
                value = value_of(statement.value, names, source)
            except (InvalidOperation, DivisionByZero, ValueError, TypeError) as error:
                if isinstance(error, CalculatorError):
                    raise
                raise CalculatorError("Arithmetic error") from None
            if not value.is_finite() or abs(value) >= MAX_VALUE:
                raise CalculatorError("Result out of range")
            names[statement.targets[0].id] = value
    return {name: format(value, "f") for name, value in names.items()}


def calculate(program):
    try:
        return {"results": run_program(program)}
    except CalculatorError as error:
        return {"error": str(error)}


def response_text(program):
    return json.dumps(calculate(program))
