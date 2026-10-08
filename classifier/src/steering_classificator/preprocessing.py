"""The same equation preprocessing used by CognitiveRouting.ipynb."""
from __future__ import annotations

import re


def normalize_equation(value: str) -> str:
    text = str(value or "")
    starts = [match.start() for match in re.finditer(r"\\boxed\{", text)]
    if starts:
        start = starts[-1] + len(r"\boxed{")
        depth, index = 1, start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            text = text[start:index - 1].strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"e\^\{([^{}]+)\}", r"exp(\1)", text)
    return re.sub(r"\s+", "", text)


def tokenize_math(expression: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+|\+|\-|\*|\/|\(|\)|=|\^|\{|\}|_|'", expression)


def preprocess_equation(equation: str) -> str:
    return " ".join(tokenize_math(normalize_equation(equation)))


def equation_key(equation: str) -> str:
    """Identify equations that become the same input to the classifier."""
    return preprocess_equation(str(equation or "").replace("−", "-").replace("–", "-"))
