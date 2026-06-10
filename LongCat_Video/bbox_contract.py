from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_box(value: Any, role: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{role} bbox must contain exactly four numeric values.")
    if not all(_is_number(item) for item in value):
        raise ValueError(f"{role} bbox must contain only numeric values.")
    return [int(item) for item in value]


def _parse_others(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("others bbox must be a flat list or a list of boxes.")
    if len(value) == 0:
        return []
    if all(_is_number(item) for item in value):
        if len(value) % 4 != 0:
            raise ValueError("others bbox flat list length must be a multiple of four.")
        return [int(item) for item in value]
    flattened: list[int] = []
    for index, box in enumerate(value):
        flattened.extend(_parse_box(box, f"others[{index}]"))
    return flattened


def parse_person_boxes(text: str | None) -> list[list[int] | list[int]] | None:
    if text is None or not str(text).strip():
        return None
    try:
        parsed = ast.literal_eval(f"[{text}]")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("p_box must be formatted as person1, person2, optional others boxes.") from exc
    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("p_box requires at least person1 and person2 boxes.")
    person1 = _parse_box(parsed[0], "person1")
    person2 = _parse_box(parsed[1], "person2")
    result: list[list[int] | list[int]] = [person1, person2]
    if len(parsed) > 2:
        others = _parse_others(parsed[2])
        if others is not None:
            result.append(others)
    if len(parsed) > 3:
        extra = _parse_others(parsed[3:])
        if extra:
            if len(result) == 2:
                result.append(extra)
            else:
                result[2].extend(extra)
    return result
