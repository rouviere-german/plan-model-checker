"""Ground propositions: a proposition name applied to concrete resources, e.g. ``backup_ok[/data]``.

Two built-in, zero-argument events frame every trace: ``start`` holds only at position 0 and
``end`` holds only at the final position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["END", "RESERVED_PROPS", "START", "GroundProp"]


@dataclass(frozen=True, slots=True, order=True)
class GroundProp:
    name: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}[{','.join(self.args)}]"

    def to_json(self) -> dict[str, Any]:
        return {"prop": self.name, "args": list(self.args)}


START = GroundProp("start")
END = GroundProp("end")
RESERVED_PROPS = frozenset({"start", "end", "true", "false"})
