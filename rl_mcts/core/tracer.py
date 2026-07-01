"""Readable per-turn action log, built from GameRecorder log events."""

from __future__ import annotations

from cg.api import LogType


def _name(names: dict[int, str], cid: int | None) -> str:
    """Card name for a log id, falling back to #id."""
    if cid is None:
        return "?"
    return names.get(cid, f"#{cid}")


def format_log(log: dict, names: dict[int, str]) -> str | None:
    """One readable line for an interesting log event, or None to skip noise."""
    t = log.get("type")
    n = lambda key: _name(names, log.get(key))
    if t == LogType.DRAW:
        return f"draw {n('cardId')}"
    if t == LogType.PLAY:
        return f"play {n('cardId')}"
    if t == LogType.ATTACH:
        return f"attach {n('cardId')} -> {n('cardIdTarget')}"
    if t == LogType.EVOLVE:
        return f"evolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.DEVOLVE:
        return f"devolve {n('cardIdTarget')} -> {n('cardId')}"
    if t == LogType.SWITCH:
        return f"switch active {n('cardIdActive')} <-> bench {n('cardIdBench')}"
    if t == LogType.CHANGE:
        return f"change active {n('cardIdBefore')} -> {n('cardIdAfter')}"
    if t == LogType.ATTACK:
        return f"ATTACK with {n('cardId')} (attackId {log.get('attackId')})"
    if t == LogType.HP_CHANGE:
        v = log.get("value")
        if v:
            return f"hp {n('cardId')} {v:+d}"
    return None


class TurnPathTracer:
    """Accumulate a player's realized actions over a turn; flush at TURN_END."""

    def __init__(self, names: dict[int, str], p0_label: str = "P0", p1_label: str = "P1", enabled: bool = True):
        self.names = names
        self.labels = (p0_label, p1_label)
        self.enabled = enabled
        self.lines: list[str] = []
        self.owner: int | None = None
        self.turn = 0

    def feed(self, logs: list[dict], turn: int) -> None:
        for log in logs or []:
            t = log.get("type")
            if t == LogType.TURN_START:
                self.lines = []
                self.owner = log.get("playerIndex")
                self.turn = turn
            elif t == LogType.TURN_END:
                self._flush()
                self.lines = []
                self.owner = None
            else:
                line = format_log(log, self.names)
                if line is not None:
                    self.lines.append(line)

    def _flush(self) -> None:
        if not self.enabled or self.owner is None:
            return
        tag = self.labels[self.owner]
        print(f"\n=== P{self.owner} ({tag}) turn {self.turn} path ===")
        for ln in self.lines:
            print(f"    {ln}")
        if not self.lines:
            print("    (no actions)")
        print("=== end turn ===")
