"""Game recorder — wraps a battle session and saves a full log for later review."""
import json
import time
from pathlib import Path

from .game import battle_finish, battle_select, battle_start, visualize_data
from .sim import set_seed


class GameRecorder:
    """Record every observation and selection during a game.

    Usage::

        rec = GameRecorder(seed=42)
        obs, start_data = rec.start(deck0, deck1)
        while obs["current"]["result"] < 0:
            selection = agent(obs)
            obs = rec.select(selection)
        rec.finish()
        rec.save("games/game_001.json")
        rec.save_visualizer("games/game_001_vis.json")

    Open debug/visualizer.html in a browser and pick the *_vis.json file.

    Review later::

        for step in GameRecorder.load("games/game_001.json"):
            print(step["selection"], step["obs"]["current"]["result"])
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
        self._deck0: list[int] = []
        self._deck1: list[int] = []
        self._steps: list[dict] = []
        self._start_obs: dict | None = None
        self._current_obs: dict | None = None
        self._obs_log: list = []    # obs the agent saw before each action; index 0 is ""
        self._action_log: list = [] # action taken at each step; index 0 is None
        self._vis_json: str | None = None  # raw visualize_data() output captured in finish()

    def start(self, deck0: list[int], deck1: list[int]) -> tuple[dict, object]:
        set_seed(self.seed)
        self._deck0 = list(deck0)
        self._deck1 = list(deck1)
        obs, start_data = battle_start(deck0, deck1)
        self._start_obs = obs
        self._current_obs = obs
        self._obs_log = [""]
        self._action_log = [None]
        return obs, start_data

    def select(self, selection: list[int]) -> dict:
        # Record what the agent saw and chose before advancing the state
        obs_snapshot = {k: v for k, v in (self._current_obs or {}).items()
                        if k != "search_begin_input"}
        self._obs_log.append(obs_snapshot)
        self._action_log.append(list(selection))

        obs = battle_select(selection)
        self._current_obs = obs
        self._steps.append({"selection": list(selection), "obs": obs})
        return obs

    def finish(self) -> None:
        # Capture the full replay before freeing the battle pointer
        self._vis_json = visualize_data()
        battle_finish()

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        record = {
            "seed": self.seed,
            "deck0": self._deck0,
            "deck1": self._deck1,
            "start_obs": self._start_obs,
            "steps": self._steps,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(record, indent=2))

    def save_visualizer(self, path: str | Path) -> None:
        """Save a JSON file readable by the browser visualizer (visualizer.html).

        Must be called after finish().  The output is the raw visualize list
        understood by ptcgvis.heroz.jp — open visualizer.html and pick this file.
        """
        if self._vis_json is None:
            raise RuntimeError("Call finish() before save_visualizer().")

        vis = json.loads(self._vis_json)
        for i in range(len(vis)):
            vis[i]["obs"] = self._obs_log[i] if i < len(self._obs_log) else ""
            vis[i]["action"] = [self._action_log[i], self._action_log[i]] \
                if i < len(self._action_log) else [None, None]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(vis))

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        """Return a list of steps.  Each step has 'selection' and 'obs'.
        The initial observation (before any selection) is in step index -1
        accessible via GameRecorder.load_record(path)['start_obs'].
        """
        data = json.loads(Path(path).read_text())
        return data["steps"]

    @staticmethod
    def load_record(path: str | Path) -> dict:
        """Return the full saved record dict (seed, decks, start_obs, steps)."""
        return json.loads(Path(path).read_text())

    @staticmethod
    def replay(path: str | Path):
        """Re-run the saved game through the engine using the stored seed.

        Yields (obs, selection) for each step — obs is what the agent saw,
        selection is what it chose.  Useful for sanity-checking or feeding
        the game log into a visualiser.
        """
        record = GameRecorder.load_record(path)
        set_seed(record["seed"])
        obs, _ = battle_start(record["deck0"], record["deck1"])
        for step in record["steps"]:
            yield obs, step["selection"]
            obs = battle_select(step["selection"])
        battle_finish()
