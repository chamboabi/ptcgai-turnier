"""Bundle a tournament submission from a data/player/<name>/ package.

A submission folder contains exactly:
    main.py           agent logic, generated from the main.py template with
                       this player's config baked in (self-contained; imports
                       only torch + bundled cg/ + bundled rewards/)
    deck.csv           60 card IDs, one per line
    player_reward.py   the player package's reward.py (make_shape entrypoint)
    rewards/            shared reward primitives (core.py, shapes.py, ...)
    cg/                 the untouched game library
    model.pth           optional trained weights (main.py uses random init if absent)

Writes to dist/<player>_submission/ and also packs that folder into
dist/<player>-submission.tar.gz, with main.py at the archive root (ready to
upload as-is, no submission-name folder in the way).

Usage:
    from build import build_submission

    build_submission("evee")
    build_submission("evee", weights="../out/model_best.pth")  # override active weights

Or from the shell:
    python build.py evee
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
PLAYER_DIR = ROOT / "data" / "player"

SRC_MAIN = HERE / "main.py"
SRC_CG = HERE / "cg"
SRC_REWARDS = ROOT / "rewards"
SRC_PREDICT = HERE / "deck_predict_lite.py"
SRC_ARCHETYPES = ROOT / "data" / "archetypes.json"
SRC_ROOT_CONFIG = ROOT / "config.json"

# main.py constant name -> (config.json section, key) it gets baked in from.
_CONFIG_CONSTANTS = {
    "D_MODEL": ("model", "d_model"),
    "NUM_HEADS": ("model", "num_heads"),
    "D_FEEDFORWARD": ("model", "d_feedforward"),
    "NUM_LAYERS_ENCODER": ("model", "num_layers_encoder"),
    "NUM_LAYERS_DECODER": ("model", "num_layers_decoder"),
    "NUM_WORDS_ENCODER": ("model", "num_words_encoder"),
    "ENCODER_SIZE": ("model", "encoder_size"),
    "DECODER_MAIN_FEATURE": ("model", "decoder_main_feature"),
    "DECODER_ATTACK_OFFSET": ("model", "decoder_attack_offset"),
    "SEARCH_COUNT": ("mcts", "search_count"),
    "MAX_ACTION_COMBINATIONS": ("mcts", "max_action_combinations"),
    "UCB_EXPLORATION": ("mcts", "ucb_exploration"),
    "POLICY_TEMPERATURE": ("mcts", "policy_temperature"),
    "UNVISITED_PENALTY": ("mcts", "unvisited_penalty"),
    "DETERMINIZATIONS": ("mcts", "determinizations"),
}


def _resolve_player_dir(player: str | Path) -> Path:
    """A player is a name looked up under data/player/, or a direct path to such a folder."""
    path = Path(player)
    if not path.is_dir():
        path = PLAYER_DIR / player
    if not path.is_dir():
        raise FileNotFoundError(f"player package not found: {player}")
    return path


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(player_dir: Path) -> dict:
    """Root config.json, deep-merged with the player's config.json if it has one."""
    cfg = json.loads(SRC_ROOT_CONFIG.read_text())
    override_path = player_dir / "config.json"
    if override_path.is_file():
        cfg = _deep_merge(cfg, json.loads(override_path.read_text()))
    return cfg


def _resolve_weights(player_dir: Path, weights_override: str | Path | None) -> Path | None:
    """Explicit --weights wins; otherwise the package's weights/manifest.json active version."""
    if weights_override is not None:
        wpath = Path(weights_override)
        if not wpath.exists():
            raise FileNotFoundError(f"Weights file not found: {wpath}")
        return wpath
    manifest_path = player_dir / "weights" / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["versions"][manifest["active"]]
    return player_dir / "weights" / entry["file"]


def _render_main(cfg: dict) -> str:
    """Fill the main.py template's config constants in from cfg (see _CONFIG_CONSTANTS)."""
    text = SRC_MAIN.read_text()

    for const_name, (section, key) in _CONFIG_CONSTANTS.items():
        value = cfg[section][key]
        text, n = re.subn(rf"(?m)^{const_name} = .+$", f"{const_name} = {value!r}", text)
        if n != 1:
            raise RuntimeError(f"expected exactly one '{const_name} = ...' line in main.py, found {n}")

    return text


def build_submission(
    player: str | Path,
    out_dir: str | Path | None = None,
    weights: str | Path | None = None,
) -> Path:
    """Assemble a tournament submission from data/player/<player>/ (or a direct path to one).

    Writes the folder to out_dir (default dist/<player>_submission/) then packs
    it into dist/<player>-submission.tar.gz. Returns the tarball path.
    """
    player_dir = _resolve_player_dir(player)
    if out_dir is None:
        out_dir = HERE / "dist" / f"{player_dir.name}_submission"
    deck_path = player_dir / "deck.csv"
    reward_path = player_dir / "reward.py"
    if not deck_path.is_file():
        raise FileNotFoundError(f"no deck.csv in {player_dir}")
    if not reward_path.is_file():
        raise FileNotFoundError(f"no reward.py in {player_dir}")
    if not SRC_MAIN.exists():
        raise FileNotFoundError(f"Missing agent template: {SRC_MAIN}")
    if not SRC_CG.is_dir():
        raise FileNotFoundError(f"Missing cg/ library: {SRC_CG}")
    for src in (SRC_REWARDS, SRC_PREDICT, SRC_ARCHETYPES):
        if not src.exists():
            raise FileNotFoundError(f"Missing required asset: {src}")

    cfg = _load_config(player_dir)
    weights_path = _resolve_weights(player_dir, weights)

    # fresh build dir
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) main.py, generated from the template with this player's config baked in
    (out_dir / "main.py").write_text(_render_main(cfg))

    # 2) shared reward primitives + this player's reward.py
    shutil.copytree(SRC_REWARDS, out_dir / "rewards", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(reward_path, out_dir / "player_reward.py")
    shutil.copy2(SRC_PREDICT, out_dir / "deck_predict.py")
    shutil.copy2(SRC_ARCHETYPES, out_dir / "archetypes.json")

    # 3) deck.csv — one ID per line (main.py reader expects 60 lines)
    ids = [int(tok) for tok in deck_path.read_text().replace(",", "\n").split() if tok.strip()]
    if len(ids) != 60:
        raise ValueError(f"Deck must have exactly 60 cards, got {len(ids)} ({deck_path}).")
    (out_dir / "deck.csv").write_text("\n".join(str(c) for c in ids) + "\n")

    # 4) cg/ untouched (skip caches so the copy stays clean)
    shutil.copytree(SRC_CG, out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 5) optional weights
    weights_label = "OMITTED (random init)"
    if weights_path is not None:
        shutil.copy2(weights_path, out_dir / "model.pth")
        weights_label = f"included ({weights_path.name})"

    print(f"Built submission: {out_dir}")
    print(f"  main.py           ({(out_dir / 'main.py').stat().st_size} bytes, config from {player_dir.name}/config.json + root config.json)")
    print(f"  player_reward.py  ({reward_path})")
    print(f"  rewards/          + deck_predict.py + archetypes.json bundled")
    print(f"  deck.csv          (60 cards)")
    print(f"  cg/               ({sum(1 for _ in (out_dir / 'cg').rglob('*') if _.is_file())} files)")
    print(f"  model.pth         {weights_label}")

    archive_base = HERE / "dist" / f"{player_dir.name}-submission"
    archive = shutil.make_archive(str(archive_base), "gztar", root_dir=out_dir)
    print(f"Tarball: {archive} (main.py at archive root)")
    return Path(archive)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle a tournament submission from a data/player/<name>/ package.")
    ap.add_argument("player", help="Player package name under data/player/, or a direct path to one.")
    ap.add_argument("--out", default=None, help="Output directory (default: dist/<player>_submission/).")
    ap.add_argument("--weights", default=None, help="Override checkpoint (.pth); defaults to the package's active weights.")
    args = ap.parse_args()
    build_submission(args.player, out_dir=args.out, weights=args.weights)


if __name__ == "__main__":
    main()
