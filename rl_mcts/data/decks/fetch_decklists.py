#!/usr/bin/env python3
import subprocess
import zipfile
import csv
import json
import os
import sys
import glob
from pathlib import Path

COMPETITION = "pokemon-tcg-ai-battle"
DECKLISTS_DIR = Path("decklists")


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{result.stderr}", file=sys.stderr)
        return None
    return result.stdout.strip()


def parse_table(output):
    if not output:
        return []
    lines = [l for l in output.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split("  ") if h.strip()]
    rows = []
    for line in lines[1:]:
        if set(line.strip()) <= set("- "):
            continue
        parts = [p.strip() for p in line.split("  ") if p.strip()]
        if parts:
            rows.append(dict(zip(header, parts)))
    return rows


def fetch_decklists(min_score: float):
    DECKLISTS_DIR.mkdir(exist_ok=True)

    # Download leaderboard
    print("Downloading leaderboard...")
    out = run(["kaggle", "competitions", "leaderboard", COMPETITION, "-d"])
    if out is None:
        return

    zip_files = glob.glob("*.zip")
    lb_zip = next((f for f in zip_files if "leaderboard" in f.lower()), None)
    if lb_zip is None:
        # kaggle downloads to cwd, find the zip
        zip_files = sorted(glob.glob("*.zip"), key=os.path.getmtime, reverse=True)
        lb_zip = zip_files[0] if zip_files else None

    if lb_zip is None:
        print("No leaderboard zip found.", file=sys.stderr)
        return

    # Extract CSV
    lb_csv_path = None
    with zipfile.ZipFile(lb_zip) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                zf.extract(name, ".")
                lb_csv_path = name
                break
    os.remove(lb_zip)

    if lb_csv_path is None:
        print("No CSV in leaderboard zip.", file=sys.stderr)
        return

    # Read leaderboard
    teams = []
    with open(lb_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row.get("Score", 0) or 0)
            if score >= min_score:
                teams.append(row)

    print(f"Teams with score >= {min_score}: {len(teams)}")

    for team in teams:
        team_id = team.get("TeamId") or team.get("teamId") or team.get("Id")
        raw_name = team.get("TeamName") or team.get("teamName") or team.get("Name") or "unknown"
        team_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in raw_name)
        print(f"\nTeam: {team_name} (id={team_id})")

        subs_out = run(["kaggle", "competitions", "team-submissions", str(team_id)])
        subs = parse_table(subs_out)

        for sub in subs:
            sub_id = sub.get("id")
            pub_score = float(sub.get("publicScore", 0) or 0)
            if pub_score < min_score:
                continue

            out_csv = DECKLISTS_DIR / f"{team_name}-{team_id}-{sub_id}-deck.csv"
            if out_csv.exists():
                print(f"  Skip {sub_id} (already exists)")
                continue

            print(f"  Submission {sub_id} score={pub_score}")

            eps_out = run(["kaggle", "competitions", "episodes", str(sub_id)])
            eps = parse_table(eps_out)
            if not eps:
                print(f"    No episodes for submission {sub_id}")
                continue

            ep_id = eps[0].get("id")
            print(f"    Episode {ep_id} — downloading replay...")

            run(["kaggle", "competitions", "replay", str(ep_id)])

            json_files = sorted(glob.glob("*.json"), key=os.path.getmtime, reverse=True)
            if not json_files:
                print("    No JSON replay found.", file=sys.stderr)
                continue
            json_path = json_files[0]

            try:
                with open(json_path) as f:
                    data = json.load(f)

                action_lists = data["steps"][0][0]["visualize"][0]["action"]

                # pick the player index matching this team
                team_names = data.get("info", {}).get("TeamNames", [])
                norm = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
                try:
                    player_idx = next(i for i, n in enumerate(team_names) if norm(n) == team_name)
                except StopIteration:
                    player_idx = 0
                    print(f"    Team '{team_name}' not in {team_names}, using player 0", file=sys.stderr)

                deck = action_lists[player_idx]

                with open(out_csv, "w", newline="") as f:
                    f.write(",".join(str(x) for x in deck) + "\n")
                print(f"    Saved {out_csv}")
            except KeyError as e:
                print(f"    Parse error: missing key {e}", file=sys.stderr)
                steps = data.get("steps", [])
                if steps:
                    s0 = steps[0]
                    print(f"    steps[0] preview: {json.dumps(s0)[:1000]}", file=sys.stderr)
            except (IndexError, json.JSONDecodeError) as e:
                print(f"    Parse error: {e}", file=sys.stderr)
            finally:
                if os.path.exists(json_path):
                    os.remove(json_path)

    os.remove(lb_csv_path)
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <min_score>")
        sys.exit(1)
    fetch_decklists(float(sys.argv[1]))
