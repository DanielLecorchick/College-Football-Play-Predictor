"""
College Football Play Prediction - Play Scraper
Author: Daniel Lecorchick
Purpose: Get all power 4 plays in using ESPN 2024 API
"""

import requests
import json
import time
from collections import deque

#input and output files
GAME_ID_FILE = "espn_game_ids_2024.txt"
OUTPUT_JSON = "all_plays_2024.json"


def classify_play(play_type, play_text):
    """Classify play as Run or Pass based on ESPN text."""
    t = (play_type or "").lower()
    txt = (play_text or "").lower()
    if "pass" in t or "pass" in txt:
        return "Pass"
    if "rush" in t or "run" in txt:
        return "Run"
    return None


def scrape_game(game_id):
    """Scrape a single game's play-by-play data and return structured info."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_id}"
    try:
        data = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"Error fetching {game_id}: {e}")
        return None

    # Handle invalid or incomplete data
    if "drives" not in data or "header" not in data:
        print(f"Skipping {game_id}, missing data structure.")
        return None

    try:
        home_team = data["header"]["competitions"][0]["competitors"][0]["team"]["displayName"]
        away_team = data["header"]["competitions"][0]["competitors"][1]["team"]["displayName"]
    except Exception:
        return None

    plays = []
    team_queues = {}

    #goes through every drive in every game
    for drive in data.get("drives", {}).get("previous", []):
        team = drive.get("team", {}).get("displayName")
        if not team:
            continue
        if team not in team_queues:
            team_queues[team] = deque(maxlen=3)

        for play in drive.get("plays", []):
            offense = play.get("start", {}).get("team", {}).get("displayName") or team
            play_type = play.get("type", {}).get("text")
            play_text = play.get("text")
            label = classify_play(play_type, play_text)
            if label not in ("Run", "Pass"):
                continue

            # Previous plays context
            prev_plays = list(team_queues[offense])
            while len(prev_plays) < 3:
                prev_plays.insert(0, None)

            prev_types = [p["label_run_pass"] if p else None for p in prev_plays]
            prev_yards = [p["yards_gained"] if p else None for p in prev_plays]
            prev_downs = [p["down"] if p else None for p in prev_plays]
            prev_dist = [p["distance"] if p else None for p in prev_plays]

            home_score = play.get("homeScore")
            away_score = play.get("awayScore")
            if offense == home_team:
                score_diff = (home_score or 0) - (away_score or 0)
            elif offense == away_team:
                score_diff = (away_score or 0) - (home_score or 0)
            else:
                score_diff = None

            plays.append({
                "offense_team": offense,
                "down": play.get("start", {}).get("down"),
                "distance": play.get("start", {}).get("distance"),
                "yard_line": play.get("start", {}).get("yardsToEndzone"),
                "period": play.get("period", {}).get("number"),
                "clock": play.get("clock", {}).get("displayValue"),
                "yards_gained": play.get("statYardage"),
                "play_type": play_type,
                "label_run_pass": label,
                "score_diff": score_diff,
                "prev1_play_type": prev_types[-1],
                "prev2_play_type": prev_types[-2],
                "prev3_play_type": prev_types[-3],
                "prev1_yards": prev_yards[-1],
                "prev2_yards": prev_yards[-2],
                "prev3_yards": prev_yards[-3],
                "prev1_down": prev_downs[-1],
                "prev1_distance": prev_dist[-1],
            })
            team_queues[offense].append(plays[-1])

    return {
        "home_team": home_team,
        "away_team": away_team,
        "plays": plays
    }


def main():
    with open(GAME_ID_FILE) as f:
        game_ids = [line.strip() for line in f if line.strip().isdigit()]

    all_data = {"2024": {}}

    for i, gid in enumerate(sorted(game_ids)):
        print(f"[{i+1}/{len(game_ids)}] Scraping Game ID {gid}...")
        game_data = scrape_game(gid)
        if game_data and game_data["plays"]:
            all_data["2024"][gid] = game_data
            print(f"{gid}: {len(game_data['plays'])} Run/Pass plays.")
        else:
            print(f"{gid}: No plays found or invalid game.")

        #save every 25 games
        if (i + 1) % 25 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2)
            print(f"Checkpoint saved ({i+1} games processed).")

        time.sleep(0.5)  #delay

    #final save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved {len(all_data['2024'])} games to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
