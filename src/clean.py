"""Tidy shots faced and goalkeeper distribution from the raw StatsBomb events.

Each shot is assigned to the goalkeeper who faced it, in this order of preference:
  1. the goalkeeper event that StatsBomb links to the shot (related_events),
  2. the opposing goalkeeper in the shot's freeze frame,
  3. the opposing goalkeeper on the pitch at that minute (starting XI plus goalkeeper substitutions).
Own goals are not shots, so they are not counted against keepers. Penalties are kept but flagged.

Outputs (data/processed/):
  shots_faced.csv     every on-target shot faced with keeper, xG, location, type and placement
  shot_counts.csv     per keeper: unblocked shots faced and on-target count (for context)
  gk_passes.csv.gz    every pass made by a goalkeeper (distribution), gzipped to keep the repo small
  gk_matches.csv   matches started/appeared per keeper
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUES = {"premier_league": "Premier League", "la_liga": "La Liga", "serie_a": "Serie A", "ligue_1": "Ligue 1"}
ON_TARGET = {"Goal", "Saved", "Saved to Post"}
GK_SHOT_TYPES = {"Shot Faced", "Shot Saved", "Goal Conceded", "Penalty Conceded", "Penalty Saved",
                 "Shot Saved to Post", "Shot Saved Off Target", "Penalty Saved to Post", "Saved to Post", "Save"}


def zone(x, y):
    """Shot zone on the StatsBomb 120x80 pitch (attacking towards x=120)."""
    if x >= 114 and 30 <= y <= 50:
        return "Six-yard box"
    if x >= 102 and 18 <= y <= 62:
        return "Central box" if 30 <= y <= 50 else "Wide box"
    return "Outside box"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pd.read_parquet(RAW / f"{l}_events.parquet") for l in LEAGUES], ignore_index=True)
    m = pd.concat([pd.read_csv(RAW / f"{l}_matches.csv") for l in LEAGUES], ignore_index=True)
    nick = pd.read_csv(RAW / "player_names.csv").set_index("player")["nickname"].dropna().to_dict()
    teams = m[["match_id", "match_date", "home_team.home_team_name", "away_team.away_team_name"]]
    teams.columns = ["match_id", "date", "home", "away"]

    # goalkeeper on the pitch, per team per match, as (from_minute, keeper)
    xi = ev[(ev.type == "Starting XI") & (ev.position == "Goalkeeper")]
    gk_timeline = {(r.match_id, r.team): [(0, r.player)] for r in xi.itertuples()}
    subs = ev[(ev.type == "Substitution") & (ev.position == "Goalkeeper")]
    for r in subs.itertuples():
        gk_timeline.setdefault((r.match_id, r.team), []).append((r.minute, r.subtype))

    def keeper_on_pitch(match_id, team, minute):
        tl = sorted(gk_timeline.get((match_id, team), []))
        cur = None
        for t, k in tl:
            if t <= minute:
                cur = k
        return cur

    gke = ev[(ev.type == "Goal Keeper") & ev.subtype.isin(GK_SHOT_TYPES)]
    linked = {}
    for r in gke.itertuples():
        for rid in str(r.related).split(","):
            if rid:
                linked[rid] = r.player

    shots = ev[ev.type == "Shot"].merge(teams, on="match_id")
    shots["opponent"] = np.where(shots.team == shots.home, shots.away, shots.home)
    shots["keeper"] = shots.id.map(linked)
    shots["keeper_source"] = np.where(shots.keeper.notna(), "linked_event", None)
    ff = shots.keeper.isna() & shots.ff_gk.notna()
    shots.loc[ff, "keeper"], shots.loc[ff, "keeper_source"] = shots.loc[ff, "ff_gk"], "freeze_frame"
    rest = shots.keeper.isna()
    shots.loc[rest, "keeper"] = [keeper_on_pitch(a, b, c) for a, b, c in
                                 shots.loc[rest, ["match_id", "opponent", "minute"]].itertuples(index=False)]
    shots.loc[rest, "keeper_source"] = "on_pitch"
    shots = shots[shots.outcome != "Blocked"].copy()   # blocked by an outfielder: keeper never tested
    shots["on_target"] = shots.outcome.isin(ON_TARGET)
    shots["goal"] = shots.outcome == "Goal"
    shots["is_penalty"] = shots.subtype == "Penalty"
    dx, dy = 120 - shots.x, 40 - shots.y
    shots["distance_m"] = np.hypot(dx, dy) * 0.9144
    shots["zone"] = [zone(a, b) for a, b in zip(shots.x, shots.y)]
    # where the shot crossed / was stopped in the goal mouth (posts at y=36/44, bar at z=2.67)
    shots["mouth_y"] = shots.end_y.clip(36, 44)
    shots["mouth_z"] = shots.end_z.clip(0, 2.67)
    shots["height"] = pd.cut(shots.end_z, [-1, 0.9, 1.8, 10], labels=["Low", "Middle", "High"])
    shots["side"] = pd.cut(shots.end_y, [-1, 38.67, 41.33, 100], labels=["Left third", "Centre", "Right third"])
    shots["league"] = shots.league.map(LEAGUES)
    shots["keeper_name"] = shots.keeper.map(lambda k: nick.get(k, k))
    shots["keeper_team"] = shots.opponent
    cols = ["match_id", "date", "league", "keeper", "keeper_name", "keeper_team", "team", "keeper_source", "minute",
            "x", "y", "distance_m", "zone", "xg", "outcome", "on_target", "goal", "subtype", "is_penalty",
            "body_part", "technique", "play_pattern", "first_time", "end_y", "end_z", "mouth_y", "mouth_z",
            "height", "side", "ff_gk_x", "ff_gk_y"]
    shots = shots[cols].rename(columns={"team": "shooting_team"}).sort_values(["date", "match_id", "minute"])
    shots.groupby(["keeper_name", "keeper_team"]).agg(unblocked_shots=("xg", "size"), on_target=("on_target", "sum"),
                  xg_all_unblocked=("xg", "sum")).round(3).to_csv(OUT / "shot_counts.csv")
    shots[shots.on_target].drop(columns=["on_target", "keeper"]).round(3).to_csv(OUT / "shots_faced.csv", index=False)

    p = ev[ev.type == "Pass"].copy()
    p["complete"] = p.outcome.isna()
    p["league"] = p.league.map(LEAGUES)
    p["keeper_name"] = p.player.map(lambda k: nick.get(k, k))
    p["length_m"] = p.length * 0.9144
    p["pass_type"] = p.subtype.fillna("Open play").replace({"Recovery": "Open play", "Interception": "Open play"})
    p[["match_id", "league", "team", "keeper_name", "end_x", "end_y", "length_m", "height", "pass_type",
       "complete"]].round(1).to_csv(OUT / "gk_passes.csv.gz", index=False, compression="gzip")

    apps = pd.concat([xi[["match_id", "team", "player"]], subs[["match_id", "team", "subtype"]].rename(columns={"subtype": "player"})])
    apps = apps.merge(m[["match_id", "league"]], on="match_id")
    gm = apps.groupby(["player", "team"]).agg(matches=("match_id", "nunique"), league=("league", "first")).reset_index()
    gm["league"] = gm.league.map(LEAGUES)
    gm["keeper_name"] = gm.player.map(lambda k: nick.get(k, k))
    gm.rename(columns={"player": "keeper"}).to_csv(OUT / "gk_matches.csv", index=False)
    print(f"{len(shots):,} unblocked shots, {shots.on_target.sum():,} on target, "
          f"keeper source: {shots.keeper_source.value_counts().to_dict()}, missing keeper: {shots.keeper.isna().sum()}")
    print(f"{len(p):,} goalkeeper passes, {gm.keeper_name.nunique()} keepers")


if __name__ == "__main__":
    main()
