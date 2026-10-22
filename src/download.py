"""Download StatsBomb Open Data for the 2015/16 Premier League, La Liga, Serie A and Ligue 1.

Source: StatsBomb Open Data, https://github.com/statsbomb/open-data
Licence: StatsBomb Public Data User Agreement (non-commercial use, StatsBomb must be credited),
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf

Full event files are 3-4 MB per match, so each one is reduced on the fly to the events this project
needs (shots, goalkeeper actions, goalkeeper passes, starting XIs and substitutions) and saved as one
parquet per league in data/raw/. Player nicknames come from the lineup files. Re-running skips leagues that are already downloaded.
(The 2015/16 Bundesliga is only partly released, so it is left out.)
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
COMPETITIONS = [(2, 27, "premier_league"), (11, 27, "la_liga"), (12, 27, "serie_a"), (7, 27, "ligue_1")]
SESSION = requests.Session()


def get(url):
    err = None
    for _ in range(6):
        try:
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccup: retry
            err = e
    raise err


def name(d):
    return (d or {}).get("name")


def reduce_match(match_id):
    rows = []
    for e in get(f"{BASE}/events/{match_id}.json"):
        t = e["type"]["name"]
        pos = name(e.get("position"))
        loc = e.get("location") or [None, None]
        base = {"match_id": match_id, "id": e["id"], "index": e["index"], "period": e["period"],
                "minute": e["minute"], "type": t, "team": name(e.get("team")),
                "player": name(e.get("player")), "player_id": (e.get("player") or {}).get("id"),
                "position": pos, "x": loc[0], "y": loc[1],
                "related": ",".join(e.get("related_events") or [])}
        if t == "Shot":
            s = e["shot"]
            end = (s.get("end_location") or []) + [None, None, None]
            gk = [f for f in s.get("freeze_frame") or []
                  if not f["teammate"] and name(f.get("position")) == "Goalkeeper"]
            rows.append({**base, "xg": s.get("statsbomb_xg"), "outcome": name(s.get("outcome")),
                         "subtype": name(s.get("type")), "body_part": name(s.get("body_part")),
                         "technique": name(s.get("technique")), "play_pattern": name(e.get("play_pattern")),
                         "first_time": bool(s.get("first_time", False)),
                         "end_x": end[0], "end_y": end[1], "end_z": end[2],
                         "ff_gk": name(gk[0]["player"]) if gk else None,
                         "ff_gk_x": gk[0]["location"][0] if gk else None,
                         "ff_gk_y": gk[0]["location"][1] if gk else None})
        elif t == "Goal Keeper":
            g = e["goalkeeper"]
            rows.append({**base, "subtype": name(g.get("type")), "outcome": name(g.get("outcome")),
                         "technique": name(g.get("technique")), "body_part": name(g.get("body_part"))})
        elif t == "Pass" and pos == "Goalkeeper":
            p = e["pass"]
            end = p.get("end_location") or [None, None]
            rows.append({**base, "outcome": name(p.get("outcome")), "subtype": name(p.get("type")),
                         "height": name(p.get("height")), "length": p.get("length"),
                         "body_part": name(p.get("body_part")), "end_x": end[0], "end_y": end[1]})
        elif t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                rows.append({**base, "player": p["player"]["name"], "player_id": p["player"]["id"],
                             "position": p["position"]["name"]})
        elif t == "Substitution":
            rows.append({**base, "subtype": name(e["substitution"].get("replacement"))})
    return rows


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, sid, label in COMPETITIONS:
        out = RAW / f"{label}_events.parquet"
        if out.exists():
            print("exists:", out.name)
            continue
        m = pd.json_normalize(get(f"{BASE}/matches/{cid}/{sid}.json"))
        m["league"] = label
        m.to_csv(RAW / f"{label}_matches.csv", index=False)
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(reduce_match, m["match_id"]))
        df = pd.DataFrame([r for rows in res for r in rows])
        df["league"] = label
        df.to_parquet(out, index=False)
        print(f"{label}: {len(m)} matches, {len(df):,} rows", flush=True)
    # player nicknames (e.g. "De Gea") from the lineup files, used for readable labels
    out = RAW / "player_names.csv"
    if not out.exists():
        ids = pd.concat([pd.read_csv(RAW / f"{l}_matches.csv")["match_id"] for *_, l in COMPETITIONS])
        with ThreadPoolExecutor(16) as ex:
            lineups = list(ex.map(lambda i: get(f"{BASE}/lineups/{i}.json"), ids))
        names = pd.DataFrame([(p["player_id"], p["player_name"], p.get("player_nickname"))
                              for lu in lineups for t in lu for p in t["lineup"]],
                             columns=["player_id", "player", "nickname"]).drop_duplicates("player_id")
        names.to_csv(out, index=False)
        print("player names:", len(names))


if __name__ == "__main__":
    sys.exit(main())
