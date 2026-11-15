"""Metrica's public sample games, as calibration matches with real (human) labels.

The competition rules allow hand labels on *other* games; the provided game gets none. These three
matches are published by Metrica with tracking and human-annotated events:

    git clone https://github.com/metrica-sports/sample-data external/sample-data

Each is loaded into the same containers the pipeline uses for BAR-ATH (players / ball / team in
metres, one row per frame) and given a frame-level H/A/D/X ground truth built from its events with
the challenge's possession rules:

  possession  belongs to the team of the latest on-ball event (pass, carry, shot, recovery, set
              piece). A lost ball stays with the losing team until the other one recovers it --
              CHALLENGE.md: possession changes when the opponents gain *controlled* possession.
  dead        from the stoppage (ball out, foul, offside, goal -- or, if none is annotated, the end
              of the last action) until the set piece that restarts play.
  X           before each half's kick-off and after its last event.

Nothing here touches the provided game.
"""

import io
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(os.path.dirname(HERE), "external", "sample-data", "data")
GAMES = (1, 2, 3)

OWNER_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}
PLAY_TYPES = OWNER_TYPES | {"BALL LOST", "CHALLENGE"}


def _to_metres(arr):
    arr[..., 0] *= F.PITCH_LENGTH
    arr[..., 1] *= F.PITCH_WIDTH
    return arr


def _with_speed(players_xy):
    """Append a speed channel computed from positions. On BAR-ATH the ATD's own speed channel
    matches this (median ratio 0.99), so the two are interchangeable."""
    v = F.velocity(players_xy)
    speed = np.hypot(v[..., 0], v[..., 1])
    return np.concatenate([players_xy, speed[..., None]], axis=-1).astype(np.float32)


# --- games 1 and 2: Metrica CSV ----------------------------------------------------------------

def _load_csv_game(g):
    base = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(g), "Sample_Game_{}_".format(g))
    tables = []
    for side in ("Home", "Away"):
        t = pd.read_csv(base + "RawTrackingData_{}_Team.csv".format(side), skiprows=2)
        tables.append(t)
    home, away = tables
    n = int(max(home["Frame"].max(), away["Frame"].max()))
    rows_h = home["Frame"].to_numpy() - 1
    rows_a = away["Frame"].to_numpy() - 1

    def xy_columns(t):
        cols = [c for c in t.columns if c.startswith("Player")]
        return [(c, t.columns[t.columns.get_loc(c) + 1]) for c in cols]

    slots = []
    for t, rows, side in ((home, rows_h, 0), (away, rows_a, 1)):
        for cx, cy in xy_columns(t):
            arr = np.full((n, 2), np.nan, dtype=np.float32)
            arr[rows, 0] = t[cx].to_numpy(dtype=np.float32)
            arr[rows, 1] = t[cy].to_numpy(dtype=np.float32)
            slots.append((arr, side))
    players = _to_metres(np.stack([s[0] for s in slots], axis=1))
    team = np.array([s[1] for s in slots], dtype=np.int8)
    bx = home.columns.get_loc("Ball")
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    ball[rows_h, 0] = home.iloc[:, bx].to_numpy(dtype=np.float32)
    ball[rows_h, 1] = home.iloc[:, bx + 1].to_numpy(dtype=np.float32)
    _to_metres(ball)

    ev = pd.read_csv(base + "RawEventsData.csv")
    events = [{"type": r["Type"], "subtype": "" if pd.isna(r["Subtype"]) else str(r["Subtype"]),
               "team": 0 if r["Team"] == "Home" else 1, "period": int(r["Period"]),
               "start": int(r["Start Frame"]) - 1, "end": int(r["End Frame"]) - 1}
              for _, r in ev.iterrows()]
    return {"players": _with_speed(players), "ball": ball, "team": team}, events


# --- game 3: FIFA EPTS, the same format as BAR-ATH, plus JSON events ---------------------------

def _load_epts_game():
    base = os.path.join(SAMPLE_DIR, "Sample_Game_3", "Sample_Game_3_")
    root = ET.parse(base + "metadata.xml").getroot()
    score = root.find(".//Score")
    home_id = score.get("idLocalTeam")
    player_team = {p.get("id"): (0 if p.get("teamId") == home_id else 1) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    all_players = list(player_team)
    index = {pid: i for i, pid in enumerate(all_players)}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = channel_player[ref.get("playerChannelId")]
            if not order or order[-1] != pid:
                order.append(pid)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    n = max(s[1] for s in specs)

    with open(base + "tracking.txt") as handle:
        lines = handle.read().splitlines()
    players = np.full((n, len(all_players), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    frame_of = np.array([int(line.split(":", 1)[0]) for line in lines])
    for first, last, order in specs:
        chunk = [line for line, f in zip(lines, frame_of) if first <= f <= last]
        text = "\n".join(chunk).replace(":", ",").replace(";", ",")
        table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                            na_values=["NaN"]).to_numpy()
        rows = table[:, 0].astype(np.int64) - 1
        xy = table[:, 1:1 + 2 * len(order)].reshape(-1, len(order), 2)
        cols = [index[pid] for pid in order]
        players[rows[:, None], np.array(cols)[None, :]] = xy
        ball[rows, :2] = table[:, 1 + 2 * len(order):1 + 2 * len(order) + 2]
    _to_metres(players)
    _to_metres(ball)
    team = np.array([player_team[pid] for pid in all_players], dtype=np.int8)

    with open(base + "events.json") as handle:
        raw = json.load(handle)["data"]
    events = []
    for e in raw:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub if s)
        elif isinstance(sub, dict):
            sub = sub.get("name") or ""
        else:
            sub = ""
        events.append({"type": e["type"]["name"], "subtype": sub,
                       "team": 0 if (e.get("team") or {}).get("id") == home_id else 1,
                       "period": int(e["period"]),
                       "start": int(e["start"]["frame"]) - 1,
                       "end": int((e.get("end") or {}).get("frame") or e["start"]["frame"]) - 1})
    return {"players": _with_speed(players), "ball": ball, "team": team}, events


# --- ground truth from events ------------------------------------------------------------------

def _is_trigger(e):
    """An event that stops play."""
    sub = (e["subtype"] or "").upper()
    if e["type"] in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if e["type"] == "CHALLENGE" and "FAULT" in sub:
        return True
    if e["type"] == "SHOT" and "GOAL" in sub and "WOODWORK-GOAL" != sub:
        return True
    return e["type"] == "BALL LOST" and ("OFFSIDE" in sub or "HAND BALL" in sub)


def ground_truth(events, n):
    """(labels, blocks): frame-level H/A/D/X from the annotated events."""
    events = sorted(events, key=lambda e: (e["start"], e["end"]))
    labels = np.full(n, "X", dtype="<U1")
    blocks = []
    for period in sorted({e["period"] for e in events}):
        evs = [e for e in events if e["period"] == period]
        kick = [e for e in evs if e["type"] == "SET PIECE"]
        if not kick:
            continue
        first = kick[0]["start"]
        last = max(max(e["end"], e["start"]) for e in evs)
        last = min(last, n - 1)
        blocks.append((period, first, last))
        owner = np.full(n, -1, dtype=np.int8)
        current = kick[0]["team"]
        cursor = first
        for e in evs:
            if e["start"] < first or e["type"] not in OWNER_TYPES:
                continue
            owner[cursor:e["start"]] = current
            current, cursor = e["team"], e["start"]
        owner[cursor:last + 1] = current
        seg = np.where(owner[first:last + 1] == 0, "H", "A")
        labels[first:last + 1] = seg
        # dead spells: stoppage -> restarting set piece
        for i, sp in enumerate(kick[1:], start=1):
            before = [e for e in evs if kick[i - 1]["start"] <= e["start"] < sp["start"]
                      and e is not kick[i - 1]]
            plays = [e for e in before if e["type"] in PLAY_TYPES]
            last_owner_start = max([e["start"] for e in before if e["type"] in OWNER_TYPES],
                                   default=kick[i - 1]["start"])
            triggers = [max(e["start"], e["end"]) if e["type"] == "SHOT" else e["start"]
                        for e in before if _is_trigger(e) and e["start"] >= last_owner_start]
            if triggers:
                dead_start = min(triggers)
            elif plays:
                dead_start = max(max(e["end"], e["start"]) for e in plays)
            else:
                dead_start = kick[i - 1]["start"]
            dead_start = min(max(dead_start, first), sp["start"])
            labels[dead_start:sp["start"]] = "D"
    return labels, blocks


def load(g):
    """data, events, labels, blocks for sample game g."""
    data, events = _load_csv_game(g) if g in (1, 2) else _load_epts_game()
    labels, blocks = ground_truth(events, len(data["ball"]))
    return data, events, labels, blocks
