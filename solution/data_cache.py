"""Read the tactical ATD once and keep it as numpy arrays.

The raw text file is ~190 MB and takes a while to parse; every later step reads the .npz instead.
Nothing here interprets the data, it only changes its container.

    players  float32 (n_frames, n_slots, 3)   x, y in metres, speed in m/s; NaN where missing
    ball     float32 (n_frames, 3)            same
    team     int8    (n_slots,)               0 home, 1 away

Row i is video frame i (ATD frame_id i + 1). The cache lives in solution/cache/, which is
gitignored: it is derived match data and must not be published.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")


def team_of_each_track(meta_path):
    """0 if the slot belongs to the home team (first team in the file), 1 if away."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_atd(raw_path, n_frames, n_slots, pitch):
    """Fields are ':' / ';' / ',' separated; flatten to one CSV and let pandas' C parser do it."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                        na_values=["NaN"], engine="c").to_numpy()
    del text
    if table.shape[1] != 1 + 3 * n_slots + 3:
        raise ValueError("unexpected column count {}".format(table.shape[1]))

    rows = table[:, 0].astype(np.int64) - 1  # frame_id counts from 1, video frames from 0
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players[rows] = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball[rows] = table[:, 1 + 3 * n_slots:]

    length, width = pitch
    for arr in (players, ball):
        arr[..., 0] *= length
        arr[..., 1] *= width
    return players, ball


def load(game=GAME, rebuild=False):
    path = os.path.join(CACHE_DIR, "{}_tactical.npz".format(game))
    if os.path.exists(path) and not rebuild:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)
    players, ball = parse_atd(raw_path, source_frames("tactical", game), len(team),
                              GAMES[game]["pitch"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


if __name__ == "__main__":
    data = load(rebuild="--rebuild" in sys.argv)
    print("players {}  ball {}  slots home/away {}/{}".format(
        data["players"].shape, data["ball"].shape,
        int((data["team"] == 0).sum()), int((data["team"] == 1).sum())))
    print("ball present on {:.1f}% of frames".format(
        100 * np.isfinite(data["ball"][:, 0]).mean()))
