"""Per-frame signals derived from the tracking and the Smart Tagging.

Everything is computed for every video frame and is NaN-aware: the ball is missing on ~43% of
frames, in long stretches, so no feature assumes it exists.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAMES, GAME  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]


def moving_mean(values, half_width):
    """Centred NaN-ignoring mean over +/- half_width frames, via cumulative sums."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    total = np.concatenate(([0.0], np.cumsum(filled)))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def ball_speed(ball):
    """m/s from positions, central difference over +/- 2 frames.

    The ATD's own ball speed channel is not usable: its median over graded frames is ~1.3 m/s,
    far too slow for a football, while the position differences give a realistic distribution.
    """
    xy = ball[:, :2].astype(np.float64)
    speed = np.full(len(xy), np.nan)
    step = 2
    delta = xy[2 * step:] - xy[:-2 * step]
    speed[step:-step] = np.hypot(delta[:, 0], delta[:, 1]) * FPS / (2 * step)
    return speed


def velocity(xy, step=2):
    """(..., 2) positions -> m/s velocity vectors by central difference over +/- step frames."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    out[step:-step] = (xy[2 * step:] - xy[:-2 * step]) * (FPS / (2.0 * step))
    return out


def team_contacts(players, ball, team):
    """For each team: distance from the ball to its nearest player, and the speed of the ball
    relative to that player. inf / NaN where there is no ball or no player.

    Relative speed is what separates control from proximity: a ball under control moves with the
    player, a pass or a long ball in flight passes him at 10+ m/s.
    """
    dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    v_players = velocity(players[:, :, :2])
    v_ball = velocity(ball[:, :2])
    rows = np.arange(len(ball))
    out = []
    for side in (0, 1):
        slots = np.flatnonzero(team == side)
        nearest = slots[np.argmin(dist[:, slots], axis=1)]
        rel = v_ball - v_players[rows, nearest]
        out.append((dist[rows, nearest], np.hypot(rel[:, 0], rel[:, 1])))
    return out


def out_of_bounds(ball, margin):
    """Ball beyond a touchline or goal line by more than `margin` metres."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -margin) | (x > PITCH_LENGTH + margin) | (y < -margin) | (y > PITCH_WIDTH + margin)


def activity(players, half_width):
    """Mean speed of the tracked players, smoothed. Low while play is stopped."""
    with np.errstate(invalid="ignore"):
        per_frame = np.nanmean(players[:, :, 2], axis=1)
    return moving_mean(per_frame, half_width)


# Smart Tagging -----------------------------------------------------------------------------------

# Phase codes that say the tagged team is in possession, and those that say it is not.
ATTACKING_CODES = {
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box", "goals",
}
DEFENDING_CODES = {
    "defending in defensive third", "defending in middle third", "defending in attacking third",
    "defensive transition",
}
# Phases name a team's activity; zone tags name where the ball is (a cross into the box can be
# cleared, a long ball can be lost), so they say less about who controls it and count half.
ZONE_CODES = {"ball in final third", "ball in the box", "players in the box", "long ball",
              "creating chances"}
PHASE_CODES = (ATTACKING_CODES | DEFENDING_CODES) - ZONE_CODES - {"goals"}
TEAM_TAGS = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def tagging_prior(events, n_frames):
    """Per frame, net count of phases saying home (+) or away (-) has the ball; and set pieces.

    Returns (team_vote, set_piece) where team_vote is a float array (positive favours home) and
    set_piece is a boolean array marking Set Pieces intervals.
    """
    vote = np.zeros(n_frames)
    set_piece = np.zeros(n_frames, dtype=bool)
    for event in events:
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if end <= start:
            continue
        code = (event["code"] or "").strip().lower()
        if code == "set pieces":
            set_piece[start:end] = True
            continue
        side = TEAM_TAGS.get((event["tags"].get("Team") or "").strip().lower())
        if side is None:
            continue
        sign = 1.0 if side == 0 else -1.0
        if code in ZONE_CODES:
            sign *= 0.5
        if code in ATTACKING_CODES:
            vote[start:end] += sign
        elif code in DEFENDING_CODES:
            vote[start:end] -= sign
    return vote, set_piece
