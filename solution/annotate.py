"""Explain the model on the match video: a short annotated reel, rendered locally.

    python solution/annotate.py                 # writes solution/cache/annotated.mp4

Four clips are picked automatically from the model's own output -- a corner restart, a turnover,
the ball going out of play, and a pass during which possession is held -- each introduced by a
title card. Over the video:

    top left      the label the model writes for this frame, and why (plain English)
    top right     a mini-map drawn from the tracking: players, ball, the player in control ringed
    bottom        +-10 s of labels and the evidence behind them, with a "now" line

The output is derived from the match video and lives in solution/cache/ (gitignored): it must not
be published or uploaded anywhere. Needs opencv-python (solution/requirements-viz.txt).
"""

import os
import sys
import warnings

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run  # noqa: E402  (sets up the remaining paths)
import data_cache  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402
import load_smart_tagging  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

FPS = F.FPS
W, H_ = 1280, 720
HOME_NAME, AWAY_NAME = "Barcelona", "Athletic"
# BGR
COL = {"H": (200, 90, 20), "A": (40, 40, 210), "D": (130, 130, 130), "X": (60, 60, 60)}
WHITE, BLACK, BALL = (255, 255, 255), (0, 0, 0), (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------------------------
# drawing primitives

def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.6):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def wrap(s, width):
    words, lines, line = s.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return lines + ([line] if line else [])


# ---------------------------------------------------------------------------------------------
# overlay pieces; each takes a plain dict `f` describing one frame, so they can be tested on
# synthetic values without touching match data

MAP_W = 380
MAP_H = int(MAP_W * F.PITCH_WIDTH / F.PITCH_LENGTH)
MAP_X, MAP_Y = W - MAP_W - 16, 16


def to_map(x, y):
    return (int(MAP_X + x / F.PITCH_LENGTH * MAP_W), int(MAP_Y + y / F.PITCH_WIDTH * MAP_H))


def draw_minimap(img, f):
    panel(img, MAP_X - 6, MAP_Y - 6, MAP_X + MAP_W + 6, MAP_Y + MAP_H + 30, 0.55)
    line = (200, 200, 200)
    L, Wd = F.PITCH_LENGTH, F.PITCH_WIDTH
    cv2.rectangle(img, to_map(0, 0), to_map(L, Wd), line, 1)
    cv2.line(img, to_map(L / 2, 0), to_map(L / 2, Wd), line, 1)
    cv2.circle(img, to_map(L / 2, Wd / 2), int(9.15 / L * MAP_W), line, 1)
    for x0, x1 in ((0, 16.5), (L - 16.5, L)):
        cv2.rectangle(img, to_map(x0, Wd / 2 - 20.16), to_map(x1, Wd / 2 + 20.16), line, 1)
    for (x, y), side in zip(f["players"], f["teams"]):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(img, to_map(x, y), 5, COL["H" if side == 0 else "A"], -1, cv2.LINE_AA)
    for key, side in (("near_home", "H"), ("near_away", "A")):
        pos, strength = f[key]
        if pos is not None and strength > 0.25:
            cv2.circle(img, to_map(*pos), 11, WHITE, 2, cv2.LINE_AA)
    if f["ball"] is not None:
        cv2.circle(img, to_map(*f["ball"]), 4, BALL, -1, cv2.LINE_AA)
        cv2.circle(img, to_map(*f["ball"]), 5, BLACK, 1, cv2.LINE_AA)
    arrow = "->" if f["home_right"] else "<-"
    text(img, "{} attacks {}   ball {}".format(HOME_NAME, arrow,
                                                "tracked" if f["ball"] is not None else "LOST"),
         (MAP_X, MAP_Y + MAP_H + 20), 0.5)


def draw_label(img, f):
    lab = f["label"]
    name = {"H": HOME_NAME + " ball", "A": AWAY_NAME + " ball", "D": "Ball dead",
            "X": "Not graded"}[lab]
    panel(img, 16, 16, 560, 66 + 26 * len(f["why"]) + 8, 0.55)
    cv2.rectangle(img, (24, 24), (60, 60), COL[lab], -1)
    text(img, name, (72, 54), 1.0, WHITE, 2)
    for i, line in enumerate(f["why"]):
        text(img, line, (26, 90 + 26 * i), 0.58)


STRIP_Y = H_ - 190


def draw_strip(img, window):
    """window: dict of arrays over +-10 s (labels, dead_ev, control_ev, shape_ev, total_ev)."""
    x0, x1 = 16, W - 16
    panel(img, x0, STRIP_Y, x1, H_ - 12, 0.6)
    n = len(window["labels"])
    xs = np.linspace(x0 + 120, x1 - 10, n).astype(int)
    # label band
    by = STRIP_Y + 14
    text(img, "label", (x0 + 8, by + 14), 0.5)
    for i in range(n - 1):
        cv2.rectangle(img, (xs[i], by), (xs[i + 1], by + 20), COL[window["labels"][i]], -1)

    def trace(y_mid, half, series, scale, legend):
        cv2.line(img, (xs[0], y_mid), (xs[-1], y_mid), (90, 90, 90), 1)
        for (name, values, color) in series:
            v = np.clip(np.nan_to_num(values) / scale, -1, 1)
            pts = np.stack([xs, (y_mid - v * half).astype(int)], axis=1)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        text(img, legend[0], (x0 + 8, y_mid - half + 12), 0.45)
        text(img, legend[1], (x0 + 8, y_mid + half), 0.45)

    trace(STRIP_Y + 72, 22, [("dead", window["dead_ev"], (180, 180, 180))], 15.0,
          ("dead ^", "live v"))
    trace(STRIP_Y + 140, 30, [("control", window["control_ev"], WHITE),
                              ("shape", window["shape_ev"], (80, 200, 80)),
                              ("total", window["total_ev"], BALL)], 40.0,
          ("BAR ^", "ATH v"))
    text(img, "control", (x1 - 250, STRIP_Y + 112), 0.45, WHITE)
    text(img, "shape x10", (x1 - 180, STRIP_Y + 112), 0.45, (80, 200, 80))
    text(img, "total", (x1 - 80, STRIP_Y + 112), 0.45, BALL)
    mid = xs[n // 2]
    cv2.line(img, (mid, STRIP_Y + 6), (mid, H_ - 18), WHITE, 1)
    text(img, "now", (mid - 14, H_ - 16), 0.4)
    text(img, "-10 s", (xs[0], H_ - 16), 0.4)
    text(img, "+10 s", (xs[-1] - 40, H_ - 16), 0.4)


def title_card(lines, sub=None):
    img = np.zeros((H_, W, 3), dtype=np.uint8)
    img[:] = (30, 24, 20)
    y = 260 - 22 * len(lines)
    for i, s in enumerate(lines):
        scale, thick = (1.3, 2) if i == 0 else (0.75, 1)
        size = cv2.getTextSize(s, FONT, scale, thick)[0]
        text(img, s, ((W - size[0]) // 2, y), scale, WHITE, thick)
        y += 70 if i == 0 else 40
    if sub:
        size = cv2.getTextSize(sub, FONT, 0.55, 1)[0]
        text(img, sub, ((W - size[0]) // 2, H_ - 60), 0.55, (170, 170, 170))
    return img


# ---------------------------------------------------------------------------------------------
# turning model internals into per-frame descriptions

class Explainer:
    def __init__(self, data, events):
        self.data = data
        self.labels, self.sig = run.predict(data, events)
        self.p = dict(M.PARAMS)
        sig, p = self.sig, self.p
        self.control_ev = p["control_gain"] * sig["control"]
        self.shape_ev = p["shape_gain"] * sig["shape_llr"]
        self.blocks = half_video_blocks(GAME)
        self.home_right = np.zeros(len(self.labels), dtype=bool)
        self.restart = {}  # alive-spell start frame -> (side, source)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _, first, last in self.blocks:
                sl = slice(first, last + 1)
                self.home_right[sl] = F.home_attacks_right(data["players"][sl], data["team"])
                sides = run.restart_sides(sig, events, p, first, last)
                no_tag = run.restart_sides(sig, events, dict(p, restart_use_tags=False),
                                           first, last)
                for a, _ in M.runs(sig["alive"][sl]):
                    side = sides(a)
                    if side is not None:
                        # The tag is consulted first; if dropping it changes nothing, the taker
                        # was also seen at the ball.
                        source = ("player at the ball" if no_tag(a) == side
                                  else "set-piece tag")
                        self.restart[first + a] = (side, source)
        self.rows = np.arange(len(self.labels))
        self._nearest()

    def _nearest(self):
        players, ball, team = self.data["players"], self.data["ball"], self.data["team"]
        dist = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        dist = np.where(np.isfinite(dist), dist, np.inf)
        self.nearest = {}
        for side in (0, 1):
            slots = np.flatnonzero(team == side)
            self.nearest[side] = slots[np.argmin(dist[:, slots], axis=1)]

    def why(self, t):
        sig, lab = self.sig, self.labels[t]
        name = {"H": HOME_NAME, "A": AWAY_NAME}
        if lab == "D":
            if sig["oob"][t]:
                return ["Ball is outside the lines:", "strong evidence the ball is dead."]
            if not sig["present"][t]:
                return ["Ball not tracked and players slow", "(avg {:.1f} m/s): a stoppage.".format(
                    sig["activity"][t])]
            return ["Players walking (avg {:.1f} m/s):".format(sig["activity"][t]),
                    "play has stopped."]
        if lab == "X":
            return []
        lines = []
        start = t
        while start > 0 and sig["alive"][start - 1]:
            start -= 1
        if t - start < 3 * FPS and start in self.restart:
            side, source = self.restart[start]
            lines += ["Restart taken by {}".format(name["HA"[side]]),
                      "(from the {}).".format(source)]
        c = sig["control"][t]
        if c > 0.25 or c < -0.25:
            side = 0 if c > 0 else 1
            d = sig["d_home"][t] if side == 0 else sig["d_away"][t]
            rel = sig["rel_home"][t] if side == 0 else sig["rel_away"][t]
            lines += ["{} player {:.1f} m from the ball,".format(name["HA"[side]], d),
                      "ball moving with him ({:.1f} m/s rel.)".format(np.nan_to_num(rel, nan=0))]
            if lab != "HA"[side]:
                lines += ["...not yet enough to change possession."]
        elif not sig["present"][t]:
            lines += ["Ball not tracked: possession stays", "with {}.".format(name[lab])]
        else:
            lines += ["Nobody controls the ball (pass or", "loose ball): stays with {}.".format(
                name[lab])]
        s = self.shape_ev[t]
        if abs(s) > 1.0:
            lines += ["Team shape points to {}.".format(name["H" if s > 0 else "A"])]
        return lines[:5]

    def frame_info(self, t):
        d = self.data
        ball = d["ball"][t, :2]
        ball = tuple(ball) if np.isfinite(ball).all() else None

        def near(side):
            slot = self.nearest[side][t]
            pos = d["players"][t, slot, :2]
            if ball is None or not np.isfinite(pos).all():
                return (None, 0.0)
            c = self.sig["control"][t]
            return (tuple(pos), c if side == 0 else -c)
        return {"label": self.labels[t], "why": self.why(t),
                "players": d["players"][t, :, :2], "teams": d["team"], "ball": ball,
                "near_home": near(0), "near_away": near(1),
                "home_right": bool(self.home_right[t])}

    def window(self, t, half=10 * FPS):
        idx = np.clip(np.arange(t - half, t + half + 1, 2), 0, len(self.labels) - 1)
        dead = self.sig["dead_evidence"][idx]
        return {"labels": self.labels[idx], "dead_ev": dead,
                "control_ev": self.control_ev[idx], "shape_ev": self.shape_ev[idx] * 10,
                "total_ev": self.sig["team_evidence"][idx]}

    # --- picking clips from the model's own output ------------------------------------------

    def spells(self):
        lab = self.labels
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(lab)]))
        return [(lab[a], a, b) for a, b in zip(bounds[:-1], bounds[1:])]

    def pick(self, events):
        sig, lab, sp = self.sig, self.labels, self.spells()
        clips = []
        # 1. a corner, restart team named by the tag
        for e in events:
            if (e["code"] or "").strip().lower() == "set pieces" and \
                    e["tags"].get("Type") == "Corner Kick":
                k = np.flatnonzero(sig["alive"][e["start_frame"]:e["start_frame"] + 12 * FPS])
                if len(k):
                    s = e["start_frame"] + k[0]
                    clips.append(("corner", s - 5 * FPS, s + 10 * FPS))
                    break
        # 2. a turnover in open play, clear control on both sides
        for (l0, a0, b0), (l1, a1, b1) in zip(sp, sp[1:]):
            if {l0, l1} == {"H", "A"} and b0 - a0 > 6 * FPS and b1 - a1 > 6 * FPS:
                c = sig["control"]
                before = c[a1 - FPS:a1] * (1 if l0 == "H" else -1)
                after = c[a1:a1 + FPS] * (1 if l1 == "H" else -1)
                if before.max() > 0.5 and after.max() > 0.5:
                    clips.append(("turnover", a1 - 7 * FPS, a1 + 7 * FPS))
                    break
        # 3. ball out of play
        for (l0, a0, b0), (l1, a1, b1) in zip(sp, sp[1:]):
            if l0 in "HA" and l1 == "D" and b0 - a0 > 6 * FPS and sig["oob"][a1 - FPS:a1 + 2 * FPS].any():
                clips.append(("out of play", a1 - 8 * FPS, a1 + 5 * FPS))
                break
        # 4. a pass: nobody in control, possession held
        for l0, a0, b0 in sp:
            if l0 not in "HA" or b0 - a0 < 12 * FPS:
                continue
            loose = (np.abs(sig["control"][a0:b0]) < 0.1) & sig["present"][a0:b0] & \
                (np.nan_to_num(sig["ball_speed"][a0:b0]) > 8)
            for a, b in M.runs(loose):
                if b - a > 1.2 * FPS and a > 3 * FPS and b0 - a0 - b > 3 * FPS:
                    clips.append(("pass", a0 + a - 4 * FPS, a0 + b + 4 * FPS))
                    break
            if len(clips) == 4:
                break
        return clips


CARDS = {
    "intro": (["How the model labels possession",
               "Step 1: is the ball alive or dead?  (ball visible, outside the lines, players moving)",
               "Step 2: inside live play, who has it?  (control, team shape, restart taker)",
               "A label changes only when the evidence clearly turns."],
              "Top left: label + why.  Top right: tracking map.  Bottom: evidence over +-10 s."),
    "corner": (["1. A corner",
                "Before the kick the ball is dead: it is out of play and players are walking.",
                "The team taking the restart starts with the ball -- the tracking often loses the",
                "ball at corners, so the Smart Tagging set-piece tag names the taker."], None),
    "turnover": (["2. A turnover",
                  "The white trace is control: a player within reach AND the ball moving with him.",
                  "Possession switches only once the other team clearly has it,",
                  "not at the first touch -- a change costs evidence."], None),
    "out of play": (["3. The ball goes out",
                     "Ball outside the lines (or lost) and players slowing down:",
                     "the grey trace rises and the label turns to dead."], None),
    "pass": (["4. A pass",
              "In flight nobody controls the ball -- the white trace drops to zero.",
              "With no evidence either way, possession stays with the passing team",
              "until someone receives it."], None),
}


def main():
    out = os.path.join(HERE, "cache", "annotated.mp4")
    data = data_cache.load()
    events = load_smart_tagging.load()
    ex = Explainer(data, events)
    clips = ex.pick(events)

    video = os.path.join(os.path.dirname(HERE), "data", "tactical", GAME + "_tactical.mp4")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open " + video)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*codec), FPS, (W, H_))
        if writer.isOpened():
            break
    frames = 0

    def card(name, seconds):
        nonlocal frames
        lines, sub = CARDS[name]
        img = title_card(lines, sub)
        for _ in range(int(seconds * FPS)):
            writer.write(img)
            frames += 1

    card("intro", 7)
    for name, a, b in clips:
        card(name, 5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for t in range(a, b):
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.resize(img, (W, H_), interpolation=cv2.INTER_AREA)
            info = ex.frame_info(t)
            draw_label(img, info)
            draw_minimap(img, info)
            draw_strip(img, ex.window(t))
            text(img, "frame {}  {}".format(t, name), (W - 260, STRIP_Y - 10), 0.45,
                 (200, 200, 200))
            writer.write(img)
            frames += 1
    writer.release()
    print("wrote", out, "-", frames, "frames,", round(frames / FPS, 1), "s;",
          "clips:", ", ".join("{} @ {}".format(n, a) for n, a, _ in clips))


if __name__ == "__main__":
    main()
