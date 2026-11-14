"""Data in, submission out.

    python solution/run.py                       # writes evaluation/results/<name>.csv
    python solution/run.py --name myteam --out .

Deterministic: same data, same file.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import data_cache  # noqa: E402
import features as F  # noqa: E402
import model as M  # noqa: E402
import pseudo  # noqa: E402
import load_smart_tagging  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402


def build_signals(data, events, p):
    players, ball, team = data["players"], data["ball"], data["team"]
    n = len(ball)
    present = np.isfinite(ball[:, 0])
    (d_home, rel_home), (d_away, rel_away) = F.team_contacts(players, ball, team)
    tag_vote, set_piece = F.tagging_prior(events, n)
    return {
        "present": present,
        "ball_speed": F.ball_speed(ball),
        "oob": present & F.out_of_bounds(ball, p["oob_margin_m"]),
        "activity": F.activity(players, int(round(p["activity_half_width_s"] * F.FPS))),
        "d_home": d_home,
        "d_away": d_away,
        "rel_home": rel_home,
        "rel_away": rel_away,
        "tag_vote": tag_vote,
        "set_piece": set_piece,
    }


def dead_evidence(sig, events, p, blocks):
    """nats/s, + favours a dead ball."""
    n = len(sig["present"])
    graded = pseudo.graded_mask(n, blocks)
    dead, live = pseudo.dead_live_masks(events, n, F.FPS, graded)
    llr = pseudo.presence_llr(sig["present"], dead, live)
    llr += pseudo.binned_llr(sig["activity"], p["activity_edges"], dead, live)[0]
    llr += pseudo.binned_llr(sig["ball_speed"], p["speed_edges"], dead, live,
                             valid=sig["present"])[0]
    evidence = p["dead_weight"] * llr
    evidence[sig["oob"]] += p["oob_dead"]
    return evidence


def team_evidence(sig, p):
    """nats/s, + favours home."""
    speed = np.nan_to_num(sig["ball_speed"], nan=0.0)
    r, soft = p["control_radius_m"], p["control_softness_m"]
    with np.errstate(over="ignore"):
        c_home = 1.0 / (1.0 + np.exp((sig["d_home"] - r) / soft))
        c_away = 1.0 / (1.0 + np.exp((sig["d_away"] - r) / soft))
    if p["control_relative_mps"]:
        # Control = close AND moving with the player. Unknown relative speed counts as moving.
        def with_player(rel):
            rel = np.nan_to_num(rel, nan=99.0)
            return 1.0 / (1.0 + (rel / p["control_relative_mps"]) ** 2)
        c_home = c_home * with_player(sig["rel_home"])
        c_away = c_away * with_player(sig["rel_away"])
        slow = 1.0
    else:
        slow = 1.0 / (1.0 + (speed / p["control_fast_mps"]) ** 4)
    evidence = p["control_gain"] * slow * (c_home * (1 - c_away) - c_away * (1 - c_home))
    tag = np.clip(sig["tag_vote"], -1, 1)
    evidence += np.where(sig["present"], p["tag_gain_present"], p["tag_gain_missing"]) * tag
    return evidence


def restart_sides(sig, events, p, first, last):
    """Function: alive-spell start (block-relative) -> team taking that restart, or None.

    Set Pieces tagged with a team (corners, goal kicks, free kicks) name it directly. Otherwise
    the taker is the player at the ball during the last second before play resumes.
    """
    tagged = []
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        side = F.TEAM_TAGS.get((event["tags"].get("Team") or "").strip().lower())
        if side is not None:
            tagged.append((event["start_frame"], side))
    window = int(round(p["restart_window_s"] * F.FPS))
    d_home, d_away = sig["d_home"], sig["d_away"]

    def side_at(start):
        s = first + start
        for frame, side in (tagged if p["restart_use_tags"] else []):  # a tagged set piece started during this dead spell
            if s - 60 * F.FPS <= frame <= s and not sig["alive"][frame:s].any():
                return side
        a, b = max(first, s - window), s + 1
        dh, da = d_home[a:b], d_away[a:b]
        near = np.minimum(dh, da) < p["restart_reach_m"]
        if not near.any():
            return None
        home = int((dh[near] < da[near]).sum())
        away = int(near.sum()) - home
        return M.H if home > away else M.A if away > home else None
    return side_at


def predict(data, events, p=None):
    p = dict(M.PARAMS, **(p or {}))
    blocks = half_video_blocks(GAME)
    sig = build_signals(data, events, p)
    dead_ev = dead_evidence(sig, events, p, blocks)
    team_ev = team_evidence(sig, p)

    labels = np.full(len(sig["present"]), "X", dtype="<U1")
    sig["alive"] = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        sl = slice(first, last + 1)
        dead = M.decode_alive(dead_ev[sl], p, F.FPS)
        sig["alive"][sl] = ~dead
        sides = restart_sides(sig, events, p, first, last) if p["use_restarts"] else None
        labels[sl] = M.LABELS[M.decode_team(team_ev[sl], ~dead, p, F.FPS, sides)]
    sig["dead_evidence"] = dead_ev
    sig["team_evidence"] = team_ev
    return labels, sig


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def summary(labels, fps=F.FPS):
    """Possession count and durations over the graded frames -- unlabelled diagnostics only."""
    graded = labels != "X"
    counts = {k: int((labels[graded] == k).sum()) for k in "HAD"}
    spells = []
    for start, end in M.runs(graded):
        seg = labels[start:end]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        spells += [(seg[a], b - a) for a, b in zip(bounds[:-1], bounds[1:])]
    poss = [n for k, n in spells if k in "HA"]
    dead = [n for k, n in spells if k == "D"]
    return ("H {H} A {A} D {D} | possessions {np} mean {mp:.1f}s median {md:.1f}s | "
            "dead spells {nd} mean {mdd:.1f}s | dead share {ds:.1%}").format(
        np=len(poss), mp=np.mean(poss) / fps, md=np.median(poss) / fps, nd=len(dead),
        mdd=np.mean(dead) / fps if dead else 0, ds=counts["D"] / graded.sum(), **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="hmm")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    data = data_cache.load()
    assert len(data["ball"]) == source_frames("tactical", GAME)
    events = load_smart_tagging.load()
    labels, _ = predict(data, events)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name + ".csv")
    write_submission(path, labels)
    print("wrote", os.path.abspath(path))
    print(summary(labels))


if __name__ == "__main__":
    main()
