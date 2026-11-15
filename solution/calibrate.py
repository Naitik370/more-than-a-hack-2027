"""Calibrate the model on Metrica's public sample games, which have real labels.

    git clone https://github.com/metrica-sports/sample-data external/sample-data
    python solution/calibrate.py               # score the current parameters on the 3 games
    python solution/calibrate.py --search      # coordinate search, with leave-one-game-out check

The provided game is never touched here. Each sample game is run through exactly the pipeline
used for BAR-ATH, fed the same kinds of input it gets there:

  dead/live pseudo-labels  a window around each set piece start (dead), live play at least 15 s
                           from any set piece (live) -- the shape of the Smart Tagging pseudo-labels
  restart tags             team-tagged corners, goal kicks and free kicks only, as Smart Tagging
                           tags them (throw-ins and kick-offs carry no team there)
  tagging prior            none (the sample games have no Smart Tagging phases)

and scored with the official metric code (evaluation/possession_metrics.py) against the frame
labels sample_games.py derives from the human-annotated events.
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run  # noqa: E402  (sets up the remaining paths)
import model as M  # noqa: E402
import features as F  # noqa: E402
import pseudo  # noqa: E402
import sample_games as S  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as PM  # noqa: E402

TAGGED_SET_PIECES = ("CORNER KICK", "GOAL KICK", "FREE KICK")


def context(events, gt, blocks, fps=F.FPS):
    n = len(gt)
    graded = pseudo.graded_mask(n, blocks)
    dead = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    tagged = []
    for e in events:
        if e["type"] != "SET PIECE":
            continue
        s = e["start"]
        dead[max(0, s - 5 * fps):s + 3 * fps] = True
        near[max(0, s - 15 * fps):s + 15 * fps] = True
        if any(k in (e["subtype"] or "").upper() for k in TAGGED_SET_PIECES):
            tagged.append((s, e["team"]))
    live = np.isin(gt, ["H", "A"]) & ~near
    return {"blocks": blocks, "dead": dead & graded, "live": live & graded, "tagged": tagged}


def load_all():
    games = {}
    for g in S.GAMES:
        data, events, gt, blocks = S.load(g)
        games[g] = (data, gt, context(events, gt, blocks))
    return games


def score(gt, pred):
    gt_c, pred_c = PM.label_codes(gt), PM.label_codes(pred)
    cm3 = PM.scored_confusion(PM.confusion(gt_c, pred_c))
    runs = PM.valid_runs(gt_c)
    spells = PM.spell_scores(gt_c, PM.fill_abstentions(pred_c, runs), runs)
    return {"spell": spells["mean_f1"], "macro": PM.macro_f1(cm3),
            "n_pred": spells["n_pred"], "n_gt": spells["n_gt"]}


def evaluate(games, p, which=None):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for g, (data, gt, ctx) in games.items():
            if which is not None and g not in which:
                continue
            pred, _ = run.predict(data, [], p, ctx)
            out[g] = score(gt, pred)
    return out


def mean_spell(result):
    return float(np.mean([r["spell"] for r in result.values()]))


def show(name, result):
    cells = "  ".join("g{} spell {:.3f} macro {:.3f} poss {}/{}".format(
        g, r["spell"], r["macro"], r["n_pred"], r["n_gt"]) for g, r in result.items())
    print("{:<34} mean spell {:.4f} | {}".format(name, mean_spell(result), cells), flush=True)


# Search space: the constants that were set from priors rather than fitted.
GRID = {
    "control_gain": [55, 73, 110, 160, 230, 330, 480],
    "control_radius_m": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "control_relative_mps": [1.0, 1.5, 2.0, 3.0, 4.0],
    "turnover_every_s": [4, 6, 9, 12, 15, 20],
    "shape_gain": [0.0, 0.5, 1.0, 2.0, 4.0],
    "dead_weight": [1.0, 1.5, 2.0, 3.0, 4.0],
    "stoppage_every_s": [15, 20, 30, 45],
    "dead_spell_s": [10, 15, 20, 30],
    "oob_dead": [3, 6, 10, 20],
    "restart_reach_m": [1.5, 3.0, 5.0],
    "alive_lead_s": [0.0, 0.4, 0.7, 1.0],
}


def coordinate_search(games, start, which=None, rounds=2):
    best = dict(start)
    best_score = mean_spell(evaluate(games, best, which))
    for r in range(rounds):
        changed = False
        for key, values in GRID.items():
            for v in values:
                if v == best.get(key):
                    continue
                trial = dict(best, **{key: v})
                s = mean_spell(evaluate(games, trial, which))
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            print("  round {} {:<22} -> {}  ({:.4f})".format(r + 1, key, best.get(key), best_score),
                  flush=True)
        if not changed:
            break
    return best, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    games = load_all()
    current = dict(M.PARAMS)
    show("current parameters", evaluate(games, current))
    if not args.search:
        return
    keys = list(GRID)
    print("leave-one-game-out:")
    for held in S.GAMES:
        train = [g for g in S.GAMES if g != held]
        fitted, s = coordinate_search(games, current, train)
        diff = {k: fitted[k] for k in keys if fitted[k] != current[k]}
        show("held-out g{} fitted on {} {}".format(held, train, diff),
             evaluate(games, fitted, [held]))
        show("held-out g{} current".format(held), evaluate(games, current, [held]))
    fitted, s = coordinate_search(games, current)
    print("fitted on all:", {k: fitted[k] for k in keys if fitted[k] != current[k]})
    show("fitted on all", evaluate(games, fitted))


if __name__ == "__main__":
    main()
