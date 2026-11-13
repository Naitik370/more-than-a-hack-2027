"""Label-free diagnostics for comparing variants of the model.

The published 5 minutes of ground truth are for checking the pipeline, not for choosing between
versions of it (RULES.md, EVALUATION.md). So variants are compared here without them:

  counts      possessions and dead-ball spells over the whole match, against typical top-flight
              numbers (~250-300 possessions as defined here, ~100-110 restarts, ~45-50% dead time)
  tag agree   how often the decoded team agrees with Smart Tagging's team phases, with the tagging
              prior switched off in the decoder so the comparison is not circular. Smart Tagging is
              an independent automatic system with its own errors, so this is a proxy -- useful for
              ranking variants of the team evidence, not an accuracy.

    python solution/diagnostics.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run  # noqa: E402  (sets up the remaining paths)
import data_cache  # noqa: E402
import features as F  # noqa: E402
import load_smart_tagging  # noqa: E402

TARGET_POSSESSIONS = 270


def possession_count(labels):
    graded = labels != "X"
    seg = labels[graded]
    starts = np.flatnonzero(np.concatenate(([True], seg[1:] != seg[:-1])))
    return int(np.isin(seg[starts], ["H", "A"]).sum())


def tag_agreement(labels, sig, events):
    """Share of decoded H/A frames agreeing with the team of a tagged phase (middle half only).

    Only phase codes (Build Up, Progression, transitions, Defending in ...): zone codes such as
    Ball in the Box say where the ball is rather than who has it.
    """
    want = np.zeros(len(labels), dtype=np.int8)  # +1 home has it, -1 away, 0 unknown
    for event in events:
        code = (event["code"] or "").strip().lower()
        side = F.TEAM_TAGS.get((event["tags"].get("Team") or "").strip().lower())
        if side is None or code not in F.PHASE_CODES:
            continue
        sign = 1 if side == 0 else -1
        if code in F.DEFENDING_CODES:
            sign = -sign
        elif code not in F.ATTACKING_CODES:
            continue
        a, b = event["start_frame"], event["end_frame"]
        q = (b - a) // 4
        want[a + q:b - q] = sign
    alive = np.isin(labels, ["H", "A"])
    m = alive & (want != 0)
    return float(((labels[m] == "H") == (want[m] > 0)).mean())


def calibrate_gain(data, events, p, target=TARGET_POSSESSIONS, lo=5.0, hi=200.0, steps=10):
    """Control gain that gives `target` possessions (count is monotone in the gain)."""
    for _ in range(steps):
        mid = np.sqrt(lo * hi)
        labels, _ = run.predict(data, events, dict(p, control_gain=mid))
        if possession_count(labels) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def report(name, data, events, p, calibrate=True):
    p = dict(p, tag_gain_missing=0.0, tag_gain_present=0.0)
    if calibrate:
        p["control_gain"] = calibrate_gain(data, events, p)
    labels, sig = run.predict(data, events, p)
    print("{:<28} gain {:>6.1f}  tag agree {:.4f}  {}".format(
        name, p["control_gain"], tag_agreement(labels, sig, events), run.summary(labels)))


if __name__ == "__main__":
    data = data_cache.load()
    events = load_smart_tagging.load()
    report("proximity + ball speed", data, events, {})
    for radius in (1.0, 1.5, 2.0):
        for rel in (1.5, 2.0, 3.0):
            report("radius {} relative {}".format(radius, rel), data, events,
                   {"control_radius_m": radius, "control_relative_mps": rel})
