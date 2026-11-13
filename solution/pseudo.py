"""Self-supervised evidence: pseudo-labels from Smart Tagging, histograms from the tracking.

No human label on the provided game is used anywhere. The pseudo-labels come from Metrica's
automatic Smart Tagging (an algorithm), and the published 5-minute ground truth is never read.

  dead  a window around the start of every SET PIECES instance (a restart: throw-in, corner,
        goal kick, free kick, kick-off). The ball is stationary or out of play there.
  live  the middle half of every attacking phase (Build Up, Progression, Creating Chances,
        Attacking Transition), away from any set piece. The ball is in play there by definition.

From those two frame sets we estimate, per feature, a binned log-likelihood ratio
log p(feature | dead) / p(feature | live). The decoder sums them, scaled to "nats per second",
because neighbouring frames are nowhere near independent observations.
"""

import numpy as np

LIVE_CODES = {"build up", "progression", "creating chances", "attacking transition"}


def graded_mask(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        mask[first:last + 1] = True
    return mask


def dead_live_masks(events, n, fps, mask, before_s=5.0, after_s=3.0):
    dead = np.zeros(n, dtype=bool)
    near_set_piece = np.zeros(n, dtype=bool)
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        start = event["start_frame"]
        dead[max(0, start - int(before_s * fps)):start + int(after_s * fps)] = True
        near_set_piece[max(0, start - 15 * fps):start + 15 * fps] = True

    live = np.zeros(n, dtype=bool)
    for event in events:
        if (event["code"] or "").strip().lower() not in LIVE_CODES:
            continue
        a, b = event["start_frame"], event["end_frame"]
        quarter = (b - a) // 4
        live[a + quarter:b - quarter] = True
    live &= ~near_set_piece
    return dead & mask, live & mask


def binned_llr(values, edges, positive, negative, valid=None, smoothing=5.0, clip=3.0):
    """Per-frame log p(bin|positive)/p(bin|negative). 0 where `valid` is False or value is NaN."""
    finite = np.isfinite(values)
    if valid is not None:
        finite &= valid
    bins = np.clip(np.digitize(np.where(finite, values, 0), edges) - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    pos = np.bincount(bins[positive & finite], minlength=k) + smoothing
    neg = np.bincount(bins[negative & finite], minlength=k) + smoothing
    table = np.clip(np.log(pos / pos.sum()) - np.log(neg / neg.sum()), -clip, clip)
    return np.where(finite, table[bins], 0.0), table


def presence_llr(present, positive, negative, smoothing=5.0):
    p_pos = (present[positive].sum() + smoothing) / (positive.sum() + 2 * smoothing)
    p_neg = (present[negative].sum() + smoothing) / (negative.sum() + 2 * smoothing)
    return np.where(present, np.log(p_pos / p_neg), np.log((1 - p_pos) / (1 - p_neg)))
