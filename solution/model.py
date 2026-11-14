"""Two sticky Markov chains, decoded with Viterbi: first alive/dead, then home/away.

Why this shape
--------------
A possession is an interval, and the ranking metric scores intervals. A per-frame classifier
flickers; a majority vote fixes flicker but smears every boundary by half its window. A Markov
chain with rare transitions does both jobs: it holds a state through stretches with no evidence
(the ball is missing on ~41% of graded frames) and places each change where the evidence turns.

The decision is split in two because the questions use different evidence. Whether the ball is
in play is read from the whole pitch (are players moving, is the ball visible, is it outside the
lines). Who has it is read from the ball and the players next to it. Decoding them jointly in one
3-state chain lets the home/away evidence leak into the alive/dead decision -- a confident team
reading makes "alive" look better than "dead" for no football reason.

Units
-----
Transition costs are nats per transition, from expected rates. Evidence is in nats per *second*
and divided by the frame rate: neighbouring frames are not independent observations, so a second
of consistent evidence counts roughly as one observation.

Where the numbers come from
---------------------------
Nothing is fitted to the published 5 minutes of ground truth (RULES.md: they are for checking the
pipeline, not for tuning). Rates come from typical top-flight football. The alive/dead evidence is
estimated on pseudo-labels from the automatic Smart Tagging (pseudo.py). The remaining thresholds
are physical (reach of a player, speed of a pass).
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = {
    # --- rates of events, from typical top-flight matches
    "stoppage_every_s": 30.0,      # ~110-130 stoppages over ~55-60 min of live ball
    "dead_spell_s": 20.0,          # a restart typically takes 10-30 s
    "turnover_every_s": 15.0,      # ~200-250 changes of possession over the live ball
    # --- alive/dead
    "dead_weight": 2.0,            # gives ~100 dead spells: a typical LaLiga match has ~100-110 restarts
    "oob_margin_m": 1.0,           # tracking noise; the whole ball must be over the line
    "oob_dead": 10.0,              # nats/s for a ball clearly outside the pitch
    "activity_half_width_s": 1.0,
    "activity_edges": [0, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 99],
    "speed_edges": [0, 1, 2, 4, 8, 15, 99],
    # --- who has it (nats per second, + favours home)
    # Control = the ball is within reach of a player AND moving with him. Proximity alone counts
    # every pass and long ball that flies past a player.
    "control_radius_m": 1.5,       # within reach of a player
    "control_softness_m": 0.3,     # tracking error on a player's position
    "control_relative_mps": 2.0,   # ball speed relative to the player: a dribbling touch
    "control_fast_mps": 8.0,       # (fallback when control_relative_mps is 0)
    # The one rate-like constant of the team chain: calibrated, without labels, so the match has
    # ~270 possessions -- turnovers (~200) plus restarts by the same team (~50 of ~100) in a
    # typical top-flight match. See solution/diagnostics.py.
    "control_gain": 84.0,
    "tag_gain_missing": 1.0,       # Smart Tagging phase team, when the ball is not visible
    "tag_gain_present": 0.3,
    # --- restarts: the player who takes a restart has the ball, so its team starts the alive
    # spell. Costs one turnover to overrule, as if the taker's possession had been observed.
    "restart_reach_m": 3.0,        # the taker stands at the ball
    "restart_window_s": 1.0,       # look at the last second of the dead spell
    "use_restarts": True,
    "restart_use_tags": True,      # Set Pieces team tags, when present, name the taker
}


def chain_costs(p_switch):
    """log transition matrix for a 2-state chain with a per-frame switch probability per state."""
    p_switch = np.asarray(p_switch, dtype=float)
    return np.log(np.array([[1 - p_switch[0], p_switch[0]],
                            [p_switch[1], 1 - p_switch[1]]]))


def viterbi(log_emit, log_trans, log_start):
    n, k = log_emit.shape
    back = np.empty((n, k), dtype=np.int8)
    score = log_start + log_emit[0]
    cols = np.arange(k)
    for t in range(1, n):
        cand = score[:, None] + log_trans          # [from, to]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], cols] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(mask):
    """[(start, end)) of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def decode_alive(dead_evidence, p, fps):
    """True where the ball is dead. dead_evidence is nats/s, + favours dead."""
    emit = np.zeros((len(dead_evidence), 2))
    emit[:, 1] = dead_evidence / fps
    trans = chain_costs([1.0 / (p["stoppage_every_s"] * fps), 1.0 / (p["dead_spell_s"] * fps)])
    # A half starts dead: the teams line up for the kick-off.
    path = viterbi(emit, trans, np.log([1e-3, 1.0]))
    return path == 1


def decode_team(team_evidence, alive, p, fps, restart_side=None):
    """H/A over the alive frames; each alive spell is decoded on its own. + favours home.

    restart_side(start) -> H, A or None names the team taking the restart that opens a spell.
    """
    out = np.full(len(alive), D, dtype=np.int8)
    p_turn = 1.0 / (p["turnover_every_s"] * fps)
    trans = chain_costs([p_turn] * 2)
    for start, end in runs(alive):
        emit = np.zeros((end - start, 2))
        emit[:, 0] = team_evidence[start:end] / fps
        log_start = np.log([0.5, 0.5])
        side = restart_side(start) if restart_side else None
        if side is not None:
            log_start = np.log([1 - p_turn, p_turn] if side == H else [p_turn, 1 - p_turn])
        out[start:end] = viterbi(emit, trans, log_start)
    return out
