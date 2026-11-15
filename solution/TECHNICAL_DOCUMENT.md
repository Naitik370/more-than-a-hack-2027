# Ball possession from automatic tracking with two sticky Markov chains

More than a Hack 2027 — Phase 1 technical document.

## 1. Approach in one paragraph

We label every frame in two passes. A two-state chain first decides whether the ball is **alive or
dead**, reading the whole pitch: is the ball visible, is it outside the lines, are the players
moving. A second two-state chain then decides **home or away** inside each alive spell. It reads
who controls the ball, how the two teams are shaped, and who took the last restart. Both chains
are decoded with Viterbi. Transitions are expensive, so the decoder holds a state through
stretches with no evidence and moves only where the evidence turns. That is what an interval-level
metric like spellF1 rewards.

No label from the provided game is used. The evidence models are learned on the match itself, from
pseudo-labels produced by the automatic Smart Tagging and from the tracking's own confident
moments. The model's constants were then fitted on three other matches with human annotations:
Metrica's public sample games, which the rules allow. Every fitted value was checked
leave-one-game-out.

## 2. Why this rather than something else

- **The metric scores intervals.** The baselines show that per-frame decisions flicker (131
  possessions in five minutes for the raw baseline). A majority vote fixes the flicker by smearing
  every boundary. A Markov chain with costly transitions fixes the flicker and keeps each boundary
  where the evidence changes.
- **The ball is missing in long stretches, not short ones.** Gaps on graded frames number about
  150 and last a median of ~11 s, so interpolating the ball was not useful. In 47% of gap frames
  the players' mean speed is below 1 m/s, against 8% when the ball is visible. A missing ball is
  mostly a stoppage, and the alive/dead chain uses exactly that. Only 4.7% of live-play frames
  lack the ball.
- **Two chains, not one three-state chain.** We first tried one H/A/D chain. Confident team
  evidence then made "alive" look better than "dead" for no football reason: the H and A states
  could collect a team bonus that the D state never could. Separating the questions removed that
  coupling.
- **A small model with named constants, calibrated elsewhere.** The provided game is the test set.
  A model with a dozen interpretable constants can be fitted on three other matches and checked
  on a held-out one. A large learned model could not be validated that way with this little
  labelled data.

## 3. Assets: used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | **used.** Everything below is computed from it. Coordinates scaled to metres (105.3 × 68.0). |
| ATD player speed channel | **used** for player activity. It matches the speed computed from positions (median ratio 0.99, correlation 0.91), which is what we substitute on the sample games, where the channel is absent. |
| ATD ball speed channel | **dropped.** Its median over graded frames is ~1.3 m/s, far too slow for a football. We compute ball speed from positions instead (central difference over ±2 frames): median ~5 m/s, p95 ~17 m/s. |
| tactical Smart Tagging | **used three ways.** (a) Set pieces and attacking phases give pseudo-labels for the dead/live evidence. (b) Phase team tags give a weak prior on who has the ball. (c) Team-tagged set pieces name the team taking a restart. |
| tactical homography, tactical video | **not used** in the pipeline. The video is used only by an optional explainer reel (`annotate.py`). |
| broadcast video, ATD, homography | **not used.** They are on their own timeline with no mapping provided, and the tactical ATD already covers 100% of frames. |
| Metrica public sample games 1–3 | **used for calibration only** (§4, §6). |

## 4. Training data

**No label of any kind from the provided game is used.**

| data | source | who labelled it | how much | used for |
|---|---|---|---|---|
| Smart Tagging pseudo-labels, provided game | Metrica's automatic Smart Tagging | an algorithm | dead: 5 s before to 3 s after each of the 59 Set Pieces, ~11.5k frames; live: the middle half of each attacking phase, ≥15 s from a set piece, ~27.7k frames | the alive/dead evidence (§5.1) |
| confident control frames, provided game | our own control reading | nobody (self-training) | ~28% of live frames | the team-shape model (§5.2) |
| **Metrica sample games 1, 2 and 3** | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), public | **Metrica's human event annotators** | 3 full matches; 973 possessions and 240 dead spells derived from the events | fitting 8 constants (§6) |

**From events to frame labels on the sample games** (`sample_games.py`), following the rules in
CHALLENGE.md:
- **Possession** belongs to the team of the latest on-ball event: a pass, carry, shot, recovery or
  set piece.
- **A lost ball** stays with the losing team until the opponents recover it.
- **The ball is dead** from the stoppage until the set piece that restarts play. The stoppage is a
  ball out, a foul, an offside, a goal or, if none is annotated, the end of the last action.

The derived counts are realistic: 297–351 possessions, 74–88 dead spells and 27–35% dead time
per match.

**The published 5-minute ground truth** was scored only at checkpoints, to confirm the pipeline:
frames aligned, labels meaning what we thought, file well-formed. No parameter was chosen by it.

## 5. The model

### 5.1 Alive or dead

Evidence, in nats per second, positive meaning dead. Neighbouring frames are not independent
observations, so evidence counts per second, not per frame.

- **Visibility.** The ball is visible on 30% of pseudo-dead frames against 88% of pseudo-live
  ones. The log-likelihood ratio comes from those two rates.
- **Player activity.** The mean tracked-player speed, smoothed over ±1 s, is converted to a
  binned log-likelihood ratio estimated on the same pseudo-labels. Its median is ~1.0 m/s
  pseudo-dead and ~2.1 m/s pseudo-live.
- **Ball speed.** A binned log-likelihood ratio, used only when the ball is visible.
- **Out of bounds.** A ball more than 1 m beyond a touchline or goal line adds +20 nats/s. One
  metre absorbs tracking error, since the whole ball must cross the line.

The learned ratios are summed and scaled by `dead_weight` = 4.

**Transitions.**
- A stoppage arrives about every 30 s of live ball.
- A dead spell is expected to last ~10 s, which makes leaving a stoppage cheap once play resumes.
- Each half starts dead, while the teams line up for the kick-off.

### 5.2 Home or away

Inside each alive spell, evidence in nats per second, positive meaning home:

- **Control.** For each team, take the nearest player to the ball. Control is the product of two
  terms:
  - *within reach*: a sigmoid in distance centred on 1.25 m, with 0.3 m softness;
  - *moving with him*: `1 / (1 + (v_rel / 2 m/s)^2)`, where `v_rel` is the speed of the ball
    relative to that player.

  Evidence is `230 × (c_home·(1−c_away) − c_away·(1−c_home))`, so a ball that both teams are
  touching gives none.
- **Team shape.** A team in possession pushes up, spreads out and moves players into the
  opponents' half. Four features describe this, each home minus away and smoothed over ±0.5 s:
  - the teams' mean forward speed;
  - spread (std of x plus std of y);
  - the offset between the two teams' blocks;
  - the players in the opponents' half.

  The attacking direction comes from where the home goalkeeper stands in each half. A logistic
  regression maps the features to a log-likelihood ratio, after removing the training set's prior
  odds. It is trained on the frames where control is unambiguous and weighted 0.5. It matters most
  where nobody is clearly on the ball: passes, aerial duels, loose balls.

  Trained on one half and tested on the other, it agrees with the confident control readings
  80–82% of the time, against 64–68% for always guessing the majority team. The coefficients have
  the same sign in both halves.
- **Smart Tagging prior.** Each frame inside a tagged phase gets +1 for the team it names as
  attacking, or −1 if the phase names it defending; zone tags count half. The prior adds 0.3
  nats/s while the ball is visible and 1.0 nats/s while it is missing.
- **Restart taker.** The team that takes a restart has the ball, so it starts the alive spell, and
  overruling it costs one turnover. The taker's team comes from, in order:
  1. the Set Pieces team tag (corners, goal kicks, free kicks);
  2. a player standing within 0.5 m of a touchline, or beyond it, before play resumes. That
     player is taking a throw-in.
  3. the player at the ball in the last second before play resumes.

  The first two are needed because the tracking often loses the ball just before a corner or a
  throw-in, so the taker is not seen "at the ball".

**Transitions.** A turnover is expected every ~20 s of live ball. One change of possession costs
6.2 nats, and a change that is reversed costs 12.4.

## 6. The possession rules we implemented, and how the constants were chosen

- **Loose balls.** A ball in flight or rolling free gives no evidence, so possession stays with
  the last team that controlled it, as CHALLENGE.md asks.
- **Control rather than proximity.** A pass or long ball flying past a player moves at 10+ m/s
  relative to him and counts for almost nothing. A ball that stays with a player counts in full.
- **Challenges.** When players of both teams are within reach, the evidence cancels. The decoder
  keeps the current possession until one side comes away with the ball.
- **Deflections against controlled changes.** A touch after which the ball stays with the new
  player pays for a change of possession. A deflection, where the ball leaves fast, does not.
- **Restarts.** A dead spell ends where the evidence for live play starts, typically at the kick.
  The restarting team owns the ball from that frame (§5.2), so a corner swung into the box stays
  with the attacking team until a defender controls it. Dead-ball ownership is not modelled: dead
  is `D`.
- **Abstention.** We never write `X` on graded frames. Every frame has a decoded state, and
  forward-filling an abstention would only reproduce the decoder's previous answer.

### 6.1 Calibration on Metrica's public sample games

`solution/calibrate.py` runs each sample game through exactly the pipeline used for BAR-ATH, with
the same kinds of input it gets there:
- **dead/live pseudo-labels** around set pieces;
- **restart team tags** only for corners, goal kicks and free kicks;
- **no tagging prior.**

It scores each run with the official metric code against the event-derived labels, then
searches the constants one at a time (coordinate search) for the best mean spellF1.

**Leave-one-game-out.** Fit on two games, score the third:

| held-out game | before (prior constants) | after (fitted on the other two) |
|---|---|---|
| 1 | 0.616 | **0.654** |
| 2 | 0.619 | **0.650** |
| 3 | 0.590 | **0.617** |

The gain holds on every held-out game. We use the values the folds agreed on:

| constant | before | after |
|---|---|---|
| control gain | 73 | 230 |
| control radius | 1.5 m | 1.25 m |
| turnover every | 15 s | 20 s |
| team-shape weight | 1.0 | 0.5 |
| dead-ball evidence weight | 2 | 4 |
| dead spell length | 20 s | 10 s |
| out-of-bounds evidence | 10 nats/s | 20 nats/s |
| restart reach | 3 m | 1.5 m |

On the three games together, spellF1 goes from 0.609 to 0.651 and macroF1 from 0.838 to 0.848.
The score is flat around the chosen values. A dead-ball weight of 3–6, an out-of-bounds weight of
10–40 or a turnover interval of 12–45 s all land within 0.004 of the best.

The main error before calibration was too few possessions: 216 against 325 in game 1. The
calibrated model finds 267.

On BAR-ATH the calibrated model gives 342 possessions. That is in line with the 300–350
annotated possessions per sample game, and with the possession rate in the public five minutes.

**Boundary timing.** On the sample games, decoded turnovers sit a median of +0.08 s from the
annotated ones. Restarts and stoppages sit +0.6–0.7 s late. Shifting every alive/dead boundary
earlier did not raise spellF1 (`alive_lead_s`, kept at 0).

### 6.2 Label-free checks on the provided game

These were used before calibration, and still serve as sanity checks (`solution/diagnostics.py`):
- **Counts against typical top-flight numbers.** Dead time, number of stoppages and number of
  possessions.
- **Agreement with Smart Tagging phase teams**, with the tagging prior switched off so the check is
  not circular.
  - It chose control-with-relative-speed over proximity.
  - It showed that zone codes agree less (56–69%) than phase codes (86–89%), which is why zone
    codes are half-weighted.
  - It favoured adding team shape (0.850 → 0.885).
- **Restart team against the team-tagged Set Pieces:** 25 of 27 right in the final model, against
  17 of 26 before the restart prior. On the sample games, where every throw-in is annotated, the
  touchline rule took throw-in restarts from 86% to 92% right.

**Tried and dropped:**
- **An "other team restarts" rule** for balls going out of play. The tracking rarely sees the ball
  cross the line, so it covered 5 of 26 restarts and got 2 right.
- **Filling short ball gaps by interpolation.** No effect, because the gaps are long.
- **Smoothing the control evidence.** Noise-sized changes.
- **A constant boundary shift.** See §6.1.

**Final output** over the 158,857 graded frames:
- 342 possessions (mean 9.8 s, median 5.9 s);
- 123 dead spells, 47.5% dead time;
- home (Barcelona) has 62% of the live ball.

**Stability.** Scaling any one constant by 0.7 or 1.3 changes at most 1.1% of graded frames, and
the possession count stays between 299 and 381.

**Determinism.** Repeated runs, including after rebuilding the tracking cache, produce
byte-identical files.

## 7. Limitations and what we would do next

- **Calibration transfers across matches, not tracking systems.** The sample games come from an
  older Metrica tracking product. Their ball is present on 86–91% of live frames, against 95% here.
  The leave-one-game-out gain was consistent, but BAR-ATH's tracking may still reward slightly
  different constants.
- **The event-derived labels are an approximation of the official ones.** Possession changes are
  placed at annotated recoveries, and stoppages at annotated outs and fouls, each with the
  annotator's own timing.
- **Restarts and stoppages trail the events by ~0.6 s** on the sample games. A detector for the
  kick, a sudden jump in ball speed from rest, could snap restart boundaries.
- **The team-shape model is fitted on the match it labels.** That is self-training, with no
  labels, but its errors correlate with those of the control reading it learns from.
- **Team evidence relies on the nearest player of each team.** Identity switches between slots of
  different teams, and duplicated tracks, put the ball near the wrong side. Aerial balls look like
  ground balls: there is no ball height, and relative speed only partly helps.
- **The dead-ball pseudo-labels inherit Smart Tagging's errors**, including missed set pieces.
- **Not used:** the broadcast view, which could recover the ball where the tactical tracking loses
  it, once the two timelines are aligned.

## 8. Reproduction

```bash
pip install -r solution/requirements.txt         # numpy 2.5.4, pandas 3.1.0; Python 3.12
python solution/run.py --name <team_name> --out submission
python evaluation/validate_submission.py submission/<team_name>.csv
```

The command reads only `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` and
`BAR-ATH_tactical_smart_tagging.xml`. The calibrated constants are written in `solution/model.py`,
so the sample games are **not** needed to reproduce the file. The output is deterministic.

To reproduce the calibration:

```bash
git clone https://github.com/metrica-sports/sample-data external/sample-data
python solution/calibrate.py            # scores the current constants on the three games
python solution/calibrate.py --search   # the leave-one-game-out search (~1 hour)
```
