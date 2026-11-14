# Ball possession from automatic tracking with two sticky Markov chains

More than a Hack 2027 — Phase 1 technical document.

## 1. Approach in one paragraph

We label every frame in two passes. A two-state chain first decides whether the ball is **alive or
dead**, reading the whole pitch: is the ball visible, is it outside the lines, are the players
moving. A second two-state chain then decides **home or away** inside each alive spell, reading
the ball and the players touching it. Both are decoded with Viterbi. Transitions are rare and
expensive, so the decoder holds a state through stretches with no evidence and moves only where
the evidence turns, which is what an interval-level metric like spellF1 rewards. No model is
trained on human labels: the dead-ball evidence is learned from pseudo-labels produced by the
automatic Smart Tagging, and the few remaining constants come from physics and from typical
top-flight match statistics.

## 2. Why this rather than something else

- **The metric scores intervals.** The baselines show that per-frame decisions flicker (131
  possessions in five minutes for the raw baseline) and that a majority vote fixes flicker by
  smearing every boundary. A Markov chain with rare transitions fixes flicker and keeps the
  boundaries where the evidence changes. On the match, our decoded boundaries sit a median of 0–2
  frames from a physical event: a kick at restarts, the ball crossing a line or disappearing at
  stoppages.
- **The ball is missing in long stretches, not short ones.** Gaps on graded frames number about
  150 and last a median of ~11 s, so interpolating the ball was not useful. In 47% of gap frames
  the players' mean speed is below 1 m/s, against 8% when the ball is visible: a missing ball is
  mostly a stoppage, and the alive/dead chain uses exactly that.
- **Two chains, not one three-state chain.** We first tried one H/A/D chain. Confident team
  evidence then made "alive" look better than "dead" for no football reason: the H and A states
  could collect a team bonus that the D state never could. Separating the questions removed that
  coupling.
- **No supervised learning on this game.** The rules exclude human labels on the provided game,
  and the published five minutes are for checking the pipeline, not for tuning. With no labels to
  learn from, a small model whose every parameter has a stated reason is easier to defend than a
  learned one we could not validate.

## 3. Assets: used and dropped

| asset | use |
|---|---|
| tactical ATD (players + ball) | **used.** Everything below is computed from it. Coordinates scaled to metres (105.3 × 68.0). |
| ATD ball speed channel | **dropped.** Its median over graded frames is ~1.3 m/s, far too slow for a football. We compute ball speed from positions instead (central difference over ±2 frames): median ~5 m/s, p95 ~17 m/s. |
| tactical Smart Tagging | **used twice.** (a) Set pieces and attacking phases give pseudo-labels for dead/live evidence. (b) Phase team tags give a weak prior on who has the ball. |
| tactical homography | **not used.** Nothing in the method needs pixels. |
| tactical video | **not used** in the pipeline. |
| broadcast video, ATD, homography | **not used.** It is on its own timeline with no mapping provided, and the tactical ATD already covers 100% of frames. |

## 4. Training data

**No human-labelled data of any kind is used, from this game or any other.**

- **Pseudo-labels, produced by an algorithm.** These are frame sets taken from Metrica's automatic
  Smart Tagging. The *dead* set is a window from 5 s before to 3 s after the start of each of the
  59 Set Pieces instances, ~11.5k graded frames. The *live* set is the middle half of each Build
  Up, Progression, Creating Chances and Attacking Transition phase, at least 15 s from any set
  piece, ~27.7k frames. No person chose or corrected any of them.
- **The published 5-minute ground truth** was scored at four checkpoints to confirm the pipeline:
  frames aligned, labels meaning what we thought, file well-formed. No parameter was chosen by its
  score. Every choice was made with the label-free diagnostics in §6.

## 5. The model

### 5.1 Alive or dead

Evidence, in nats per second, positive meaning dead:

- **Visibility.** The ball is visible on 30% of pseudo-dead frames against 88% of pseudo-live
  ones. The log-likelihood ratio comes from those two rates.
- **Player activity.** The mean tracked-player speed, smoothed over ±1 s, is converted to a
  binned log-likelihood ratio estimated on the same pseudo-labels. Its median is ~1.0 m/s
  pseudo-dead and ~2.1 m/s pseudo-live.
- **Ball speed.** A binned log-likelihood ratio, used only when the ball is visible.
- **Out of bounds.** A ball more than 1 m beyond a touchline or goal line adds +10 nats/s. One
  metre absorbs tracking error, since the whole ball must cross the line.

The ratios are summed and scaled by `dead_weight` = 2. Neighbouring frames are not independent
observations, so we count evidence per second rather than per frame.

**Transitions** come from typical match rates. A stoppage arrives about every 30 s of live ball
(~110–130 per match). A dead spell lasts about 20 s. Each half starts dead, while the teams line
up for the kick-off.

### 5.2 Home or away

Inside each alive spell, evidence in nats per second, positive meaning home:

- **Control.** For each team, take the nearest player to the ball. Control is the product of two
  terms. The first is *within reach*: a sigmoid in distance centred on 1.5 m with 0.3 m softness.
  The second is *moving with him*: `1 / (1 + (v_rel / 2 m/s)^2)`, where `v_rel` is the speed of
  the ball relative to that player. Evidence is `gain × (c_home·(1−c_away) − c_away·(1−c_home))`,
  so a ball both teams are touching gives none.
- **Smart Tagging prior.** Each frame inside a tagged phase gets +1 for the team it names as
  attacking, or −1 if the phase names it defending; zone tags count half. The prior adds 0.3
  nats/s while the ball is visible and 1.0 nats/s while it is missing.

- **Restart taker.** The team that takes a restart has the ball, so it starts the alive spell,
  and overruling it costs one turnover. The taker's team comes from the Set Pieces team tag when
  Smart Tagging gives one (corners, goal kicks, free kicks). Otherwise it is the team of the player
  at the ball in the last second before play resumes. Both are needed: before most corners the
  tracking has lost the ball, so the taker cannot be seen.

**Transitions.** A turnover arrives every ~15 s of live ball, which costs 5.9 nats per change.

## 6. The possession rules we implemented

- **Loose balls.** A ball in flight or rolling free gives no evidence, so possession stays with
  the last team that controlled it, as CHALLENGE.md asks.
- **Control rather than proximity.** A pass or long ball flying past a player moves at 10+ m/s
  relative to him and counts for almost nothing. A ball that stays with a player counts in full.
  Measured over the match, the label-free proxy below prefers this to the baselines' "nearest
  player" rule.
- **Challenges.** When players of both teams are within reach, the evidence cancels and the
  decoder keeps the current possession until one side comes away with the ball.
- **Deflections against controlled changes.** One change costs 5.9 nats and a change that is
  reversed costs 11.9. A touch where the ball stays with the new player for a fraction of a second
  pays for a change. A deflection, where the ball leaves fast, does not.
- **Restarts.** A dead spell ends where the evidence for live play starts, typically at the kick.
  The restarting team owns the ball from that frame (§5.2), so a corner swung into the box stays
  with the attacking team until a defender controls it. Dead-ball ownership is not modelled: dead
  is `D`.
- **Abstention.** We never write `X` on graded frames. Every frame has a decoded state, and
  forward-filling an abstention would only reproduce the decoder's previous answer.

### Label-free diagnostics used to make choices

`solution/diagnostics.py` compares variants without the ground truth, using two checks:

1. **Counts against typical top-flight numbers.** The output should have ~45–50% dead time,
   ~100–110 dead spells and ~250–300 possessions, where a possession here is turnovers (~200)
   plus restarts by the team that already had the ball. Two constants were set this way:
   `dead_weight` (→ 101 dead spells) and the control `gain` (→ ~270 possessions with the prior
   off).
2. **Agreement with Smart Tagging phase teams**, with the tagging prior switched off in the
   decoder so the check is not circular. Smart Tagging has its own errors, so we treat this as a
   ranking proxy, not an accuracy. It chose control-with-relative-speed over proximity (0.84
   against 0.83), and it showed that zone codes agree less (56–69%) than phase codes (86–89%).
   That is why zone codes are half-weighted in the prior.
3. **Restart team against the team-tagged Set Pieces** (26 with a visible restart). With no
   restart prior, the first decoded team was right on 17. The nearest-player rule alone gets 18,
   and with the tags 21. Tag agreement rose from 0.840 to 0.843.

Final output over the 158,857 graded frames: 277 possessions (mean 12.1 s, median 7.3 s), 101 dead
spells, 47.1% dead time, and home has 63% of the live ball.

**Stability.** Scaling any one parameter by 0.7 or 1.3 changes at most 1.7% of graded frames. The
possession count stays between 231 and 286. The rate constants (turnover, stoppage and dead-spell
frequencies, and the tagging prior) move fewer than 0.6% of frames. The control constants (radius,
relative speed and gain) matter most, which is why the gain is the one constant calibrated against
match statistics.

## 7. Limitations and what we would do next

- **Restarts with no tag and no visible ball.** Throw-ins are not team-tagged, and the ball is often
  missing just before one is taken. A rule from where the ball left the pitch and which team's
  players crowd that area would cover them.
- **Possession count is calibrated to a prior, not to the truth.** If the ground truth's notion of
  "controlled" is stricter or looser than ours, every possession boundary moves together. With
  labelled matches from other games (Metrica's open sample games, for example), we would fit the
  control gain and radius there, degrading the tracking to look like this automatic tracking.
- **Team evidence relies on the nearest player of each team.** Identity switches between slots of
  different teams, and duplicated tracks, put the ball near the wrong side. Aerial balls look like
  ground balls: there is no ball height, and relative speed only partly helps. A small ball-height
  or event detector on the tactical video would address both.
- **Quick restarts and short stoppages** (under ~5 s) can be missed. A stoppage has to pay ~12
  nats of transitions in and out.
- **The dead-ball pseudo-labels inherit Smart Tagging's errors**, including missed set pieces:
  only 26 throw-ins are tagged, where a match has ~40.
- **Not used:** the broadcast view, which could recover the ball where the tactical tracking loses
  it once the two timelines are aligned.

## 8. Reproduction

```bash
pip install -r solution/requirements.txt         # numpy 2.5.4, pandas 3.1.0; Python 3.12
python solution/run.py --name <team_name> --out submission
python evaluation/validate_submission.py submission/<team_name>.csv
```

The command reads only `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` and
`BAR-ATH_tactical_smart_tagging.xml`, and it is deterministic. `python solution/diagnostics.py`
reprints the label-free comparisons in §6.
