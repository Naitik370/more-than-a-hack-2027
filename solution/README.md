# Possession by sticky Markov chains

Our entry for More than a Hack 2027, Phase 1. Everything we wrote is in this folder; the rest of
the repository is the organisers' starter kit (MIT), kept so the scorer and loaders sit next to it.
The method is described in [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md).

## Reproduce the submission

Python 3.12 (3.9+ should work), and the data bundle unpacked into `data/` as described in
[../DATA.md](../DATA.md) — only `data/tactical/BAR-ATH_tactical_atd.{xml,txt}` and
`data/tactical/BAR-ATH_tactical_smart_tagging.xml` are read.

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows;  source .venv/bin/activate elsewhere
pip install -r solution/requirements.txt

python solution/run.py --name <team_name> --out submission
python evaluation/validate_submission.py submission/<team_name>.csv
```

The first run parses the tracking into `solution/cache/` (gitignored, a few seconds); later runs
take about two seconds. The output is deterministic: the same data gives the same file, byte for
byte.

## Calibration (optional)

The constants in `model.py` were fitted on Metrica's public sample games, which have human event
annotations (allowed: they are other games). They are not needed to reproduce the submission. To
reproduce the calibration:

```bash
git clone https://github.com/metrica-sports/sample-data external/sample-data
python solution/calibrate.py            # current constants on the three games
python solution/calibrate.py --search   # leave-one-game-out search, ~1 hour
```

## Files

| file | what it does |
|---|---|
| `run.py` | entry point: data in, `frame,label` CSV out |
| `data_cache.py` | parses the ATD text file once into numpy arrays (metres, video-frame indexed) |
| `features.py` | ball speed, player activity, out-of-bounds, per-team contact, team shape, Smart Tagging prior |
| `pseudo.py` | self-supervised evidence: dead/live from Smart Tagging pseudo-labels, team shape from confident control frames |
| `model.py` | the two Viterbi chains and every parameter, with where it comes from |
| `diagnostics.py` | label-free comparison of variants: counts and agreement with Smart Tagging |
| `sample_games.py` | loads Metrica's public sample games and derives frame labels from their human-annotated events |
| `calibrate.py` | scores the model on the sample games with the official metric; leave-one-game-out search of the constants |
| `annotate.py` | optional explainer reel: model labels, reasons and evidence drawn over four auto-picked clips of the tactical video (needs `requirements-viz.txt`; output stays in the gitignored `solution/cache/`) |

## What is not in here

No match data, no derived frame-level data, and no labels. `solution/cache/`, `external/` and
`evaluation/results/*.csv` are gitignored.
