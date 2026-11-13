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

## Files

| file | what it does |
|---|---|
| `run.py` | entry point: data in, `frame,label` CSV out |
| `data_cache.py` | parses the ATD text file once into numpy arrays (metres, video-frame indexed) |
| `features.py` | ball speed, player activity, out-of-bounds, per-team contact, Smart Tagging prior |
| `pseudo.py` | self-supervised dead/live evidence from Smart Tagging pseudo-labels |
| `model.py` | the two Viterbi chains and every parameter, with where it comes from |
| `diagnostics.py` | label-free comparison of variants: counts and agreement with Smart Tagging |

## What is not in here

No match data, no derived frame-level data, and no labels. `solution/cache/` and
`evaluation/results/*.csv` are gitignored.
