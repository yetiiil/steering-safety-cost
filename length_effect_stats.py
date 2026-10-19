#!/usr/bin/env python3
"""
Summarize the length effect of a steering vector from open-ended generations.

run_open_ended.py --type open_ended writes, per (layer, multiplier):
    <results_root>/<behavior>/results_single_<behavior>_open_ended_<layer>_<mult>_<size>_<family>.json   (custom / CAST vector)
    <results_root>/<behavior>/results_<behavior>_open_ended_<layer>_<mult>_<size>_<family>.json          (default / original CAA vector)
each a JSON list of {"question": ..., "model_output": ...}.

This script reports mean/median output length (words and chars, plus tokens if a
tokenizer is given) per multiplier, separately for the original and sanitized
vectors, so you can eyeball whether sanitization preserved the length effect.

Example:
    python length_effect_stats.py --behavior length_conciseness
    python length_effect_stats.py --behavior length_conciseness --tokenizer qwen/qwen2.5-7b-instruct
"""
import argparse
import glob
import json
import os
import re
import statistics
from pathlib import Path

from cast.paths import REPO_ROOT
# steering_settings.py is the single source of truth for the behaviour surface.
from steering_settings import PAPER_BEHAVIORS

# results_[single_|sanitized_dim_N_]<behavior>_open_ended_<layer>_<mult>_<size>_<family>.json
# Anchor on "_open_ended_" so an underscore-containing behavior name (e.g.
# length_conciseness) does not confuse the field split.
# The "single" group flags the sanitized vector: run_open_ended.py emits
# "results_single_..." for any --custom_vector, with an optional --run_tag inserted
# right after the prefix ("results_single_dim_2_..."), which the ".*" below absorbs.
# The "sanitized_dim_N_" alternative is kept so result trees written by the earlier
# per-dimension scripts still parse.
_FNAME_RE = re.compile(
    r"^results_(?P<single>(?:single_|sanitized(?:_dim_\d+)?_))?.*_open_ended_"
    r"(?P<layer>\d+)_(?P<mult>-?\d+(?:\.\d+)?)_(?P<size>[^_]+)_(?P<family>[^_]+)\.json$"
)


def _load_outputs(path):
    with open(path) as f:
        data = json.load(f)
    return [d.get("model_output", "") or "" for d in data]


def _summary(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behavior", required=True, choices=PAPER_BEHAVIORS,
                    help="Behavior name; length_conciseness is the one this summary is for")
    ap.add_argument("--results_dir", default="results",
                    help="Base results dir, matching run_open_ended.py --results_root "
                         "(default: results). A relative path is taken relative to the "
                         "repository root.")
    ap.add_argument("--baseline_dir", default=None,
                    help="Optional separate base dir holding the ORIGINAL (non-sanitized) "
                         "open-ended baseline, e.g. 'results' when sanitized results live in "
                         "'results_dim_N'. Only its original (non-sanitized) files are read.")
    ap.add_argument("--tokenizer", default=None, help="Optional HF tokenizer name/path to also count tokens")
    # Filters: without these the (is_sanitized, mult) key collapses across
    # different layers/sizes/families and silently overwrites rows. Set them to
    # restrict to one config; otherwise the same key from different configs
    # will be folded into the same row.
    ap.add_argument("--layer", type=int, default=None,
                    help="Restrict to result files for this steering layer (recommended).")
    ap.add_argument("--model_size", type=str, default=None,
                    help="Restrict to result files for this model size, e.g. 7b.")
    ap.add_argument("--model_family", type=str, default=None,
                    help="Restrict to result files for this model family, e.g. qwen.")
    args = ap.parse_args()

    behavior_dir = os.path.join(str(REPO_ROOT / args.results_dir), args.behavior)
    if not os.path.isdir(behavior_dir):
        raise SystemExit(f"No results directory: {behavior_dir}")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    files = sorted(glob.glob(os.path.join(behavior_dir, "results_*open_ended_*.json")))
    # Pull the dim-independent original baseline from a separate dir if requested.
    # Skip any sanitized files there so they don't clobber results_dir's sanitized rows.
    if args.baseline_dir:
        baseline_behavior_dir = os.path.join(str(REPO_ROOT / args.baseline_dir), args.behavior)
        for p in sorted(glob.glob(os.path.join(baseline_behavior_dir, "results_*open_ended_*.json"))):
            m = _FNAME_RE.match(os.path.basename(p))
            if m and m.group("single"):
                continue
            files.append(p)
    if not files:
        raise SystemExit(f"No open-ended result files under {behavior_dir}")

    # rows[(is_sanitized, multiplier)] = stats dicts
    rows = {}
    seen_configs = set()  # to detect silent collisions
    matched_paths = []
    for path in files:
        m = _FNAME_RE.match(os.path.basename(path))
        if not m:
            print(f"  (skipping unrecognized file: {os.path.basename(path)})")
            continue
        layer = int(m.group("layer"))
        size = m.group("size")
        family = m.group("family")
        if args.layer is not None and layer != args.layer:
            continue
        if args.model_size is not None and size != args.model_size:
            continue
        if args.model_family is not None and family != args.model_family:
            continue

        is_sanitized = m.group("single") is not None
        mult = float(m.group("mult"))
        outs = _load_outputs(path)
        words = [len(o.split()) for o in outs]
        chars = [len(o) for o in outs]
        entry = {"words": _summary(words), "chars": _summary(chars), "n": len(outs)}
        if tok is not None:
            ntoks = [len(tok.encode(o, add_special_tokens=False)) for o in outs]
            entry["tokens"] = _summary(ntoks)

        key = (is_sanitized, mult)
        cfg = (layer, size, family)
        if key in rows and seen_configs and cfg not in {c for k, c in seen_configs if k == key}:
            print(f"  WARNING: row {key} already populated by a different config; "
                  f"overwriting with {os.path.basename(path)} ({cfg})")
        rows[key] = entry
        seen_configs.add((key, cfg))
        matched_paths.append(path)

    if not rows:
        filters = []
        if args.layer is not None: filters.append(f"layer={args.layer}")
        if args.model_size is not None: filters.append(f"size={args.model_size}")
        if args.model_family is not None: filters.append(f"family={args.model_family}")
        raise SystemExit(
            f"No files in {behavior_dir} matched the filter ({', '.join(filters) or 'none'}). "
            f"Files present:\n  " + "\n  ".join(os.path.basename(p) for p in files)
        )

    metrics = ["words", "chars"] + (["tokens"] if tok is not None else [])
    header = f"{'vector':<12}{'mult':>7}{'n':>5}  " + "  ".join(f"{me+'(mean/med)':>20}" for me in metrics)
    filter_str = ", ".join(filter(None, [
        f"layer={args.layer}" if args.layer is not None else None,
        f"model_size={args.model_size}" if args.model_size is not None else None,
        f"model_family={args.model_family}" if args.model_family is not None else None,
    ])) or "ALL CONFIGS (warning: may mix layers/sizes/families)"
    print(f"\nLength effect for behavior: {args.behavior}  [{filter_str}]")
    print(f"Summarizing {len(matched_paths)} files\n")
    print(header)
    print("-" * len(header))
    for (is_sanitized, mult) in sorted(rows, key=lambda k: (k[0], k[1])):
        e = rows[(is_sanitized, mult)]
        label = "sanitized" if is_sanitized else "original"
        cells = []
        for me in metrics:
            s = e[me]
            cells.append(f"{s['mean']:>9.1f}/{s['median']:<10.0f}" if s else f"{'-':>20}")
        print(f"{label:<12}{mult:>7}{e['n']:>5}  " + "  ".join(cells))
    print(
        "\nRead: a working length vector should make output length move monotonically "
        "with the multiplier (shorter for conciseness, longer for verbose). "
        "If 'sanitized' rows track the 'original' rows closely, the effect survived sanitization."
    )


if __name__ == "__main__":
    main()
