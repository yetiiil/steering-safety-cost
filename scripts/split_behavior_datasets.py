#!/usr/bin/env python3
"""Split a behaviour's A/B dataset into its generation and test halves.

Vendored from the research tree's own split script.  The seed (42), the ratio
(9:1) and the order of operations are the ones that produced the released
splits, so what this writes is what the paper's vectors were extracted from and
what the A/B tables were scored on.

    python scripts/split_behavior_datasets.py \\
        --behaviors corrigible-more-HHH power-seeking-inclination

Reads  data/raw/behaviors/<behavior>.json  (written by
scripts/00_download_data.py --only behaviors) and writes

    data/behaviors/<behavior>/generate_dataset.json   90%
    data/behaviors/<behavior>/test_dataset_ab.json    10%

The original script kept the two halves in separate trees; this repository puts
every file of a behaviour in one directory, which is what cast.paths.behavior_dir
resolves to.

Seeding
-------
The original seeds the generator once, at import, and then shuffles a fixed list
of behaviour files in order.  Each file's split therefore depends on how much
randomness the files ahead of it consumed, not on the seed alone.  This release
ships two of those behaviours and none of the ones that came before them, so the
generator is advanced here by shuffling a placeholder list of the same length
instead: random.shuffle() consumes an amount of randomness that depends only on
the list's length, so the state it leaves behind -- and hence the split -- is
identical without the other files being present.
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / 'data' / 'raw' / 'behaviors'
BEHAVIOR_DIR = REPO_ROOT / 'data' / 'behaviors'

TRAIN_RATIO = 0.9
SEED = 42

# Record counts of the behaviour files shuffled ahead of each of these two, in
# the order the original script processed them.  Only the lengths matter; see
# "Seeding" above.
PRECEDING_LENGTHS = {
    'corrigible-more-HHH': (1000, 468),
    'power-seeking-inclination': (1000, 468, 1000, 1000, 1000, 1000, 1000,
                                  1000, 1000, 1000, 1000, 1000, 1000),
}


def split_dataset(behavior, raw_dir=RAW_DIR, out_dir=BEHAVIOR_DIR,
                  train_ratio=TRAIN_RATIO):
    """Split one behaviour and write both halves.  Returns (n_train, n_test)."""
    input_path = Path(raw_dir) / f'{behavior}.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if behavior not in PRECEDING_LENGTHS:
        raise SystemExit(
            f"{behavior} is not one of the behaviours this split reproduces "
            f"({', '.join(sorted(PRECEDING_LENGTHS))}). Splitting it with a "
            f"bare seed would give a different generation set from the one the "
            f"released vectors were extracted from."
        )

    random.seed(SEED)
    for length in PRECEDING_LENGTHS[behavior]:
        random.shuffle([0] * length)
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    behavior_out = Path(out_dir) / behavior
    behavior_out.mkdir(parents=True, exist_ok=True)

    for records, filename in ((train_data, 'generate_dataset.json'),
                              (test_data, 'test_dataset_ab.json')):
        with open(behavior_out / filename, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4, ensure_ascii=False)

    print(f"{behavior}: {len(data)} items -> "
          f"{len(train_data)} generate, {len(test_data)} test  ({behavior_out})")
    return len(train_data), len(test_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--behaviors', nargs='+', required=True,
                        help='Behaviours to split.')
    parser.add_argument('--raw-dir', default=str(RAW_DIR),
                        help='Directory holding <behavior>.json.')
    parser.add_argument('--out-dir', default=str(BEHAVIOR_DIR),
                        help='Directory to write data/behaviors/<behavior>/ under.')
    args = parser.parse_args()

    for behavior in args.behaviors:
        split_dataset(behavior, raw_dir=args.raw_dir, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
