# Data

Fixed inputs for the main experiments are included in
[`paper/inputs.zip`](paper/inputs.zip). Restore them from the repository root:

```bash
python scripts/00_download_data.py
```

The archive restores 26 files for CAST and DIM training, the three main behavior
datasets, benign evaluation, and attack evaluation. File hashes, record counts,
provenance, and preprocessing are recorded in
[`paper/manifest.json`](paper/manifest.json). Extracted inputs are gitignored.

See the [Data Card](../DATA_CARD.md) for dataset sources, split definitions, and
usage terms, and the [README](../README.md) for the training and evaluation
workflow.
