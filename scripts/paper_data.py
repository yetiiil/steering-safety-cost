"""Restore the fixed paper inputs without sampling or network access.

Only input prompts are packaged. Model responses, scores and analysis are not
included. File identities and provenance live in data/paper/manifest.json.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "data" / "paper"
PAPER_GROUPS = (
    "train", "dim-splits", "dim-eval-set", "behaviors-main", "xstest",
    "orbench", "alpaca-eval", "jailbreakbench", "attacks-autodan",
    "attacks-pair", "attacks-gcg",
)


def _manifest():
    manifest = json.loads((PAPER_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported paper input manifest version")
    return manifest


def _entries(manifest, group):
    if group is not None and group not in PAPER_GROUPS:
        raise ValueError(f"Unknown paper input group: {group}")
    return [entry for entry in manifest["files"]
            if group is None or entry["group"] == group]


def _destination(relative_path):
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid paper input path: {relative_path}")
    destination = (REPO_ROOT / relative).resolve()
    if not destination.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"Paper input path escapes repository: {relative_path}")
    return destination


def _matches(data, entry):
    return (len(data) == entry["bytes"]
            and hashlib.sha256(data).hexdigest() == entry["sha256"])


def paper_paths(group=None):
    return [_destination(entry["path"]) for entry in _entries(_manifest(), group)]


def prepare_paper_data(group=None, force=False):
    """Check the archive and restore missing inputs; protect modified inputs.

    `force=True` restores the recorded bytes even when a destination differs.
    All selected payloads and destinations are checked before writing any file.
    """
    manifest = _manifest()
    archive_path = PAPER_DIR / manifest["archive"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError(f"Paper input archive checksum mismatch: {archive_path}")

    pending = []
    entries = _entries(manifest, group)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {
            entry["path"] for entry in manifest["files"]
        }:
            raise ValueError("Paper input archive members differ from the manifest")
        for entry in entries:
            data = archive.read(entry["path"])
            if not _matches(data, entry):
                raise ValueError(f"Paper input checksum mismatch: {entry['path']}")
            destination = _destination(entry["path"])
            if destination.exists():
                if _matches(destination.read_bytes(), entry):
                    continue
                if not force:
                    raise ValueError(
                        f"{entry['path']} differs from the fixed paper input. "
                        "Use --force to restore the recorded file."
                    )
            pending.append((destination, data))

    for destination, data in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"[paper] {group or 'all'}: {len(entries)} inputs verified, "
          f"{len(pending)} restored")

