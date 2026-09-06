#!/usr/bin/env python
"""
Download the trained model artifacts from the project's GitHub release into
``models/``.

The pickled pipelines total ~200 MB (one is 167 MB), which is over GitHub's
100 MB per-file limit for the git repository itself, so they are published as
release assets instead.

Usage:
    python scripts/download_models.py            # fetch anything missing
    python scripts/download_models.py --force    # re-download everything

Environment overrides:
    MODELS_RELEASE_TAG   release tag to pull from        (default: v1.0.0)
    MODELS_RELEASE_REPO  owner/repo to pull from          (default: OutisAyo/council-classifier-)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.environ.get("MODELS_RELEASE_REPO", "OutisAyo/council-classifier-")
TAG = os.environ.get("MODELS_RELEASE_TAG", "v1.0.0")
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ARTIFACTS = (
    "department_pipeline_rf.pkl",
    "priority_pipeline_final.pkl",
    "retrieval_vectorizer.pkl",
    "response_templates.pkl",
    "urgency_lexicon.pkl",
    "dataset_for_retrieval.pkl",
)


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed github.com host
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = read * 100 // total
                    print(f"\r  {dest.name}: {pct:3d}%  ({read >> 20}/{total >> 20} MB)", end="")
        print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching model artifacts from {REPO}@{TAG}\n")

    failures = []
    for name in ARTIFACTS:
        dest = MODELS_DIR / name
        if dest.exists() and not args.force:
            print(f"  {name}: already present, skipping")
            continue
        url = f"{BASE_URL}/{name}"
        try:
            _download(url, dest)
        except urllib.error.HTTPError as exc:
            failures.append((name, f"HTTP {exc.code}"))
            print(f"  {name}: FAILED ({exc.code} {exc.reason})")
        except urllib.error.URLError as exc:
            failures.append((name, str(exc.reason)))
            print(f"  {name}: FAILED ({exc.reason})")

    if failures:
        print(
            "\nSome artifacts could not be downloaded. Confirm the release "
            f"'{TAG}' exists at https://github.com/{REPO}/releases and has the "
            "assets attached, or regenerate them by running the notebooks.",
            file=sys.stderr,
        )
        return 1

    print("\nAll model artifacts are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
