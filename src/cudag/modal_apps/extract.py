#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Modal function to extract uploaded dataset archives (single or chunked) on a volume.

Pipeline: upload_dataset -> modal_extract -> preprocess
"""
from __future__ import annotations

import sys

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names are loaded from config/adapters.yaml via the SDK.
# Users can customize these by editing the YAML file.

try:
    from sdk.modal_compat import get_volume_name
    DEFAULT_VOLUME = get_volume_name("lora_training")
except ImportError:
    # Fallback when SDK not available
    DEFAULT_VOLUME = "claimhawk-lora-training"


def _get_generator_name() -> str:
    """Extract generator name from --dataset-name arg for dynamic app naming."""
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset-name" and i + 1 < len(sys.argv):
            ds_name = sys.argv[i + 1]
            return ds_name.split("-")[0] if ds_name else "cudag"
    return "cudag"


app = modal.App(f"{_get_generator_name()}-extract")
VOLUME = modal.Volume.from_name(DEFAULT_VOLUME, create_if_missing=True)


@app.function(volumes={"/data": VOLUME}, timeout=600)
def extract(ds_name: str) -> str:
    """Extract a tarball (single or chunked) on the Modal volume."""
    import json
    import shutil
    import tarfile
    from pathlib import Path

    datasets_dir = Path("/data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    chunks_dir = datasets_dir / f"{ds_name}_chunks"
    legacy_archive = datasets_dir / f"{ds_name}.tar.gz"
    extract_dir = datasets_dir

    # Check for chunked upload first
    if chunks_dir.exists():
        manifest_path = chunks_dir / f"{ds_name}.manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        num_chunks = manifest["num_chunks"]
        print(f"Reassembling {num_chunks} chunks...")

        # Reassemble the archive from chunks
        reassembled_path = datasets_dir / f"{ds_name}.tar.gz"

        # Handle single-chunk case (archive wasn't split)
        if num_chunks == 1:
            # Find the single chunk (could be .tar.gz or other naming)
            chunk_names = list(manifest["chunks"].keys())
            if chunk_names:
                chunk_path = chunks_dir / chunk_names[0]
                if chunk_path.exists():
                    print(f"  Moving single chunk {chunk_path.name}")
                    shutil.copy2(chunk_path, reassembled_path)
                else:
                    raise FileNotFoundError(f"Chunk not found: {chunk_path}")
            else:
                raise FileNotFoundError("No chunks found in manifest")
        else:
            with open(reassembled_path, "wb") as outfile:
                for i in range(num_chunks):
                    # Try to find the chunk with different naming patterns
                    chunk_path = None
                    for name in manifest["chunks"]:
                        if f"part{i:03d}" in name:
                            chunk_path = chunks_dir / name
                            break

                    if chunk_path is None or not chunk_path.exists():
                        raise FileNotFoundError(f"Chunk {i} not found")

                    print(f"  Adding {chunk_path.name}")
                    with open(chunk_path, "rb") as chunk:
                        outfile.write(chunk.read())

        # Extract the reassembled archive
        print("Extracting archive...")
        with tarfile.open(reassembled_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")

        # Cleanup: remove chunks directory and reassembled archive
        shutil.rmtree(chunks_dir)
        reassembled_path.unlink()
        VOLUME.commit()
        print(f"Extracted {ds_name} to /data/datasets/{ds_name}")

    # Fall back to legacy single-file archive
    elif legacy_archive.exists():
        print("Extracting single archive...")
        with tarfile.open(legacy_archive, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")

        legacy_archive.unlink()
        VOLUME.commit()
        print(f"Extracted {ds_name} to /data/datasets/{ds_name}")

    else:
        raise FileNotFoundError(
            f"No archive found for {ds_name}. "
            f"Checked: {chunks_dir} and {legacy_archive}"
        )

    return ds_name


@app.local_entrypoint()
def main(dataset_name: str) -> None:
    """Entry point for modal run command."""
    result = extract.remote(dataset_name)
    print(f"Extraction complete: {result}")
