#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Archive datasets from Modal volume to archive volume.

Compresses the raw dataset into a .tgz archive.

Volume structure:
    claimhawk-archives/
        datasets/[ds-name].tgz
        loras/[ds-name]/[run-name].tgz
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
    TRAINING_VOLUME = get_volume_name("lora_training")
    ARCHIVE_VOLUME = get_volume_name("archives")
except ImportError:
    # Fallback when SDK not available
    TRAINING_VOLUME = "claimhawk-lora-training"
    ARCHIVE_VOLUME = "claimhawk-archives"


def _get_generator_name() -> str:
    """Extract generator name from --ds-name arg for dynamic app naming."""
    for i, arg in enumerate(sys.argv):
        if arg == "--ds-name" and i + 1 < len(sys.argv):
            ds_name = sys.argv[i + 1]
            return ds_name.split("-")[0] if ds_name else "cudag"
    return "cudag"


app = modal.App(f"{_get_generator_name()}-archive")
training_vol = modal.Volume.from_name(TRAINING_VOLUME, create_if_missing=True)
archive_vol = modal.Volume.from_name(ARCHIVE_VOLUME, create_if_missing=True)


@app.function(
    volumes={
        "/training": training_vol,
        "/archive": archive_vol,
    },
    timeout=1800,  # 30 min for large datasets
)
def archive_dataset(ds_name: str) -> str:
    """Archive a dataset to the archive volume.

    Reads from:
        /training/datasets/[ds_name]/

    Writes to:
        /archive/datasets/[ds_name].tgz
    """
    import tarfile
    from pathlib import Path

    dataset_path = Path(f"/training/datasets/{ds_name}")
    archive_dir = Path("/archive/datasets")
    archive_path = archive_dir / f"{ds_name}.tgz"

    # Verify source path exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating archive: {ds_name}.tgz")
    print(f"  Dataset: {dataset_path}")

    # Create tar.gz archive
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dataset_path, arcname=ds_name)

    # Get archive size
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive size: {size_mb:.1f} MB")

    # Commit to volume
    archive_vol.commit()

    print(f"Archived to: /archive/datasets/{ds_name}.tgz")
    return str(archive_path)


@app.local_entrypoint()
def main(ds_name: str) -> None:
    """Archive a dataset."""
    result = archive_dataset.remote(ds_name)
    print(f"Archive complete: {result}")
