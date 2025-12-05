#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Compress a dataset run into chunks and upload to Modal with resume support.

Pipeline: upload -> extract -> preprocess

Usage:
    uv run python -m modal_apps.upload              # Upload and auto-preprocess
    uv run python -m modal_apps.upload --dry        # Upload only, no preprocess
    uv run python -m modal_apps.upload --no-resume  # Force re-upload all chunks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

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
DATASETS_ROOT = Path("datasets")
CHUNK_SIZE_MB = 500  # Size of each chunk in MB


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Archive and upload a dataset run to Modal.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Dataset subdirectory under ./datasets to upload (defaults to newest run).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_MB,
        help=f"Chunk size in MB (default: {CHUNK_SIZE_MB})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume a previously interrupted upload (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume and re-upload all chunks",
    )
    return parser.parse_args()


def pick_latest_run() -> Path:
    """Find the most recently modified dataset run directory."""
    if not DATASETS_ROOT.exists():
        raise SystemExit("datasets/ directory not found")
    runs = [p for p in DATASETS_ROOT.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit("No dataset runs found under ./datasets")
    return max(runs, key=lambda p: p.stat().st_mtime)


def file_md5(path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_archive(run_path: Path, temp_dir: Path) -> Path:
    """Create a gzipped tarball of the dataset run."""
    archive_path = temp_dir / f"{run_path.name}.tar.gz"
    print(f"Creating archive {archive_path.name}...")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(run_path, arcname=run_path.name)
    return archive_path


def split_archive(archive_path: Path, chunk_size_mb: int) -> list[Path]:
    """Split archive into chunks of specified size."""
    chunk_size = chunk_size_mb * 1024 * 1024
    archive_size = archive_path.stat().st_size
    num_chunks = math.ceil(archive_size / chunk_size)

    if num_chunks == 1:
        return [archive_path]

    print(f"Splitting into {num_chunks} chunks of {chunk_size_mb}MB each...")
    chunks = []
    with open(archive_path, "rb") as f:
        for i in range(num_chunks):
            chunk_path = archive_path.parent / f"{archive_path.stem}.part{i:03d}"
            with open(chunk_path, "wb") as chunk_file:
                chunk_file.write(f.read(chunk_size))
            chunks.append(chunk_path)
            print(f"  Created {chunk_path.name}")

    return chunks


def create_manifest(
    ds_name: str, chunks: list[Path], temp_dir: Path
) -> tuple[Path, dict[str, str]]:
    """Create a manifest file with chunk info and checksums."""
    chunks_dict: dict[str, dict[str, object]] = {}
    checksums: dict[str, str] = {}

    for chunk in chunks:
        checksum = file_md5(chunk)
        chunks_dict[chunk.name] = {
            "size": chunk.stat().st_size,
            "md5": checksum,
        }
        checksums[chunk.name] = checksum

    manifest: dict[str, object] = {
        "ds_name": ds_name,
        "num_chunks": len(chunks),
        "chunks": chunks_dict,
    }

    manifest_path = temp_dir / f"{ds_name}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path, checksums


def get_uploaded_chunks(ds_name: str) -> dict[str, str]:
    """Get list of already uploaded chunks from Modal volume."""
    try:
        result = subprocess.run(
            ["uvx", "modal", "volume", "ls", DEFAULT_VOLUME, f"/datasets/{ds_name}_chunks/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return {}

        # Parse the ls output to get filenames
        uploaded = {}
        for line in result.stdout.strip().split("\n"):
            if line and not line.startswith("Directory"):
                # Extract filename from ls output
                parts = line.split()
                if parts:
                    filename = parts[-1]
                    if filename.endswith(".md5"):
                        continue
                    # Try to get the checksum file
                    md5_result = subprocess.run(
                        [
                            "uvx",
                            "modal",
                            "volume",
                            "get",
                            DEFAULT_VOLUME,
                            f"/datasets/{ds_name}_chunks/{filename}.md5",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if md5_result.returncode == 0:
                        uploaded[filename] = md5_result.stdout.strip()
        return uploaded
    except Exception:
        return {}


def ensure_volume() -> None:
    """Create the Modal volume if it doesn't exist."""
    subprocess.run(["uvx", "modal", "volume", "create", DEFAULT_VOLUME], check=False)


def upload_chunk(chunk_path: Path, ds_name: str, checksum: str) -> None:
    """Upload a single chunk to the Modal volume."""
    chunk_name = chunk_path.name
    remote_path = f"/datasets/{ds_name}_chunks/{chunk_name}"

    # Upload the chunk
    subprocess.run(
        [
            "uvx",
            "modal",
            "volume",
            "put",
            "-f",
            DEFAULT_VOLUME,
            str(chunk_path),
            remote_path,
        ],
        check=True,
    )

    # Upload checksum file for resume verification
    checksum_path = chunk_path.parent / f"{chunk_name}.md5"
    with open(checksum_path, "w") as f:
        f.write(checksum)

    subprocess.run(
        [
            "uvx",
            "modal",
            "volume",
            "put",
            "-f",
            DEFAULT_VOLUME,
            str(checksum_path),
            f"{remote_path}.md5",
        ],
        check=True,
    )


def upload_manifest(manifest_path: Path, ds_name: str) -> None:
    """Upload the manifest file."""
    subprocess.run(
        [
            "uvx",
            "modal",
            "volume",
            "put",
            "-f",
            DEFAULT_VOLUME,
            str(manifest_path),
            f"/datasets/{ds_name}_chunks/{manifest_path.name}",
        ],
        check=True,
    )


def main() -> None:
    """Archive and upload a dataset run to Modal with chunking and resume support."""
    args = parse_args()
    run_path = args.run_dir if args.run_dir else pick_latest_run()
    run_path = run_path.resolve()
    if not run_path.exists():
        raise SystemExit(f"Run path {run_path} does not exist")

    ds_name = run_path.name
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_archive_"))

    try:
        ensure_volume()

        # Check for existing uploads if resuming
        uploaded_chunks: dict[str, str] = {}
        if args.resume:
            print("Checking for previously uploaded chunks...")
            uploaded_chunks = get_uploaded_chunks(ds_name)
            if uploaded_chunks:
                print(f"Found {len(uploaded_chunks)} previously uploaded chunks")

        # Create archive
        archive_path = create_archive(run_path, temp_dir)
        total_size = archive_path.stat().st_size
        print(f"Archive size: {total_size / (1024 * 1024):.1f}MB")

        # Split into chunks
        chunks = split_archive(archive_path, args.chunk_size)

        # Create manifest with checksums
        manifest_path, checksums = create_manifest(ds_name, chunks, temp_dir)

        # Upload chunks (skip already uploaded ones with matching checksums)
        for chunk in chunks:
            chunk_name = chunk.name
            expected_checksum = checksums[chunk_name]

            if chunk_name in uploaded_chunks:
                if uploaded_chunks[chunk_name] == expected_checksum:
                    print(f"Skipping {chunk_name} (already uploaded, checksum matches)")
                    continue
                else:
                    print(f"Re-uploading {chunk_name} (checksum mismatch)")

            print(f"Uploading {chunk_name}...")
            upload_chunk(chunk, ds_name, expected_checksum)

        # Upload manifest
        print("Uploading manifest...")
        upload_manifest(manifest_path, ds_name)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nUploaded {ds_name} to Modal volume '{DEFAULT_VOLUME}' ({len(chunks)} chunks)")
    # Output dataset name for shell script to use
    print(f"DATASET_NAME={ds_name}")


if __name__ == "__main__":
    main()
