# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""CUDAG Server - FastAPI server for annotation-to-generator workflow.

This module provides a local HTTP server that the Annotator UI can call
to generate CUDAG projects from annotations without using the terminal.

Start the server:
    cudag serve --port 8420

The server exposes:
    GET  /health           - Health check
    POST /api/v1/generate  - Generate project from annotation
    GET  /api/v1/status/{job_id} - Check generation progress
"""

from cudag.server.app import create_app, run_server

__all__ = ["create_app", "run_server"]
