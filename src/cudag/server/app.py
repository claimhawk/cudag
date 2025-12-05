# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""FastAPI application for CUDAG server."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cudag import __version__
from cudag.server.services.generator import GeneratorService

# In-memory job storage (for MVP - would use Redis/DB in production)
_jobs: dict[str, dict[str, Any]] = {}


class GenerateOptions(BaseModel):
    """Options for project generation."""

    project_name: str = Field(..., description="Name for the generated project")
    output_dir: str | None = Field(None, description="Output directory (default: ~/cudag-projects)")
    num_samples: int = Field(1000, description="Number of samples per task")
    generate_immediately: bool = Field(True, description="Run generation after scaffolding")


class GenerateRequest(BaseModel):
    """Request body for generate endpoint."""

    annotation: dict = Field(..., description="Full annotation.json data")
    original_image: str = Field(..., description="Base64 encoded original image")
    masked_image: str | None = Field(None, description="Base64 encoded masked image")
    icons: dict[str, str] | None = Field(None, description="Map of icon names to base64 images")
    options: GenerateOptions


class GenerateResponse(BaseModel):
    """Response from generate endpoint."""

    status: str
    project_path: str | None = None
    files_created: list[str] | None = None
    job_id: str | None = None
    error: str | None = None


class StatusResponse(BaseModel):
    """Response from status endpoint."""

    progress: int
    total: int
    current_task: str | None = None
    done: bool
    error: str | None = None


class HealthResponse(BaseModel):
    """Response from health endpoint."""

    status: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    yield
    # Shutdown - cleanup jobs
    _jobs.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CUDAG Server",
        description="Generate CUDAG projects from annotations",
        version=__version__,
        lifespan=lifespan,
    )

    # Configure CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Check server health."""
        return HealthResponse(status="healthy", version=__version__)

    @app.post("/api/v1/generate", response_model=GenerateResponse)
    async def generate_project(
        request: GenerateRequest,
        background_tasks: BackgroundTasks,
    ) -> GenerateResponse:
        """Generate a CUDAG project from annotation data."""
        try:
            service = GeneratorService()

            # Validate annotation
            validation_error = service.validate_annotation(request.annotation)
            if validation_error:
                return GenerateResponse(status="error", error=validation_error)

            # Determine output directory
            output_dir = Path(request.options.output_dir or "~/cudag-projects").expanduser()
            project_dir = output_dir / request.options.project_name

            # Scaffold the project
            files_created = service.scaffold_project(
                annotation=request.annotation,
                original_image=request.original_image,
                masked_image=request.masked_image,
                icons=request.icons,
                project_dir=project_dir,
            )

            # If generate_immediately, run generation in background
            if request.options.generate_immediately:
                job_id = str(uuid.uuid4())
                _jobs[job_id] = {
                    "progress": 0,
                    "total": request.options.num_samples,
                    "current_task": "Starting generation...",
                    "done": False,
                    "error": None,
                    "project_dir": str(project_dir),
                }

                background_tasks.add_task(
                    _run_generation,
                    job_id=job_id,
                    project_dir=project_dir,
                    num_samples=request.options.num_samples,
                )

                return GenerateResponse(
                    status="generating",
                    project_path=str(project_dir),
                    files_created=files_created,
                    job_id=job_id,
                )

            return GenerateResponse(
                status="success",
                project_path=str(project_dir),
                files_created=files_created,
            )

        except Exception as e:
            return GenerateResponse(status="error", error=str(e))

    @app.get("/api/v1/status/{job_id}", response_model=StatusResponse)
    async def get_status(job_id: str) -> StatusResponse:
        """Get the status of a generation job."""
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = _jobs[job_id]
        return StatusResponse(
            progress=job["progress"],
            total=job["total"],
            current_task=job.get("current_task"),
            done=job["done"],
            error=job.get("error"),
        )

    return app


async def _run_generation(
    job_id: str,
    project_dir: Path,
    num_samples: int,
) -> None:
    """Run dataset generation in background."""
    try:
        service = GeneratorService()

        # Update progress callback
        def on_progress(progress: int, task: str) -> None:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = progress
                _jobs[job_id]["current_task"] = task

        await asyncio.to_thread(
            service.run_generation,
            project_dir=project_dir,
            num_samples=num_samples,
            progress_callback=on_progress,
        )

        if job_id in _jobs:
            _jobs[job_id]["done"] = True
            _jobs[job_id]["current_task"] = "Generation complete"

    except Exception as e:
        if job_id in _jobs:
            _jobs[job_id]["done"] = True
            _jobs[job_id]["error"] = str(e)


def run_server(host: str = "127.0.0.1", port: int = 8420, reload: bool = False) -> None:
    """Run the CUDAG server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    uvicorn.run(
        "cudag.server.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
