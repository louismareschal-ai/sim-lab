from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse

import main

app = FastAPI(title="sim-lab service", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> HTMLResponse:
        html = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>sim-lab</title>
        <link rel="icon" href="/favicon.ico" />
    </head>
    <body style="font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5;">
        <h1 style="margin: 0 0 0.5rem 0;">sim-lab</h1>
        <p style="margin-top: 0;">Tiny evolution simulation wrapped as a minimal web service.</p>
        <ul>
            <li><a href="/simulate?task=center&seed=42">/simulate?task=center&seed=42</a></li>
            <li><a href="/docs">/docs</a></li>
            <li><a href="/health">/health</a></li>
        </ul>
    </body>
</html>
"""
        return HTMLResponse(content=html)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")


@app.get("/simulate")
def simulate(
    task: str = Query(default="center", description="Task name: center | wall | corner"),
    seed: int = Query(default=42, ge=0, description="Random seed for deterministic run"),
) -> dict[str, object]:
    genes = main.random_genes()
    score = main.simulate_task_once(genes=genes, task_name=task, run_seed=seed)

    return {
        "task": task,
        "seed": seed,
        "score": score,
    }