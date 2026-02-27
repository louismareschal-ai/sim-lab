from __future__ import annotations

from fastapi import FastAPI, Query

import main

app = FastAPI(title="sim-lab service", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "sim-lab",
        "mode": "web-wrapper",
        "description": "Tiny evolution simulation wrapper around CLI demo.",
        "endpoints": ["/health", "/simulate"],
    }


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