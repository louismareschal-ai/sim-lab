# AI Game — Tiny Evolution (MVP)

This is a simple starter game where a population of tiny creatures evolves over generations.
Each creature has a small policy (weights/"genes") and learns to move toward food in a 2D world.

## Why this approach

- Starts **simple** and fast to iterate.
- Uses a genetic algorithm (evolution) as a first step toward NEAT-like evolution.
- Very lightweight dependencies, with optional GUI rendering via `pygame`.

## Install

From this folder:

```bash
python3 -m pip install -r requirements.txt
```

## Run

From this folder:

```bash
python3 main.py
```

You should see generation stats improving over time (best/average fitness).

Default profile now uses:

- `biased-corners` spawn (not fully random, but not fully fixed corners either)
- progressive episode length ramp (`120 -> 320` steps over first `500` generations)

This is tuned to make evolution patterns easier to observe over long runs.

## Live Rendering (GUI)

Run the pygame window renderer:

```bash
python3 main.py --renderer pygame --fullscreen --fps 4 --render-step-skip 1
```

Evolution-pattern run (corner spawns + longer generations over time):

```bash
python3 main.py --renderer pygame --early-stop off --spawn-mode corners --steps-growth-final 320 --steps-growth-generations 500 --log-interval 20

# Clockwise food pattern (BR -> BL -> TL -> TR every ~5s-like window)
python3 main.py --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1
```

Shortcut:

```bash
python3 main.py --pygame
```

In the pygame UI:

- Opens fullscreen by default via `just render`.
- Right panel shows a persistent **Legend** and a live **Tutorial** of what phase is happening.
- Hover any world cell to see a tooltip (food, species, local energy, predator/obstacle/meat).
- Live chart shows best/average fitness over generations.
- Checkpoint benchmark table compares the same tasks at generations **0**, **100**, and **1000**.
- Species mix and event counters are updated in real time.
- Foreground uses a **focus mode** (1 main agent by default) with trails and optional target lines so humans can follow learning.
- Background still runs fast; foreground showcases selected generations in slow motion.
- A click-through onboarding tutorial appears on top; use **Next/Back/Finish** (mouse) to step through.
- Simulation waits for onboarding completion before progressing (no background progress while tutorial is active).
- Checkpoints are non-blocking; you get a short "checkpoint ready" notice instead of a freeze.
- Clickable controls in sidebar: **Pause/Resume**, **Tip Back/Next**, **Follow A/B/C**.
- `just render` runs with `--early-stop off`, so noisy/plateau phases do not force-stop your session.
- Training now uses adaptive mutation (strong early, softer later) and rollback recovery when performance drifts too far from the best known state.
- Rendering now follows **train then watch** cadence: most generations train fast in background, then milestone generations (1, 200, 400, ..., 1000) are shown as a ~30s focused playback of one main agent.
- Press `H` to hide/show the tutorial.
- Press `G` to hide/show the chart.
- Press `M` to toggle `showcase mode` (fast background sim + slow visual focus).
- Press `L` to toggle focus target lines.
- Press `Space` to pause/unpause.
- Press `Esc` or click the window `X` to close.

Fallback terminal renderer:

```bash
python3 main.py --renderer terminal --fps 24 --render-step-skip 2
```

Tips:

- Increase `--render-step-skip` for faster simulation.
- Lower `--fps` to slow down and better follow behavior.
- Use `--generations 10` for quick feedback loops.
- Use `--spawn-mode corners` to force births near corners and observe spread patterns.
- Use `--steps-growth-final 300 --steps-growth-generations 400` to progressively lengthen generations.
- In pygame mode, the window stays open at the end; close with `Esc` or the window `X`.
- Difficulty levels ramp automatically: Level 1 basics, Level 2 obstacles+predators, Level 3 battles+meat.
- Simulation auto-stops early if learning plateaus or gets noisy with no clear trend.

## Test / Check Before Running

From this folder:

```bash
python3 -m unittest discover -s tests -v
```

Or run the one-command check script:

```bash
bash check.sh
```

This gives you a fast regression check before trying new gameplay changes.

## One-command workflow with just

If you use `just`, these commands are available:

```bash
just test
just render
just ci
just scenario-base
just scenario-evolve
just scenario-corners
just scenario-clockwise
just scenario-clockwise-force
just scenario-clockwise-best10
just scenario-clockwise-blockers
just scenario-clockwise-best10-blockers
just scenario-tuned-staged
just scenario-progressive-peak
just sweep-local
just experience

# Optional long-run integration soak (not part of default test flow)
just soak-1000

# Train snapshot models at 100/500/1000/2000 generations (offline)
just train-2000

# Serve trained snapshots in UI (switch between training levels live)
just serve-models

# Full requested workflow
just train-and-render
```

`just render` now serves trained snapshots (it no longer runs live training).
In serve mode, playback stays slow and continuous by default.
If you want sparse showcase behavior, override with:

```bash
GAME_ARGS="--showcase-interval-generations 200" just render
```

Scenario guide:

- `just scenario-base`: near old behavior (random spawn, fixed short episodes).
- `just scenario-evolve`: recommended default evolution profile.
- `just scenario-corners`: strongest corner bias to test pattern learning under pressure.
- `just scenario-clockwise`: trains/uses a dedicated clockwise model file, then serves with snapshot switching enabled + strict clockwise food zones.
- `just scenario-clockwise-force`: always retrains the clockwise model before serving (use this when testing learning changes).
- `just scenario-clockwise-best10`: serves 10 clones of snapshot best agent for behavior-convergence checks.
- `just scenario-clockwise-blockers`: enables simple blocker interference so agents can hinder rivals.
- `just scenario-clockwise-best10-blockers`: same as above, but with 10 clones of best agent.
- `just scenario-tuned-staged`: recommended local profile for stable learning (easy start, progressive difficulty ramp).
- `just scenario-progressive-peak`: aggressive profile that often reaches higher best-agent peaks.
- `just sweep-local`: runs easy/progressive/gentle/tuned training sweep and writes logs under `runs/`.
- `just experience`: user-oriented showcase (auto-train tuned snapshot if missing, then launches pygame serve mode).

## Train once, serve many times

You can now separate training and visualization:

- Train in background and save snapshots: generation 100, 500, 1000, 2000.
- Serve snapshots interactively in pygame, then switch snapshot while running.

Serve controls:

- `1..9` or `[` / `]`: choose which trained snapshot to view.
- Click snapshot buttons in the sidebar: direct snapshot selection.
- `Space`: pause/resume.
- `+` / `-`: increase/decrease simulation speed.
- `S` (or `Esc`): stop.

Training behavior:

- Training now starts from an easy baseline and adapts level based on measured progress.
- If progress is stable, it increases difficulty; if learning is noisy/stalled, it relaxes difficulty.
- Logs now include progress status (`warming-up`, `improving`, `stalled`) so you can validate whether training is converging.

The `just` recipes automatically create and use a local `.venv` and install dependencies.

`just render` opens the pygame window.

`just ci` runs both tests and a short smoke simulation.

## CI

A GitHub Actions workflow is available in `.github/workflows/ci.yml` and runs:

- Unit tests
- A short smoke run (`--generations 2`)

## What’s next (when you say go)

1. Add a simple visual map (ASCII or `pygame`).
2. Add reproduction based on energy in-simulation.
3. Introduce predators / obstacles.
4. Move toward **NEAT-style topology evolution**.
5. Add reinforcement-learning hybrid behavior.
