from __future__ import annotations

import random
import signal
import time
from typing import Callable

import game_logic as core
from game_constants import (
    BENCHMARK_TRIALS,
    BLOCKER_ENERGY_COST,
    BLOCKER_TTL_STEPS,
    CHECKPOINT_GENERATIONS,
    DEFAULT_FOOD_RESPAWN_MODE,
    DEFAULT_INTERFERENCE_MODE,
    DEFAULT_SPAWN_MODE,
    DEFAULT_STEPS_GROWTH_FINAL,
    DEFAULT_STEPS_GROWTH_GENERATIONS,
    DEFAULT_SNAPSHOT_GENERATIONS,
    ELITE_COUNT,
    ENERGY_REWARD_SCALE,
    FOOD_COUNT,
    FOOD_RESPAWN_BATCH_RATIO,
    FOOD_RESPAWN_CYCLE_STEPS,
    FOOD_RESPAWN_INTERVAL_STEPS,
    FOOD_PRIMARY_CAP,
    FOOD_REWARD_PRIMARY,
    FOOD_REWARD_SECONDARY,
    GENERATIONS,
    MUTATION_RATE,
    MUTATION_STRENGTH,
    POPULATION_SIZE,
    SEED,
    SHOWCASE_INTERVAL_GENERATIONS,
    SHOWCASE_PLAYBACK_SECONDS,
    SPECIES_NAMES,
    SPOTLIGHT_SECONDS,
    STEPS_PER_GENERATION,
    SURVIVAL_REWARD_PER_STEP,
    WORLD_SIZE,
)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _sync_runtime_constants() -> None:
    synced = {
        "WORLD_SIZE": WORLD_SIZE,
        "POPULATION_SIZE": POPULATION_SIZE,
        "FOOD_COUNT": FOOD_COUNT,
        "GENERATIONS": GENERATIONS,
        "STEPS_PER_GENERATION": STEPS_PER_GENERATION,
        "MUTATION_RATE": MUTATION_RATE,
        "MUTATION_STRENGTH": MUTATION_STRENGTH,
        "ELITE_COUNT": ELITE_COUNT,
        "SEED": SEED,
        "CHECKPOINT_GENERATIONS": CHECKPOINT_GENERATIONS,
        "BENCHMARK_TRIALS": BENCHMARK_TRIALS,
        "SPOTLIGHT_SECONDS": SPOTLIGHT_SECONDS,
        "SHOWCASE_INTERVAL_GENERATIONS": SHOWCASE_INTERVAL_GENERATIONS,
        "SHOWCASE_PLAYBACK_SECONDS": SHOWCASE_PLAYBACK_SECONDS,
        "DEFAULT_SNAPSHOT_GENERATIONS": DEFAULT_SNAPSHOT_GENERATIONS,
        "DEFAULT_FOOD_RESPAWN_MODE": DEFAULT_FOOD_RESPAWN_MODE,
        "FOOD_RESPAWN_INTERVAL_STEPS": FOOD_RESPAWN_INTERVAL_STEPS,
        "FOOD_RESPAWN_BATCH_RATIO": FOOD_RESPAWN_BATCH_RATIO,
        "FOOD_RESPAWN_CYCLE_STEPS": FOOD_RESPAWN_CYCLE_STEPS,
        "DEFAULT_INTERFERENCE_MODE": DEFAULT_INTERFERENCE_MODE,
        "BLOCKER_TTL_STEPS": BLOCKER_TTL_STEPS,
        "BLOCKER_ENERGY_COST": BLOCKER_ENERGY_COST,
        "DEFAULT_SPAWN_MODE": DEFAULT_SPAWN_MODE,
        "DEFAULT_STEPS_GROWTH_FINAL": DEFAULT_STEPS_GROWTH_FINAL,
        "DEFAULT_STEPS_GROWTH_GENERATIONS": DEFAULT_STEPS_GROWTH_GENERATIONS,
        "SURVIVAL_REWARD_PER_STEP": SURVIVAL_REWARD_PER_STEP,
        "FOOD_REWARD_PRIMARY": FOOD_REWARD_PRIMARY,
        "FOOD_REWARD_SECONDARY": FOOD_REWARD_SECONDARY,
        "FOOD_PRIMARY_CAP": FOOD_PRIMARY_CAP,
        "ENERGY_REWARD_SCALE": ENERGY_REWARD_SCALE,
        "SPECIES_NAMES": SPECIES_NAMES,
    }
    for name, value in synced.items():
        setattr(core, name, value)
    core.random = random
    core.time = time


def safe_print(*args, **kwargs) -> None:
    return core.safe_print(*args, **kwargs)


Creature = core.Creature
TerminalRenderer = core.TerminalRenderer
PygameRenderer = core.PygameRenderer


def build_renderer(renderer_mode: str, fps: float, step_skip: int, fullscreen: bool = False):
    _sync_runtime_constants()
    return core.build_renderer(renderer_mode, fps=fps, step_skip=step_skip, fullscreen=fullscreen)


classify_species = core.classify_species
spawn_obstacles = core.spawn_obstacles
spawn_predators = core.spawn_predators
step_towards = core.step_towards
random_genes = core.random_genes
spawn_food = core.spawn_food
replenish_food = core.replenish_food
metabolic_drain_for_step = core.metabolic_drain_for_step
creature_fitness = core.creature_fitness
nearest_food_vector = core.nearest_food_vector
wall_features = core.wall_features
crossover = core.crossover
mutate_with_parameters = core.mutate_with_parameters
mutation_schedule = core.mutation_schedule
should_rollback_to_best = core.should_rollback_to_best
level_for_generation = core.level_for_generation
decide_move_from_genes = core.decide_move_from_genes
simulate_task_once = core.simulate_task_once
benchmark_trial_worker = core.benchmark_trial_worker
evaluate_checkpoint_benchmark = core.evaluate_checkpoint_benchmark
run_checkpoint_benchmark_job = core.run_checkpoint_benchmark_job
select_initial_champion = core.select_initial_champion
advance_onboarding_state = core.advance_onboarding_state
checkpoint_requires_resume = core.checkpoint_requires_resume
checkpoint_should_defer = core.checkpoint_should_defer
should_stop_early = core.should_stop_early
resolve_early_stop_mode = core.resolve_early_stop_mode
playback_fps_for_steps = core.playback_fps_for_steps
should_log_generation = core.should_log_generation
should_showcase_generation = core.should_showcase_generation
parse_checkpoint_generations = core.parse_checkpoint_generations
compute_progress_signal = core.compute_progress_signal
adapt_curriculum_level = core.adapt_curriculum_level
save_snapshot_file = core.save_snapshot_file
load_snapshot_file = core.load_snapshot_file
analyze_stall = core.analyze_stall
interactive_train_and_serve = core.interactive_train_and_serve


def mutate(genes: list[float]) -> list[float]:
    _sync_runtime_constants()
    return core.mutate(genes)


def run_generation(
    population_genes: list[list[float]],
    step_callback: Callable[[int, list[Creature], set[tuple[int, int]]], None] | None = None,
    level: int = 1,
    food_coverage_init: float = 1.0,
    spawn_mode: str = DEFAULT_SPAWN_MODE,
    steps_override: int | None = None,
    food_respawn_mode: str = DEFAULT_FOOD_RESPAWN_MODE,
    food_respawn_interval_steps: int = FOOD_RESPAWN_INTERVAL_STEPS,
    food_respawn_batch_ratio: float = FOOD_RESPAWN_BATCH_RATIO,
    food_respawn_cycle_steps: int = FOOD_RESPAWN_CYCLE_STEPS,
    interference_mode: str = DEFAULT_INTERFERENCE_MODE,
    blocker_ttl_steps: int = BLOCKER_TTL_STEPS,
    blocker_energy_cost: float = BLOCKER_ENERGY_COST,
) -> list[tuple[float, list[float], int]]:
    _sync_runtime_constants()
    return core.run_generation(
        population_genes,
        step_callback=step_callback,
        level=level,
        food_coverage_init=food_coverage_init,
        spawn_mode=spawn_mode,
        steps_override=steps_override,
        food_respawn_mode=food_respawn_mode,
        food_respawn_interval_steps=food_respawn_interval_steps,
        food_respawn_batch_ratio=food_respawn_batch_ratio,
        food_respawn_cycle_steps=food_respawn_cycle_steps,
        interference_mode=interference_mode,
        blocker_ttl_steps=blocker_ttl_steps,
        blocker_energy_cost=blocker_energy_cost,
    )


def train_and_save_snapshots(
    generations: int,
    checkpoints: list[int],
    checkpoint_file: str,
    top_k: int = 12,
    seed: int = SEED,
    log_interval: int = 25,
    stall_window: int = core.STALL_WINDOW,
    food_coverage_init: float = 1.0,
    spawn_mode: str = DEFAULT_SPAWN_MODE,
    steps_growth_final: int | None = DEFAULT_STEPS_GROWTH_FINAL,
    steps_growth_generations: int = DEFAULT_STEPS_GROWTH_GENERATIONS,
    food_respawn_mode: str = DEFAULT_FOOD_RESPAWN_MODE,
    food_respawn_interval_steps: int = FOOD_RESPAWN_INTERVAL_STEPS,
    food_respawn_batch_ratio: float = FOOD_RESPAWN_BATCH_RATIO,
    food_respawn_cycle_steps: int = FOOD_RESPAWN_CYCLE_STEPS,
    interference_mode: str = DEFAULT_INTERFERENCE_MODE,
    blocker_ttl_steps: int = BLOCKER_TTL_STEPS,
    blocker_energy_cost: float = BLOCKER_ENERGY_COST,
) -> None:
    _sync_runtime_constants()
    core.train_and_save_snapshots(
        generations=generations,
        checkpoints=checkpoints,
        checkpoint_file=checkpoint_file,
        top_k=top_k,
        seed=seed,
        log_interval=log_interval,
        stall_window=stall_window,
        food_coverage_init=food_coverage_init,
        spawn_mode=spawn_mode,
        steps_growth_final=steps_growth_final,
        steps_growth_generations=steps_growth_generations,
        food_respawn_mode=food_respawn_mode,
        food_respawn_interval_steps=food_respawn_interval_steps,
        food_respawn_batch_ratio=food_respawn_batch_ratio,
        food_respawn_cycle_steps=food_respawn_cycle_steps,
        interference_mode=interference_mode,
        blocker_ttl_steps=blocker_ttl_steps,
        blocker_energy_cost=blocker_energy_cost,
    )


def serve_saved_snapshots(
    checkpoint_file: str,
    renderer_mode: str,
    fps: float,
    render_step_skip: int,
    fullscreen: bool,
    food_coverage_init: float = 1.0,
    spawn_mode: str = DEFAULT_SPAWN_MODE,
    food_respawn_mode: str = DEFAULT_FOOD_RESPAWN_MODE,
    food_respawn_interval_steps: int = FOOD_RESPAWN_INTERVAL_STEPS,
    food_respawn_batch_ratio: float = FOOD_RESPAWN_BATCH_RATIO,
    food_respawn_cycle_steps: int = FOOD_RESPAWN_CYCLE_STEPS,
    serve_best_clones: int = 0,
    interference_mode: str = DEFAULT_INTERFERENCE_MODE,
    blocker_ttl_steps: int = BLOCKER_TTL_STEPS,
    blocker_energy_cost: float = BLOCKER_ENERGY_COST,
) -> None:
    _sync_runtime_constants()
    core.serve_saved_snapshots(
        checkpoint_file=checkpoint_file,
        renderer_mode=renderer_mode,
        fps=fps,
        render_step_skip=render_step_skip,
        fullscreen=fullscreen,
        food_coverage_init=food_coverage_init,
        spawn_mode=spawn_mode,
        food_respawn_mode=food_respawn_mode,
        food_respawn_interval_steps=food_respawn_interval_steps,
        food_respawn_batch_ratio=food_respawn_batch_ratio,
        food_respawn_cycle_steps=food_respawn_cycle_steps,
        serve_best_clones=serve_best_clones,
        interference_mode=interference_mode,
        blocker_ttl_steps=blocker_ttl_steps,
        blocker_energy_cost=blocker_energy_cost,
    )


def evolve(
    render: bool = False,
    renderer_mode: str = "none",
    fps: float = 20.0,
    render_step_skip: int = 2,
    render_every_generation: int = 1,
    generations: int | None = None,
    seed: int | None = None,
    fullscreen: bool = False,
    early_stop_mode: str | None = None,
    log_interval: int = 1,
    showcase_interval_generations: int = SHOWCASE_INTERVAL_GENERATIONS,
    food_coverage_init: float = 1.0,
    spawn_mode: str = DEFAULT_SPAWN_MODE,
    steps_growth_final: int | None = DEFAULT_STEPS_GROWTH_FINAL,
    steps_growth_generations: int = DEFAULT_STEPS_GROWTH_GENERATIONS,
    food_respawn_mode: str = DEFAULT_FOOD_RESPAWN_MODE,
    food_respawn_interval_steps: int = FOOD_RESPAWN_INTERVAL_STEPS,
    food_respawn_batch_ratio: float = FOOD_RESPAWN_BATCH_RATIO,
    food_respawn_cycle_steps: int = FOOD_RESPAWN_CYCLE_STEPS,
    interference_mode: str = DEFAULT_INTERFERENCE_MODE,
    blocker_ttl_steps: int = BLOCKER_TTL_STEPS,
    blocker_energy_cost: float = BLOCKER_ENERGY_COST,
) -> None:
    _sync_runtime_constants()
    core.evolve(
        render=render,
        renderer_mode=renderer_mode,
        fps=fps,
        render_step_skip=render_step_skip,
        render_every_generation=render_every_generation,
        generations=generations,
        seed=seed,
        fullscreen=fullscreen,
        early_stop_mode=early_stop_mode,
        log_interval=log_interval,
        showcase_interval_generations=showcase_interval_generations,
        food_coverage_init=food_coverage_init,
        spawn_mode=spawn_mode,
        steps_growth_final=steps_growth_final,
        steps_growth_generations=steps_growth_generations,
        food_respawn_mode=food_respawn_mode,
        food_respawn_interval_steps=food_respawn_interval_steps,
        food_respawn_batch_ratio=food_respawn_batch_ratio,
        food_respawn_cycle_steps=food_respawn_cycle_steps,
        interference_mode=interference_mode,
        blocker_ttl_steps=blocker_ttl_steps,
        blocker_energy_cost=blocker_energy_cost,
    )


def parse_args():
    _sync_runtime_constants()
    return core.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selected_renderer = args.renderer
    if args.pygame:
        selected_renderer = "pygame"

    if args.interactive_start:
        if selected_renderer == "none":
            selected_renderer = "pygame"
        interactive_train_and_serve(
            checkpoint_file=args.snapshot_file,
            renderer_mode=selected_renderer,
            fps=args.fps,
            render_step_skip=args.render_step_skip,
            fullscreen=args.fullscreen,
            seed=args.seed,
            log_interval=max(100, args.log_interval),
        )
        raise SystemExit(0)
    if args.train_snapshots:
        checkpoints = parse_checkpoint_generations(args.snapshot_generations)
        train_and_save_snapshots(
            generations=max(1, args.train_generations),
            checkpoints=checkpoints,
            checkpoint_file=args.snapshot_file,
            top_k=max(1, args.snapshot_top_k),
            seed=args.seed,
            log_interval=max(100, args.log_interval),
            food_coverage_init=args.food_coverage_init,
            spawn_mode=args.spawn_mode,
            steps_growth_final=args.steps_growth_final,
            steps_growth_generations=args.steps_growth_generations,
            food_respawn_mode=args.food_respawn_mode,
            food_respawn_interval_steps=args.food_respawn_interval_steps,
            food_respawn_batch_ratio=args.food_respawn_batch_ratio,
            food_respawn_cycle_steps=args.food_respawn_cycle_steps,
            interference_mode=args.interference_mode,
            blocker_ttl_steps=args.blocker_ttl_steps,
            blocker_energy_cost=args.blocker_energy_cost,
        )
    elif args.serve_snapshots:
        if selected_renderer == "none":
            selected_renderer = "pygame"
        serve_saved_snapshots(
            checkpoint_file=args.snapshot_file,
            renderer_mode=selected_renderer,
            fps=args.fps,
            render_step_skip=args.render_step_skip,
            fullscreen=args.fullscreen,
            food_coverage_init=args.food_coverage_init,
            spawn_mode=args.spawn_mode,
            food_respawn_mode=args.food_respawn_mode,
            food_respawn_interval_steps=args.food_respawn_interval_steps,
            food_respawn_batch_ratio=args.food_respawn_batch_ratio,
            food_respawn_cycle_steps=args.food_respawn_cycle_steps,
            serve_best_clones=args.serve_best_clones,
            interference_mode=args.interference_mode,
            blocker_ttl_steps=args.blocker_ttl_steps,
            blocker_energy_cost=args.blocker_energy_cost,
        )
    else:
        evolve(
            render=args.render,
            renderer_mode=selected_renderer,
            fps=args.fps,
            render_step_skip=args.render_step_skip,
            render_every_generation=args.render_every_generation,
            generations=args.generations,
            seed=args.seed,
            fullscreen=args.fullscreen,
            early_stop_mode=args.early_stop,
            log_interval=args.log_interval,
            showcase_interval_generations=args.showcase_interval_generations,
            food_coverage_init=args.food_coverage_init,
            spawn_mode=args.spawn_mode,
            steps_growth_final=args.steps_growth_final,
            steps_growth_generations=args.steps_growth_generations,
            food_respawn_mode=args.food_respawn_mode,
            food_respawn_interval_steps=args.food_respawn_interval_steps,
            food_respawn_batch_ratio=args.food_respawn_batch_ratio,
            food_respawn_cycle_steps=args.food_respawn_cycle_steps,
            interference_mode=args.interference_mode,
            blocker_ttl_steps=args.blocker_ttl_steps,
            blocker_energy_cost=args.blocker_energy_cost,
        )
