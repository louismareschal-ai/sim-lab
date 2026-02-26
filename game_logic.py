from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from game_constants import (
    BENCHMARK_TRIALS,
    BLOCKER_ENERGY_COST,
    BLOCKER_TTL_STEPS,
    CHECKPOINT_GENERATIONS,
    DEFAULT_INTERFERENCE_MODE,
    DEFAULT_FOOD_RESPAWN_MODE,
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
from game_rendering import (
    PygameRenderer as _PygameRenderer,
    TerminalRenderer as _TerminalRenderer,
    advance_onboarding_state as _advance_onboarding_state,
    build_renderer as _build_renderer,
    checkpoint_requires_resume as _checkpoint_requires_resume,
    checkpoint_should_defer as _checkpoint_should_defer,
    playback_fps_for_steps as _playback_fps_for_steps,
)
import game_rendering as rendering_module

random.seed(SEED)

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        raise SystemExit(0)


def session_codename(seed_value: int | None) -> str:
    key = int(SEED if seed_value is None else seed_value)
    adjectives = [
        "Silent",
        "Emerald",
        "Neon",
        "Arc",
        "Obsidian",
        "Solar",
        "Crimson",
        "Aurora",
    ]
    nouns = [
        "Swarm",
        "Voyager",
        "Pulse",
        "Drift",
        "Labyrinth",
        "Beacon",
        "Frontier",
        "Nexus",
    ]
    adj = adjectives[abs(key) % len(adjectives)]
    noun = nouns[(abs(key) // len(adjectives)) % len(nouns)]
    return f"{adj} {noun}"


def print_run_header(mode_label: str, seed_value: int | None) -> None:
    codename = session_codename(seed_value)
    resolved_seed = SEED if seed_value is None else int(seed_value)
    safe_print("=" * 64)
    safe_print(f"{mode_label} | Session: {codename} | Seed: {resolved_seed}")
    safe_print("Objective: adapt, survive, and dominate the evolving world.")
    safe_print("=" * 64)


def build_renderer(renderer_mode: str, fps: float, step_skip: int, fullscreen: bool = False) -> TerminalRenderer | PygameRenderer | None:
    return _build_renderer(renderer_mode, fps=fps, step_skip=step_skip, fullscreen=fullscreen)


TerminalRenderer = _TerminalRenderer
PygameRenderer = _PygameRenderer


def classify_species(genes: list[float]) -> str:
    if genes[0] + genes[1] > 0.8:
        return "forager"
    if genes[9] + genes[14] > 0.8:
        return "sprinter"
    if genes[2] + genes[7] + genes[12] + genes[17] > 0.5:
        return "tank"
    return "hunter"


def spawn_obstacles(count: int) -> set[tuple[int, int]]:
    obstacles = set()
    while len(obstacles) < count:
        obstacles.add((random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)))
    return obstacles


def spawn_predators(count: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    predators = []
    while len(predators) < count:
        pos = (random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE))
        if pos not in blocked:
            predators.append(pos)
    return predators


def step_towards(src: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    sx, sy = src
    tx, ty = target
    dx = 0 if tx == sx else (1 if tx > sx else -1)
    dy = 0 if ty == sy else (1 if ty > sy else -1)
    if random.random() < 0.5:
        return dx, 0
    return 0, dy


@dataclass
class Creature:
    genes: list[float]
    x: int
    y: int
    agent_id: int = 0
    energy: float = 5.0
    food_eaten: int = 0
    survival_steps: int = 0
    species: str = "forager"

    def decide_move(self, nearest_food_dx: float, nearest_food_dy: float, wall_x: float, wall_y: float) -> tuple[int, int]:
        w = self.genes

        left = (
            w[0] * nearest_food_dx
            + w[1] * nearest_food_dy
            + w[2] * wall_x
            + w[3] * wall_y
            + w[4]
        )
        right = (
            w[5] * nearest_food_dx
            + w[6] * nearest_food_dy
            + w[7] * wall_x
            + w[8] * wall_y
            + w[9]
        )
        up = (
            w[10] * nearest_food_dx
            + w[11] * nearest_food_dy
            + w[12] * wall_x
            + w[13] * wall_y
            + w[14]
        )
        down = (
            w[15] * nearest_food_dx
            + w[16] * nearest_food_dy
            + w[17] * wall_x
            + w[18] * wall_y
            + w[19]
        )

        scores = [left, right, up, down]
        action = max(range(4), key=lambda i: scores[i])
        if action == 0:
            return -1, 0
        if action == 1:
            return 1, 0
        if action == 2:
            return 0, -1
        return 0, 1


def random_genes() -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(20)]


def spawn_food(count: int) -> set[tuple[int, int]]:
    food = set()
    while len(food) < count:
        food.add((random.randrange(WORLD_SIZE), random.randrange(WORLD_SIZE)))
    return food


def spawn_positions_for_population(population_size: int, spawn_mode: str) -> list[tuple[int, int]]:
    sample_size = min(population_size, WORLD_SIZE * WORLD_SIZE)
    all_cells = [(x, y) for x in range(WORLD_SIZE) for y in range(WORLD_SIZE)]
    corners = [(0, 0), (WORLD_SIZE - 1, 0), (0, WORLD_SIZE - 1), (WORLD_SIZE - 1, WORLD_SIZE - 1)]

    def corner_distance(cell: tuple[int, int]) -> int:
        return min(abs(cell[0] - cx) + abs(cell[1] - cy) for cx, cy in corners)

    if spawn_mode == "corners":
        ordered = sorted(all_cells, key=lambda cell: (corner_distance(cell), random.random()))
        selected = ordered[:sample_size]
        random.shuffle(selected)
        return selected

    if spawn_mode == "biased-corners":
        sampled = random.sample(all_cells, k=sample_size)
        selected: list[tuple[int, int]] = []
        for cell in sampled:
            bias_chance = max(0.20, 0.85 - 0.12 * corner_distance(cell))
            if random.random() < bias_chance:
                selected.append(cell)

        if len(selected) < sample_size:
            ordered = sorted(all_cells, key=lambda cell: (corner_distance(cell), random.random()))
            selected_set = set(selected)
            for cell in ordered:
                if cell not in selected_set:
                    selected.append(cell)
                    selected_set.add(cell)
                if len(selected) >= sample_size:
                    break
        random.shuffle(selected)
        return selected[:sample_size]

    return random.sample(all_cells, k=sample_size)


def estimate_generation_steps(
    generation: int,
    base_steps: int,
    final_steps: int | None,
    ramp_generations: int,
) -> int:
    base = max(1, int(base_steps))
    if final_steps is None:
        return base

    target = max(1, int(final_steps))
    if target <= base or ramp_generations <= 1:
        return target if generation > 1 else base

    progress = min(1.0, max(0.0, (generation - 1) / (ramp_generations - 1)))
    interpolated = base + (target - base) * progress
    return max(1, int(round(interpolated)))


def random_free_cell(blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
    all_cells = [(x, y) for x in range(WORLD_SIZE) for y in range(WORLD_SIZE) if (x, y) not in blocked]
    if not all_cells:
        return None
    return random.choice(all_cells)


def normalize_food_coverage(value: float) -> float:
    coverage = float(value)
    if coverage > 1.0:
        coverage = coverage / 100.0
    return max(0.01, min(1.0, coverage))


def spawn_food_with_coverage(coverage: float) -> set[tuple[int, int]]:
    normalized = normalize_food_coverage(coverage)
    total_cells = WORLD_SIZE * WORLD_SIZE
    target_food = max(1, int(round(total_cells * normalized)))
    all_cells = [(x, y) for x in range(WORLD_SIZE) for y in range(WORLD_SIZE)]
    return set(random.sample(all_cells, k=min(target_food, total_cells)))


def quadrant_for_step(step: int, cycle_steps: int) -> str:
    effective_cycle_steps = max(1, int(cycle_steps))
    phase = (max(0, int(step)) // effective_cycle_steps) % 4
    order = ["bottom-right", "bottom-left", "top-left", "top-right"]
    return order[phase]


def food_spawn_candidates(respawn_mode: str, step: int, cycle_steps: int) -> list[tuple[int, int]]:
    if respawn_mode == "right-side":
        return [
            (x, y)
            for x in range(max(0, WORLD_SIZE // 2), WORLD_SIZE)
            for y in range(WORLD_SIZE)
        ]

    if respawn_mode == "clockwise-quadrants":
        half_x = WORLD_SIZE // 2
        half_y = WORLD_SIZE // 2
        quadrant = quadrant_for_step(step, cycle_steps)
        if quadrant == "bottom-right":
            return [(x, y) for x in range(half_x, WORLD_SIZE) for y in range(half_y, WORLD_SIZE)]
        if quadrant == "bottom-left":
            return [(x, y) for x in range(0, half_x) for y in range(half_y, WORLD_SIZE)]
        if quadrant == "top-left":
            return [(x, y) for x in range(0, half_x) for y in range(0, half_y)]
        return [(x, y) for x in range(half_x, WORLD_SIZE) for y in range(0, half_y)]

    return [(x, y) for x in range(WORLD_SIZE) for y in range(WORLD_SIZE)]


def replenish_food(
    food: set[tuple[int, int]],
    target_count: int,
    blocked: set[tuple[int, int]] | None = None,
    respawn_mode: str = "random",
    batch_count: int | None = None,
    step: int = 0,
    cycle_steps: int = FOOD_RESPAWN_CYCLE_STEPS,
) -> None:
    blocked_positions = blocked or set()
    candidate_pool = food_spawn_candidates(respawn_mode, step=step, cycle_steps=cycle_steps)

    if not candidate_pool:
        return

    free_capacity = max(0, target_count - len(food))
    if free_capacity <= 0:
        return

    additions_target = free_capacity if batch_count is None else min(free_capacity, max(1, int(batch_count)))
    max_attempts = max(1, target_count * 20)
    attempts = 0
    additions = 0
    while additions < additions_target and attempts < max_attempts:
        candidate = random.choice(candidate_pool)
        if candidate not in blocked_positions:
            if candidate not in food:
                food.add(candidate)
                additions += 1
        attempts += 1


def metabolic_drain_for_step(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.2
    progress = step / (total_steps - 1)
    return 0.12 + 0.58 * (progress ** 1.35)


def creature_fitness(creature: "Creature") -> float:
    primary_food = min(creature.food_eaten, FOOD_PRIMARY_CAP)
    secondary_food = max(0, creature.food_eaten - FOOD_PRIMARY_CAP)
    food_reward = primary_food * FOOD_REWARD_PRIMARY + secondary_food * FOOD_REWARD_SECONDARY
    survival_reward = creature.survival_steps * SURVIVAL_REWARD_PER_STEP
    energy_reward = max(0.0, creature.energy) * ENERGY_REWARD_SCALE
    return survival_reward + food_reward + energy_reward


def nearest_food_vector(x: int, y: int, food: set[tuple[int, int]]) -> tuple[float, float]:
    if not food:
        return 0.0, 0.0

    nearest = min(food, key=lambda f: (f[0] - x) ** 2 + (f[1] - y) ** 2)
    dx = nearest[0] - x
    dy = nearest[1] - y
    length = math.sqrt(dx * dx + dy * dy) + 1e-9
    return dx / length, dy / length


def wall_features(x: int, y: int) -> tuple[float, float]:
    center = (WORLD_SIZE - 1) / 2
    fx = (x - center) / center
    fy = (y - center) / center
    return fx, fy


def blocker_probability_from_genes(genes: list[float]) -> float:
    if len(genes) < 20:
        return 0.0
    signal = max(-1.0, min(1.0, float(genes[19])))
    return 0.05 + ((signal + 1.0) * 0.5) * 0.35


def choose_blocker_target_cell(
    actor: "Creature",
    rivals: list["Creature"],
    food: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    if not rivals:
        return None
    rival = min(rivals, key=lambda creature: (creature.x - actor.x) ** 2 + (creature.y - actor.y) ** 2)
    nfdx, nfdy = nearest_food_vector(rival.x, rival.y, food)
    if abs(nfdx) >= abs(nfdy):
        dx = 0 if abs(nfdx) < 1e-9 else (1 if nfdx > 0 else -1)
        dy = 0
    else:
        dx = 0
        dy = 0 if abs(nfdy) < 1e-9 else (1 if nfdy > 0 else -1)

    target = (rival.x + dx, rival.y + dy)
    tx, ty = target
    if tx < 0 or tx >= WORLD_SIZE or ty < 0 or ty >= WORLD_SIZE:
        return None
    if target in blocked:
        return None
    if target in food:
        return None
    return target


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
    callback_accepts_info = False
    if step_callback is not None:
        callback_accepts_info = len(inspect.signature(step_callback).parameters) >= 4

    effective_spawn_mode = spawn_mode if spawn_mode in {"random", "corners", "biased-corners"} else DEFAULT_SPAWN_MODE
    spawn_positions = spawn_positions_for_population(len(population_genes), effective_spawn_mode)
    creatures = [
        Creature(
            genes=g[:],
            x=spawn_positions[index][0],
            y=spawn_positions[index][1],
            agent_id=index,
            species=classify_species(g),
        )
        for index, g in enumerate(population_genes)
    ]
    food = spawn_food_with_coverage(food_coverage_init)
    food_target_count = len(food)
    obstacles: set[tuple[int, int]] = set()
    dynamic_blockers: dict[tuple[int, int], int] = {}
    predators: list[tuple[int, int]] = []
    meat: set[tuple[int, int]] = set()
    event_counts = {
        "plant_eats": 0,
        "meat_eats": 0,
        "predator_kills": 0,
        "cannibal_kills": 0,
        "blocks_placed": 0,
    }

    total_steps = max(1, int(steps_override)) if steps_override is not None else STEPS_PER_GENERATION
    effective_respawn_mode = (
        food_respawn_mode if food_respawn_mode in {"off", "random", "right-side", "clockwise-quadrants"} else DEFAULT_FOOD_RESPAWN_MODE
    )
    effective_respawn_interval = max(1, int(food_respawn_interval_steps))
    effective_respawn_batch_ratio = max(0.01, min(1.0, float(food_respawn_batch_ratio)))
    effective_respawn_cycle_steps = max(1, int(food_respawn_cycle_steps))
    effective_interference_mode = interference_mode if interference_mode in {"off", "simple-block"} else DEFAULT_INTERFERENCE_MODE
    effective_blocker_ttl = max(1, int(blocker_ttl_steps))
    effective_blocker_energy_cost = max(0.0, float(blocker_energy_cost))
    last_active_quadrant = ""
    clockwise_respawn_tick = 0

    if effective_respawn_mode in {"right-side", "clockwise-quadrants"} and food_target_count > 0:
        food = set()
        initial_quadrant_step = 0
        if effective_respawn_mode == "clockwise-quadrants":
            last_active_quadrant = quadrant_for_step(initial_quadrant_step, effective_respawn_cycle_steps)
        replenish_food(
            food,
            target_count=food_target_count,
            blocked=obstacles,
            respawn_mode=effective_respawn_mode,
            batch_count=food_target_count,
            step=initial_quadrant_step,
            cycle_steps=effective_respawn_cycle_steps,
        )
        if effective_respawn_mode == "clockwise-quadrants":
            clockwise_respawn_tick = 1

    for step in range(total_steps):
        if dynamic_blockers:
            expired = []
            for cell, ttl in dynamic_blockers.items():
                next_ttl = ttl - 1
                if next_ttl <= 0:
                    expired.append(cell)
                else:
                    dynamic_blockers[cell] = next_ttl
            for cell in expired:
                dynamic_blockers.pop(cell, None)

        alive_creatures = [creature for creature in creatures if creature.energy > 0]
        if not alive_creatures:
            break

        obstacles = set(dynamic_blockers.keys())

        if effective_respawn_mode != "off" and (not food or step % effective_respawn_interval == 0):
            blocked_positions = obstacles | {(creature.x, creature.y) for creature in alive_creatures}
            batch_count = max(1, int(round(food_target_count * effective_respawn_batch_ratio)))
            respawn_reference = step
            if effective_respawn_mode == "clockwise-quadrants":
                respawn_reference = clockwise_respawn_tick
                active_quadrant = quadrant_for_step(respawn_reference, effective_respawn_cycle_steps)
                if active_quadrant != last_active_quadrant:
                    food.clear()
                    last_active_quadrant = active_quadrant
            replenish_food(
                food,
                target_count=food_target_count,
                blocked=blocked_positions,
                respawn_mode=effective_respawn_mode,
                batch_count=batch_count,
                step=respawn_reference,
                cycle_steps=effective_respawn_cycle_steps,
            )
            if effective_respawn_mode == "clockwise-quadrants":
                clockwise_respawn_tick += 1

        intents: dict[int, tuple[int, int]] = {}
        for creature in alive_creatures:
            nfdx, nfdy = nearest_food_vector(creature.x, creature.y, food)
            wx, wy = wall_features(creature.x, creature.y)
            mx, my = creature.decide_move(nfdx, nfdy, wx, wy)

            nx = max(0, min(WORLD_SIZE - 1, creature.x + mx))
            ny = max(0, min(WORLD_SIZE - 1, creature.y + my))
            if (nx, ny) in obstacles:
                nx, ny = creature.x, creature.y
            intents[creature.agent_id] = (nx, ny)

        contenders: defaultdict[tuple[int, int], list[Creature]] = defaultdict(list)
        for creature in alive_creatures:
            contenders[intents[creature.agent_id]].append(creature)

        winners_by_target: dict[tuple[int, int], int] = {}
        for target, candidates in contenders.items():
            max_energy = max(candidate.energy for candidate in candidates)
            best = [candidate for candidate in candidates if abs(candidate.energy - max_energy) < 1e-9]
            winner = random.choice(best)
            winners_by_target[target] = winner.agent_id

        reservation_order = sorted(alive_creatures, key=lambda creature: (creature.energy, random.random()), reverse=True)
        reserved: set[tuple[int, int]] = set()
        final_positions: dict[int, tuple[int, int]] = {}

        for creature in reservation_order:
            wanted = intents[creature.agent_id]
            if winners_by_target.get(wanted) == creature.agent_id and wanted not in reserved:
                final = wanted
            elif (creature.x, creature.y) not in reserved:
                final = (creature.x, creature.y)
            else:
                free_cell = random_free_cell(reserved | obstacles)
                final = free_cell if free_cell is not None else (creature.x, creature.y)
            final_positions[creature.agent_id] = final
            reserved.add(final)

        metabolic_drain = metabolic_drain_for_step(step, total_steps)
        for creature in alive_creatures:
            creature.x, creature.y = final_positions[creature.agent_id]
            creature.energy -= metabolic_drain

        occupied_positions = {(creature.x, creature.y) for creature in alive_creatures}
        for creature in alive_creatures:
            if creature.energy <= 0:
                continue

            if (creature.x, creature.y) in food:
                food.remove((creature.x, creature.y))
                creature.food_eaten += 1
                creature.energy += 1.0
                event_counts["plant_eats"] += 1

            if (creature.x, creature.y) in meat:
                meat.remove((creature.x, creature.y))
                creature.food_eaten += 1
                creature.energy += 1.0
                event_counts["meat_eats"] += 1

            if creature.energy > 0:
                creature.survival_steps += 1

        if effective_interference_mode == "simple-block":
            blocked_cells = set(obstacles) | occupied_positions
            for creature in alive_creatures:
                if creature.energy <= effective_blocker_energy_cost:
                    continue
                if random.random() >= blocker_probability_from_genes(creature.genes):
                    continue
                rivals = [other for other in alive_creatures if other.agent_id != creature.agent_id and other.energy > 0]
                target = choose_blocker_target_cell(creature, rivals, food, blocked_cells)
                if target is None:
                    continue
                dynamic_blockers[target] = effective_blocker_ttl
                blocked_cells.add(target)
                creature.energy -= effective_blocker_energy_cost
                event_counts["blocks_placed"] += 1

        if False:
            positions: defaultdict[tuple[int, int], list[Creature]] = defaultdict(list)
            for creature in creatures:
                if creature.energy > 0:
                    positions[(creature.x, creature.y)].append(creature)

            for pos, same_cell_creatures in positions.items():
                if len(same_cell_creatures) < 2:
                    continue
                same_cell_creatures.sort(key=lambda c: c.energy, reverse=True)
                hunter = same_cell_creatures[0]
                victim = same_cell_creatures[-1]
                if hunter is victim or victim.energy <= 0:
                    continue
                if hunter.energy - victim.energy > 2.0 and random.random() < 0.35:
                    victim.energy = 0.0
                    hunter.energy += 4.0
                    meat.add(pos)
                    event_counts["cannibal_kills"] += 1

        if False:
            alive_creatures = [creature for creature in creatures if creature.energy > 0]
            new_predators = []
            for predator in predators:
                px, py = predator
                if alive_creatures:
                    target = min(alive_creatures, key=lambda c: (c.x - px) ** 2 + (c.y - py) ** 2)
                    mx, my = step_towards((px, py), (target.x, target.y))
                    nx = max(0, min(WORLD_SIZE - 1, px + mx))
                    ny = max(0, min(WORLD_SIZE - 1, py + my))
                    if (nx, ny) in obstacles:
                        nx, ny = px, py
                else:
                    nx, ny = px, py

                new_predators.append((nx, ny))

                for creature in creatures:
                    if creature.energy > 0 and creature.x == nx and creature.y == ny:
                        creature.energy = 0.0
                        meat.add((nx, ny))
                        event_counts["predator_kills"] += 1
            predators = new_predators

        if step_callback is not None:
            alive_count = sum(1 for creature in creatures if creature.energy > 0)
            step_info = {
                "level": level,
                "obstacles": obstacles,
                "predators": predators,
                "meat": meat,
                "event_counts": event_counts,
                "species_counts": Counter(c.species for c in creatures if c.energy > 0),
                "food_remaining": len(food),
                "food_target": food_target_count,
                "alive_agents": alive_count,
                "spawn_mode": effective_spawn_mode,
                "steps_per_generation": total_steps,
                "food_respawn_mode": effective_respawn_mode,
                "food_respawn_quadrant": (
                    last_active_quadrant
                    if effective_respawn_mode == "clockwise-quadrants"
                    else ""
                ),
                "interference_mode": effective_interference_mode,
            }
            if callback_accepts_info:
                step_callback(step, creatures, food, step_info)
            else:
                step_callback(step, creatures, food)

    scored = []
    for c in creatures:
        fitness = creature_fitness(c)
        scored.append((fitness, c.genes, c.food_eaten))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def crossover(parent_a: list[float], parent_b: list[float]) -> list[float]:
    child = []
    for a, b in zip(parent_a, parent_b):
        child.append(a if random.random() < 0.5 else b)
    return child


def mutate(genes: list[float]) -> list[float]:
    new_genes = genes[:]
    for i in range(len(new_genes)):
        if random.random() < MUTATION_RATE:
            new_genes[i] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    return new_genes


def mutate_with_parameters(genes: list[float], mutation_rate: float, mutation_strength: float) -> list[float]:
    new_genes = genes[:]
    for i in range(len(new_genes)):
        if random.random() < mutation_rate:
            new_genes[i] += random.uniform(-mutation_strength, mutation_strength)
    return new_genes


def mutation_schedule(generation: int) -> tuple[float, float]:
    if generation <= 20:
        return 0.18, 0.30
    if generation <= 80:
        return 0.12, 0.22
    if generation <= 160:
        return 0.09, 0.16
    return 0.06, 0.12


def should_rollback_to_best(avg_history: list[float], best_avg_seen: float) -> bool:
    if len(avg_history) < 15:
        return False
    recent_avg = sum(avg_history[-15:]) / 15.0
    return recent_avg < best_avg_seen - 4.0


def level_for_generation(generation: int) -> int:
    if generation <= 20:
        return 1
    if generation <= 120:
        return 2
    return 3


def decide_move_from_genes(
    genes: list[float],
    nearest_food_dx: float,
    nearest_food_dy: float,
    wall_x: float,
    wall_y: float,
) -> tuple[int, int]:
    left = (
        genes[0] * nearest_food_dx
        + genes[1] * nearest_food_dy
        + genes[2] * wall_x
        + genes[3] * wall_y
        + genes[4]
    )
    right = (
        genes[5] * nearest_food_dx
        + genes[6] * nearest_food_dy
        + genes[7] * wall_x
        + genes[8] * wall_y
        + genes[9]
    )
    up = (
        genes[10] * nearest_food_dx
        + genes[11] * nearest_food_dy
        + genes[12] * wall_x
        + genes[13] * wall_y
        + genes[14]
    )
    down = (
        genes[15] * nearest_food_dx
        + genes[16] * nearest_food_dy
        + genes[17] * wall_x
        + genes[18] * wall_y
        + genes[19]
    )
    scores = [left, right, up, down]
    action = max(range(4), key=lambda i: scores[i])
    if action == 0:
        return -1, 0
    if action == 1:
        return 1, 0
    if action == 2:
        return 0, -1
    return 0, 1


def simulate_task_once(genes: list[float], task_name: str, run_seed: int) -> float:
    rng = random.Random(run_seed)
    local_world = 16
    steps = 90
    x, y = local_world // 2, local_world // 2
    energy = 20.0
    food_eaten = 0
    survival_steps = 0

    food: set[tuple[int, int]] = set()
    obstacles: set[tuple[int, int]] = set()
    predator = (rng.randrange(local_world), rng.randrange(local_world))

    if task_name == "forage_basic":
        while len(food) < 25:
            food.add((rng.randrange(local_world), rng.randrange(local_world)))
    elif task_name == "forage_maze":
        while len(food) < 25:
            food.add((rng.randrange(local_world), rng.randrange(local_world)))
        while len(obstacles) < 30:
            candidate = (rng.randrange(local_world), rng.randrange(local_world))
            if candidate != (x, y):
                obstacles.add(candidate)
    else:
        while len(food) < 20:
            food.add((rng.randrange(local_world), rng.randrange(local_world)))

    for step in range(steps):
        if energy <= 0:
            break

        if food:
            nearest = min(food, key=lambda f: (f[0] - x) ** 2 + (f[1] - y) ** 2)
            dx = nearest[0] - x
            dy = nearest[1] - y
            norm = math.sqrt(dx * dx + dy * dy) + 1e-9
            nfdx = dx / norm
            nfdy = dy / norm
        else:
            nfdx = 0.0
            nfdy = 0.0
        center = (local_world - 1) / 2
        wx = (x - center) / center
        wy = (y - center) / center

        mx, my = decide_move_from_genes(genes, nfdx, nfdy, wx, wy)
        nx = max(0, min(local_world - 1, x + mx))
        ny = max(0, min(local_world - 1, y + my))
        if (nx, ny) in obstacles:
            nx, ny = x, y
        x, y = nx, ny
        energy -= 1.0

        if (x, y) in food:
            food.remove((x, y))
            food_eaten += 1
            energy += 10.0

        if task_name == "predator_escape":
            pdx, pdy = step_towards(predator, (x, y))
            predator = (
                max(0, min(local_world - 1, predator[0] + pdx)),
                max(0, min(local_world - 1, predator[1] + pdy)),
            )
            if predator == (x, y):
                energy -= 8.0

        if energy > 0:
            survival_steps += 1

    primary_food = min(food_eaten, FOOD_PRIMARY_CAP)
    secondary_food = max(0, food_eaten - FOOD_PRIMARY_CAP)
    fitness = (
        survival_steps * SURVIVAL_REWARD_PER_STEP
        + primary_food * FOOD_REWARD_PRIMARY
        + secondary_food * FOOD_REWARD_SECONDARY
        + max(0.0, energy) * ENERGY_REWARD_SCALE
    )
    if task_name == "predator_escape":
        fitness += 4.0 if energy > 0 else -6.0
    return fitness


def benchmark_trial_worker(args: tuple[list[float], str, int]) -> tuple[str, float]:
    genes, task_name, trial_seed = args
    return task_name, simulate_task_once(genes, task_name, trial_seed)


def evaluate_checkpoint_benchmark(
    genes: list[float],
    checkpoint_generation: int,
    trials: int = BENCHMARK_TRIALS,
    use_parallel: bool = True,
) -> dict[str, float]:
    task_names = ["forage_basic", "forage_maze", "predator_escape"]
    jobs: list[tuple[list[float], str, int]] = []
    for task_name in task_names:
        for trial_index in range(trials):
            trial_seed = SEED + checkpoint_generation * 997 + trial_index * 101 + len(task_name)
            jobs.append((genes, task_name, trial_seed))

    scores: defaultdict[str, list[float]] = defaultdict(list)

    if use_parallel and len(jobs) > 4:
        max_workers = max(1, min((os.cpu_count() or 2), 6))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for task_name, value in executor.map(benchmark_trial_worker, jobs):
                    scores[task_name].append(value)
        except Exception:
            for job in jobs:
                task_name, value = benchmark_trial_worker(job)
                scores[task_name].append(value)
    else:
        for job in jobs:
            task_name, value = benchmark_trial_worker(job)
            scores[task_name].append(value)

    result = {
        "forage_basic": sum(scores["forage_basic"]) / max(1, len(scores["forage_basic"])),
        "forage_maze": sum(scores["forage_maze"]) / max(1, len(scores["forage_maze"])),
        "predator_escape": sum(scores["predator_escape"]) / max(1, len(scores["predator_escape"])),
    }
    result["overall"] = (result["forage_basic"] + result["forage_maze"] + result["predator_escape"]) / 3.0
    return result


def run_checkpoint_benchmark_job(genes: list[float], generation: int) -> tuple[int, dict[str, float]]:
    return generation, evaluate_checkpoint_benchmark(genes, generation, use_parallel=False)


def select_initial_champion(population_genes: list[list[float]]) -> list[float]:
    return max(
        population_genes,
        key=lambda genes: simulate_task_once(genes, "forage_basic", SEED + 71),
    )


advance_onboarding_state = _advance_onboarding_state
checkpoint_requires_resume = _checkpoint_requires_resume
checkpoint_should_defer = _checkpoint_should_defer


def should_stop_early(
    best_history: list[float],
    avg_history: list[float],
    min_generations: int = 25,
    patience: int = 18,
    plateau_delta: float = 1.5,
    noise_std_threshold: float = 6.0,
    trend_delta: float = 0.8,
) -> tuple[bool, str]:
    if len(best_history) < max(min_generations, patience):
        return False, ""

    best_window = best_history[-patience:]
    avg_window = avg_history[-patience:]
    best_range = max(best_window) - min(best_window)
    best_trend = best_window[-1] - best_window[0]
    avg_trend = avg_window[-1] - avg_window[0]

    if best_range < plateau_delta:
        return True, "plateau"

    if (
        abs(best_trend) < trend_delta
        and abs(avg_trend) < trend_delta
        and statistics.pstdev(best_window) > noise_std_threshold
    ):
        return True, "noisy"

    return False, ""


def resolve_early_stop_mode(renderer_mode: str, requested_mode: str | None) -> str:
    if requested_mode in {"off", "warn", "auto"}:
        return requested_mode

    if renderer_mode == "pygame":
        return "warn"
    return "auto"


playback_fps_for_steps = _playback_fps_for_steps


def should_log_generation(generation: int, total_generations: int | None, log_interval: int) -> bool:
    if log_interval <= 1:
        return True
    if generation == 1:
        return True
    if total_generations is not None and generation == total_generations:
        return True
    return generation % log_interval == 0


def should_showcase_generation(generation: int, interval_generations: int) -> bool:
    if generation == 1:
        return True
    if interval_generations <= 1:
        return True
    return generation % interval_generations == 0


def parse_checkpoint_generations(raw: str) -> list[int]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    values = sorted({int(item) for item in parts if int(item) > 0})
    if not values:
        raise ValueError("At least one positive checkpoint generation is required.")
    return values


def compute_progress_signal(best_history: list[float], avg_history: list[float], window: int = 20) -> dict[str, float | bool]:
    if len(best_history) < max(3, window):
        return {
            "ready": False,
            "best_delta": 0.0,
            "avg_delta": 0.0,
            "stability": 0.0,
            "score": 0.0,
        }

    best_window = best_history[-window:]
    avg_window = avg_history[-window:]
    best_delta = best_window[-1] - best_window[0]
    avg_delta = avg_window[-1] - avg_window[0]
    stability = statistics.pstdev(avg_window)
    score = avg_window[-1] + 0.4 * best_window[-1] - 0.3 * stability
    return {
        "ready": True,
        "best_delta": best_delta,
        "avg_delta": avg_delta,
        "stability": stability,
        "score": score,
    }


def adapt_curriculum_level(current_level: int, progress: dict[str, float | bool]) -> int:
    if not bool(progress.get("ready", False)):
        return current_level

    avg_delta = float(progress["avg_delta"])
    best_delta = float(progress["best_delta"])
    stability = float(progress["stability"])

    should_promote = avg_delta > 4.0 and best_delta > 5.0 and stability < 10.0
    should_relax = avg_delta < 1.0 and best_delta < 2.0 and stability > 12.0

    if should_promote and current_level < 3:
        return current_level + 1
    if should_relax and current_level > 1:
        return current_level - 1
    return current_level


def save_snapshot_file(
    file_path: str,
    snapshots: dict[int, dict[str, object]],
    total_generations: int,
    seed: int,
    training_curve: dict[str, list[float | int]] | None = None,
) -> None:
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "total_generations": total_generations,
        "saved_at": int(time.time()),
        "training_curve": training_curve or {
            "generations": [],
            "avg_fitness": [],
            "max_fitness": [],
            "min_fitness": [],
        },
        "snapshots": {
            str(generation): snapshot for generation, snapshot in sorted(snapshots.items())
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_snapshot_file(file_path: str) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    snapshots_raw = payload.get("snapshots", {})
    snapshots: dict[int, dict[str, object]] = {}
    for generation_key, snapshot in snapshots_raw.items():
        generation = int(generation_key)
        if isinstance(snapshot, dict):
            snapshots[generation] = snapshot
    metadata = {
        "seed": payload.get("seed"),
        "total_generations": payload.get("total_generations"),
        "saved_at": payload.get("saved_at"),
        "training_curve": payload.get("training_curve", {}),
    }
    return snapshots, metadata


def select_default_snapshot_generation(snapshots: dict[int, dict[str, object]]) -> int:
    if not snapshots:
        raise ValueError("No snapshots available")
    return max(snapshots.keys())


def build_serve_population(
    snapshot: dict[str, object],
    target_population_size: int,
    serve_best_clones: int,
) -> list[list[float]]:
    top_genes_raw = snapshot.get("top_genes", [])
    if not isinstance(top_genes_raw, list) or not top_genes_raw:
        raise SystemExit("Snapshot has no usable top_genes.")

    top_genes: list[list[float]] = [
        [float(value) for value in genes]
        for genes in top_genes_raw
        if isinstance(genes, list) and len(genes) == 20
    ]
    if not top_genes:
        raise SystemExit("Snapshot has invalid gene vectors.")

    if serve_best_clones > 0:
        best_raw = snapshot.get("best_genes", top_genes[0])
        if isinstance(best_raw, list) and len(best_raw) == 20:
            best_genes = [float(value) for value in best_raw]
        else:
            best_genes = top_genes[0][:]
        clone_count = max(1, int(serve_best_clones))
        return [best_genes[:] for _ in range(clone_count)]

    target_size = max(1, int(target_population_size))
    population: list[list[float]] = [genes[:] for genes in top_genes]
    while len(population) < target_size:
        population.append(random.choice(top_genes)[:])
    return population[:target_size]


STALL_WINDOW = 300  # Stop training if best fitness hasn't improved in this many generations


def analyze_stall(
    best_fitness_history: list[float],
    avg_fitness_history: list[float],
    curriculum_level: int,
    stall_window: int,
) -> None:
    """Print a human-readable diagnosis when training stalls."""
    best_ever = max(best_fitness_history)
    last_best = best_fitness_history[-1]
    avg_last = avg_fitness_history[-1] if avg_fitness_history else 0.0
    avg_first = avg_fitness_history[0] if avg_fitness_history else 0.0
    total_avg_improvement = avg_last - avg_first
    likely_survival_ratio = max(0.0, min(1.0, best_ever / (STEPS_PER_GENERATION + 24.0)))

    safe_print("\n" + "=" * 60)
    safe_print("STALL DETECTED — Training analysis")
    safe_print("=" * 60)
    safe_print(f"  Best fitness ever reached : {best_ever:.2f}")
    safe_print(f"  Current best fitness      : {last_best:.2f}")
    safe_print(f"  Avg fitness (last gen)    : {avg_last:.2f}")
    safe_print(f"  Total avg improvement     : {total_avg_improvement:+.2f} over {len(avg_fitness_history)} gens")
    safe_print(f"  Curriculum level          : {curriculum_level} (1=basic, 2=obstacles, 3=battles)")
    safe_print(f"  Survival score share      : ~{likely_survival_ratio * 100:4.1f}%")
    safe_print("")
    safe_print("AI technique: genetic algorithm (GA) with linear policy genes.")
    safe_print("  Each creature has 20 weights mapping (food direction, wall pos) -> move direction.")
    safe_print("  No neural network layers or backpropagation — pure evolutionary search.")
    safe_print("")
    if best_ever < 30:
        safe_print("  Diagnosis: agents are barely eating. The population may be stuck in random walk.")
        safe_print("  Fix: increase POPULATION_SIZE or FOOD_COUNT, or flatten obstacles at level 1.")
    elif total_avg_improvement < 2.0:
        safe_print("  Diagnosis: average fitness is flat — genetic diversity collapsed (premature convergence).")
        safe_print("  Fix: increase MUTATION_RATE transiently, or add diversity-based selection.")
    elif curriculum_level >= 2 and best_ever < 60:
        safe_print("  Diagnosis: level jumped to difficulty 2+ before agents mastered the basics.")
        safe_print("  Fix: raise curriculum promotion threshold so level 1 is fully mastered first.")
    else:
        safe_print("  Diagnosis: fitness plateaued at a local optimum.")
        safe_print("  Fix: try more generations, stronger mutation, or a larger population.")
    safe_print("="* 60 + "\n")


def train_and_save_snapshots(
    generations: int,
    checkpoints: list[int],
    checkpoint_file: str,
    top_k: int = 12,
    seed: int = SEED,
    log_interval: int = 100,
    stall_window: int = STALL_WINDOW,
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
    random.seed(seed)
    population = [random_genes() for _ in range(POPULATION_SIZE)]
    best_population_snapshot = [genes[:] for genes in population]
    snapshots: dict[int, dict[str, object]] = {}
    best_fitness_history: list[float] = []
    avg_fitness_history: list[float] = []
    min_fitness_history: list[float] = []
    generation_history: list[int] = []
    best_avg_seen = -10**9
    curriculum_level = 1
    all_time_best_fitness: float = -10**9
    last_best_improved_at: int = 1

    print_run_header("Snapshot Training", seed)
    safe_print("=== Snapshot Training ===")
    safe_print(f"Target generations: {generations}")
    safe_print(f"Checkpoints: {checkpoints}")
    safe_print(f"Food coverage init: {normalize_food_coverage(food_coverage_init) * 100:.1f}%")
    safe_print(f"Spawn mode: {spawn_mode}")
    safe_print(
        f"Food respawn: mode={food_respawn_mode}, interval={max(1, int(food_respawn_interval_steps))} steps, batch={max(0.01, min(1.0, float(food_respawn_batch_ratio))) * 100:.0f}%, cycle={max(1, int(food_respawn_cycle_steps))} steps"
    )
    safe_print(
        f"Interference: mode={interference_mode}, blocker_ttl={max(1, int(blocker_ttl_steps))}, blocker_cost={max(0.0, float(blocker_energy_cost)):.2f}"
    )
    if steps_growth_final is not None:
        safe_print(
            f"Steps ramp: {STEPS_PER_GENERATION} -> {max(1, int(steps_growth_final))} over {max(1, int(steps_growth_generations))} generations"
        )
    else:
        safe_print(f"Steps per generation: {STEPS_PER_GENERATION}")
    safe_print(f"Stall window: {stall_window} gens (stop if best fitness flat for this long)")
    safe_print(f"Log interval: every {max(1, log_interval)} generations")
    safe_print(f"AI technique: genetic algorithm with 20-weight linear policy per creature\n")

    for generation in range(1, generations + 1):
        if generation == 1:
            curriculum_level = 1
        elif generation % 20 == 1:
            progress = compute_progress_signal(best_fitness_history, avg_fitness_history, window=20)
            new_level = adapt_curriculum_level(curriculum_level, progress)
            if new_level != curriculum_level:
                direction = "harder" if new_level > curriculum_level else "easier"
                safe_print(
                    f"Curriculum update at G{generation:04d}: level {curriculum_level} -> {new_level} ({direction})."
                )
            curriculum_level = new_level

        level = curriculum_level
        steps_this_generation = estimate_generation_steps(
            generation,
            STEPS_PER_GENERATION,
            steps_growth_final,
            steps_growth_generations,
        )
        scored = run_generation(
            population,
            level=level,
            food_coverage_init=food_coverage_init,
            spawn_mode=spawn_mode,
            steps_override=steps_this_generation,
            food_respawn_mode=food_respawn_mode,
            food_respawn_interval_steps=food_respawn_interval_steps,
            food_respawn_batch_ratio=food_respawn_batch_ratio,
            food_respawn_cycle_steps=food_respawn_cycle_steps,
            interference_mode=interference_mode,
            blocker_ttl_steps=blocker_ttl_steps,
            blocker_energy_cost=blocker_energy_cost,
        )
        best_fitness, best_genes, best_food = scored[0]
        avg_fitness = sum(item[0] for item in scored) / len(scored)
        min_fitness = scored[-1][0]
        generation_history.append(generation)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        min_fitness_history.append(min_fitness)

        if best_fitness > all_time_best_fitness:
            all_time_best_fitness = best_fitness
            last_best_improved_at = generation

        progress = compute_progress_signal(best_fitness_history, avg_fitness_history, window=20)

        if avg_fitness > best_avg_seen:
            best_avg_seen = avg_fitness
            best_population_snapshot = [genes[:] for _, genes, _ in scored]

        if generation in checkpoints:
            top_genes = [genes[:] for _, genes, _ in scored[: max(1, top_k)]]
            snapshots[generation] = {
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_food": best_food,
                "level": level,
                "progress_score": float(progress["score"]),
                "best_genes": best_genes[:],
                "top_genes": top_genes,
            }
            safe_print(
                f"Saved checkpoint G{generation:04d} | best={best_fitness:6.2f} | avg={avg_fitness:6.2f} | level={level} | progress={float(progress['score']):6.2f}"
            )

        if should_log_generation(generation, generations, log_interval):
            progress_status = "warming-up"
            if bool(progress["ready"]):
                progress_status = (
                    "improving"
                    if float(progress["avg_delta"]) > 2.0
                    else "stalled"
                )
            safe_print(
                f"Train G{generation:04d} | best={best_fitness:6.2f} | avg={avg_fitness:6.2f} | best food={best_food} | level={level} | steps={steps_this_generation} | status={progress_status}"
            )

        next_population = [genes[:] for _, genes, _ in scored[:ELITE_COUNT]]
        mutation_rate, mutation_strength = mutation_schedule(generation)
        top_half = [genes for _, genes, _ in scored[: POPULATION_SIZE // 2]]
        while len(next_population) < POPULATION_SIZE:
            parent_a = random.choice(top_half)
            parent_b = random.choice(top_half)
            child = crossover(parent_a, parent_b)
            next_population.append(mutate_with_parameters(child, mutation_rate, mutation_strength))

        if should_rollback_to_best(avg_fitness_history, best_avg_seen):
            rollback_elites = [genes[:] for genes in best_population_snapshot[:ELITE_COUNT]]
            next_population = rollback_elites
            while len(next_population) < POPULATION_SIZE:
                base = random.choice(best_population_snapshot[: max(ELITE_COUNT * 2, 8)])
                next_population.append(mutate_with_parameters(base, mutation_rate * 0.7, mutation_strength * 0.7))

        population = next_population

        gens_since_best_improved = generation - last_best_improved_at
        if gens_since_best_improved >= stall_window:
            safe_print(
                f"\nBest fitness unchanged for {gens_since_best_improved} generations "
                f"(last improved at G{last_best_improved_at:04d}, value={all_time_best_fitness:.2f}). Stopping."
            )
            analyze_stall(best_fitness_history, avg_fitness_history, curriculum_level, stall_window)
            break

    if snapshots:
        save_snapshot_file(
            checkpoint_file,
            snapshots,
            total_generations=generation,
            seed=seed,
            training_curve={
                "generations": generation_history,
                "avg_fitness": avg_fitness_history,
                "max_fitness": best_fitness_history,
                "min_fitness": min_fitness_history,
            },
        )
        safe_print(f"Saved snapshot file: {checkpoint_file} ({len(snapshots)} checkpoints)")
    else:
        safe_print("Warning: no checkpoints were reached before stall. Try a lower stall_window or earlier checkpoints.")


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
    snapshots, metadata = load_snapshot_file(checkpoint_file)
    available = sorted(snapshots.keys())
    if not available:
        raise SystemExit(f"No snapshots found in {checkpoint_file}")

    selected_generation = select_default_snapshot_generation(snapshots)
    serve_food_coverage = normalize_food_coverage(food_coverage_init)
    serve_world_size = WORLD_SIZE
    serve_population_size = POPULATION_SIZE
    serve_steps_per_generation = STEPS_PER_GENERATION

    def apply_environment_updates() -> None:
        global WORLD_SIZE, POPULATION_SIZE, STEPS_PER_GENERATION, FOOD_COUNT
        WORLD_SIZE = int(serve_world_size)
        POPULATION_SIZE = int(serve_population_size)
        STEPS_PER_GENERATION = int(serve_steps_per_generation)
        FOOD_COUNT = WORLD_SIZE * WORLD_SIZE
        rendering_module.WORLD_SIZE = WORLD_SIZE
        rendering_module.STEPS_PER_GENERATION = STEPS_PER_GENERATION

    apply_environment_updates()
    renderer = build_renderer(renderer_mode, fps=fps, step_skip=render_step_skip, fullscreen=fullscreen)
    if renderer is None:
        raise SystemExit("Serve mode requires a renderer (use --renderer pygame or --renderer terminal).")

    safe_print("=== Snapshot Serve Mode ===")
    safe_print(f"Loaded snapshots: {available}")
    safe_print(f"Model file: {checkpoint_file}")
    if metadata.get("total_generations"):
        safe_print(f"Training horizon: {metadata['total_generations']} generations")
    if serve_best_clones > 0:
        safe_print(f"Evaluation mode: {max(1, int(serve_best_clones))} clones of snapshot best agent")
    safe_print(
        f"Interference: mode={interference_mode}, blocker_ttl={max(1, int(blocker_ttl_steps))}, blocker_cost={max(0.0, float(blocker_energy_cost)):.2f}"
    )
    safe_print("Controls: Space pause/resume | +/- speed | S stop | click snapshot buttons or use 1-9 / [ ]\n")

    if isinstance(renderer, PygameRenderer):
        renderer.onboarding_active = False
        renderer.checkpoint_resume_required = False
        renderer.set_snapshot_choices(available, selected_generation)
        renderer.set_active_food_coverage(serve_food_coverage)
        renderer.set_active_environment(serve_world_size, serve_population_size, serve_steps_per_generation)
        renderer.set_showcase_profile(True, STEPS_PER_GENERATION)
        curve = metadata.get("training_curve", {})
        if isinstance(curve, dict):
            renderer.set_training_curve(
                generations=curve.get("generations", []),
                avg_fitness=curve.get("avg_fitness", []),
                max_fitness=curve.get("max_fitness", []),
                min_fitness=curve.get("min_fitness", []),
            )

    episode = 1
    try:
        while not renderer.should_stop():
            restart_episode = False
            if isinstance(renderer, PygameRenderer):
                requested = renderer.consume_snapshot_request()
                if requested in snapshots:
                    selected_generation = requested
                    renderer.set_active_snapshot_generation(selected_generation)
                    safe_print(f"Switched to snapshot G{selected_generation}")
                requested_coverage = renderer.consume_food_coverage_request()
                if requested_coverage is not None:
                    serve_food_coverage = normalize_food_coverage(requested_coverage)
                    renderer.set_active_food_coverage(serve_food_coverage)
                    safe_print(f"Food coverage set to {serve_food_coverage * 100:.1f}%")
                env_updates = renderer.consume_environment_request()
                if env_updates:
                    if "world_size" in env_updates:
                        serve_world_size = int(env_updates["world_size"])
                    if "population_size" in env_updates:
                        serve_population_size = int(env_updates["population_size"])
                    if "steps_per_generation" in env_updates:
                        serve_steps_per_generation = int(env_updates["steps_per_generation"])
                    apply_environment_updates()
                    renderer.set_active_environment(serve_world_size, serve_population_size, serve_steps_per_generation)
                    safe_print(
                        f"Environment set to grid={serve_world_size}, agents={serve_population_size}, steps={serve_steps_per_generation}"
                    )
                renderer.reset_episode_visuals()

            snapshot = snapshots[selected_generation]
            try:
                population = build_serve_population(
                    snapshot,
                    target_population_size=serve_population_size,
                    serve_best_clones=serve_best_clones,
                )
            except SystemExit as error:
                raise SystemExit(f"Snapshot G{selected_generation}: {error}") from error

            if isinstance(renderer, PygameRenderer):
                renderer.set_active_environment(serve_world_size, len(population), serve_steps_per_generation)

            level = int(snapshot.get("level", level_for_generation(selected_generation)))

            def on_step(step: int, creatures: list[Creature], food: set[tuple[int, int]], step_info: dict | None = None) -> None:
                nonlocal restart_episode, selected_generation
                nonlocal serve_food_coverage, serve_world_size, serve_population_size, serve_steps_per_generation
                if renderer is None:
                    return

                renderer.poll_events()
                if renderer.should_stop():
                    return

                if isinstance(renderer, PygameRenderer):
                    requested = renderer.consume_snapshot_request()
                    if requested in snapshots and requested != selected_generation:
                        selected_generation = requested
                        renderer.set_active_snapshot_generation(selected_generation)
                        safe_print(f"Switched to snapshot G{selected_generation}")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    requested_coverage = renderer.consume_food_coverage_request()
                    if requested_coverage is not None:
                        serve_food_coverage = normalize_food_coverage(requested_coverage)
                        renderer.set_active_food_coverage(serve_food_coverage)
                        safe_print(f"Food coverage set to {serve_food_coverage * 100:.1f}%")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    env_updates = renderer.consume_environment_request()
                    if env_updates:
                        if "world_size" in env_updates:
                            serve_world_size = int(env_updates["world_size"])
                        if "population_size" in env_updates:
                            serve_population_size = int(env_updates["population_size"])
                        if "steps_per_generation" in env_updates:
                            serve_steps_per_generation = int(env_updates["steps_per_generation"])
                        apply_environment_updates()
                        renderer.set_active_environment(serve_world_size, serve_population_size, serve_steps_per_generation)
                        safe_print(
                            f"Environment set to grid={serve_world_size}, agents={serve_population_size}, steps={serve_steps_per_generation}"
                        )
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    if renderer.consume_restart_request():
                        safe_print("Restart requested")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                scores = [creature_fitness(creature) for creature in creatures]
                best = max(scores)
                avg = sum(scores) / len(scores)

                while isinstance(renderer, PygameRenderer) and renderer.is_paused() and not renderer.should_stop():
                    renderer.render_generation(
                        episode,
                        None,
                        step,
                        creatures,
                        food,
                        best,
                        avg,
                        step_info,
                    )
                    renderer.poll_events()
                    requested = renderer.consume_snapshot_request()
                    if requested in snapshots and requested != selected_generation:
                        selected_generation = requested
                        renderer.set_active_snapshot_generation(selected_generation)
                        safe_print(f"Switched to snapshot G{selected_generation}")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    requested_coverage = renderer.consume_food_coverage_request()
                    if requested_coverage is not None:
                        serve_food_coverage = normalize_food_coverage(requested_coverage)
                        renderer.set_active_food_coverage(serve_food_coverage)
                        safe_print(f"Food coverage set to {serve_food_coverage * 100:.1f}%")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    env_updates = renderer.consume_environment_request()
                    if env_updates:
                        if "world_size" in env_updates:
                            serve_world_size = int(env_updates["world_size"])
                        if "population_size" in env_updates:
                            serve_population_size = int(env_updates["population_size"])
                        if "steps_per_generation" in env_updates:
                            serve_steps_per_generation = int(env_updates["steps_per_generation"])
                        apply_environment_updates()
                        renderer.set_active_environment(serve_world_size, serve_population_size, serve_steps_per_generation)
                        safe_print(
                            f"Environment set to grid={serve_world_size}, agents={serve_population_size}, steps={serve_steps_per_generation}"
                        )
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                    if renderer.consume_restart_request():
                        safe_print("Restart requested")
                        restart_episode = True
                        for creature in creatures:
                            creature.energy = 0.0
                        return

                renderer.render_generation(
                    episode,
                    None,
                    step,
                    creatures,
                    food,
                    best,
                    avg,
                    step_info,
                )

            scored = run_generation(
                population,
                step_callback=on_step,
                level=level,
                food_coverage_init=serve_food_coverage,
                spawn_mode=spawn_mode,
                food_respawn_mode=food_respawn_mode,
                food_respawn_interval_steps=food_respawn_interval_steps,
                food_respawn_batch_ratio=food_respawn_batch_ratio,
                food_respawn_cycle_steps=food_respawn_cycle_steps,
                interference_mode=interference_mode,
                blocker_ttl_steps=blocker_ttl_steps,
                blocker_energy_cost=blocker_energy_cost,
            )
            if restart_episode:
                continue
            best_fitness = scored[0][0]
            avg_fitness = sum(item[0] for item in scored) / len(scored)
            renderer.record_generation_result(episode, best_fitness, avg_fitness)
            safe_print(
                f"Serve episode {episode:04d} | snapshot G{selected_generation:04d} | best={best_fitness:6.2f} | avg={avg_fitness:6.2f}"
            )
            episode += 1
    finally:
        if isinstance(renderer, PygameRenderer) and not renderer.should_stop():
            renderer.hold_until_closed("Serve mode stopped. Close window (X) or press Esc.")
        renderer.close()


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
    if seed is not None:
        random.seed(seed)

    effective_renderer_mode = renderer_mode
    if render and renderer_mode == "none":
        effective_renderer_mode = "terminal"

    if generations is not None:
        total_generations: int | None = generations
    elif effective_renderer_mode == "pygame":
        total_generations = None
    else:
        total_generations = GENERATIONS
    population = [random_genes() for _ in range(POPULATION_SIZE)]
    best_population_snapshot = [genes[:] for genes in population]

    try:
        renderer = build_renderer(effective_renderer_mode, fps=fps, step_skip=render_step_skip, fullscreen=fullscreen)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    effective_early_stop_mode = resolve_early_stop_mode(effective_renderer_mode, early_stop_mode)

    print_run_header("Live Evolution", seed)
    safe_print("=== Tiny Evolution Game (MVP) ===")
    safe_print("Creatures learn to find food in a 2D world via genetic evolution.\n")
    safe_print(
        f"Run profile: spawn={spawn_mode}, steps={STEPS_PER_GENERATION}"
        + (
            f" -> {max(1, int(steps_growth_final))} over {max(1, int(steps_growth_generations))} generations"
            if steps_growth_final is not None
            else ""
        )
    )
    safe_print(
        f"Food respawn: mode={food_respawn_mode}, interval={max(1, int(food_respawn_interval_steps))} steps, batch={max(0.01, min(1.0, float(food_respawn_batch_ratio))) * 100:.0f}%, cycle={max(1, int(food_respawn_cycle_steps))} steps"
    )
    safe_print(
        f"Interference: mode={interference_mode}, blocker_ttl={max(1, int(blocker_ttl_steps))}, blocker_cost={max(0.0, float(blocker_energy_cost)):.2f}"
    )

    try:
        generation = 1
        showcase_every_generation = max(1, showcase_interval_generations)
        best_fitness_history: list[float] = []
        avg_fitness_history: list[float] = []
        best_avg_seen = -10**9
        stop_message = "Simulation finished. Close window (X) or press Esc."
        checkpoint_results: dict[int, dict[str, float]] = {}
        last_early_stop_warning_generation = -999
        if isinstance(renderer, PygameRenderer):
            initial_champion = select_initial_champion(population)
            checkpoint_zero = evaluate_checkpoint_benchmark(initial_champion, 0, trials=BENCHMARK_TRIALS, use_parallel=False)
            checkpoint_results[0] = checkpoint_zero
            renderer.record_checkpoint_result(0, checkpoint_zero)
            safe_print(
                "Benchmark G000 | "
                f"overall={checkpoint_zero['overall']:.1f} "
                f"basic={checkpoint_zero['forage_basic']:.1f} "
                f"maze={checkpoint_zero['forage_maze']:.1f} "
                f"escape={checkpoint_zero['predator_escape']:.1f}"
            )

        while total_generations is None or generation <= total_generations:
            level = level_for_generation(generation)
            is_showcase_generation = should_showcase_generation(generation, showcase_every_generation)
            steps_this_generation = estimate_generation_steps(
                generation,
                STEPS_PER_GENERATION,
                steps_growth_final,
                steps_growth_generations,
            )

            if isinstance(renderer, PygameRenderer):
                renderer.set_showcase_profile(is_showcase_generation, steps_this_generation)

            def on_step(step: int, creatures: list[Creature], food: set[tuple[int, int]], step_info: dict | None = None) -> None:
                if renderer is None:
                    return

                renderer.poll_events()
                if renderer.should_stop():
                    return

                if isinstance(renderer, PygameRenderer):
                    while renderer.requires_user_action() and not renderer.should_stop():
                        current_scores = [creature_fitness(creature) for creature in creatures]
                        best = max(current_scores)
                        avg = sum(current_scores) / len(current_scores)
                        renderer.render_generation(generation, total_generations, step, creatures, food, best, avg, step_info)

                is_pygame = isinstance(renderer, PygameRenderer)
                if is_pygame:
                    if not is_showcase_generation:
                        return
                elif step % max(1, render_step_skip) != 0:
                    return

                current_scores = [creature_fitness(creature) for creature in creatures]
                best = max(current_scores)
                avg = sum(current_scores) / len(current_scores)
                if generation % max(1, render_every_generation) == 0:
                    renderer.render_generation(generation, total_generations, step, creatures, food, best, avg, step_info)

                if isinstance(renderer, PygameRenderer):
                    while renderer.requires_checkpoint_resume() and not renderer.should_stop():
                        renderer.render_generation(generation, total_generations, step, creatures, food, best, avg, step_info)

            scored = run_generation(
                population,
                step_callback=on_step if renderer else None,
                level=level,
                food_coverage_init=food_coverage_init,
                spawn_mode=spawn_mode,
                steps_override=steps_this_generation,
                food_respawn_mode=food_respawn_mode,
                food_respawn_interval_steps=food_respawn_interval_steps,
                food_respawn_batch_ratio=food_respawn_batch_ratio,
                food_respawn_cycle_steps=food_respawn_cycle_steps,
                interference_mode=interference_mode,
                blocker_ttl_steps=blocker_ttl_steps,
                blocker_energy_cost=blocker_energy_cost,
            )
            best_fitness, _, best_food = scored[0]
            avg_fitness = sum(item[0] for item in scored) / len(scored)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if avg_fitness > best_avg_seen:
                best_avg_seen = avg_fitness
                best_population_snapshot = [genes[:] for _, genes, _ in scored]

            if renderer is not None:
                renderer.record_generation_result(generation, best_fitness, avg_fitness)

            if (
                isinstance(renderer, PygameRenderer)
                and generation in CHECKPOINT_GENERATIONS
                and generation not in checkpoint_results
            ):
                checkpoint = evaluate_checkpoint_benchmark(scored[0][1], generation, trials=BENCHMARK_TRIALS, use_parallel=False)
                checkpoint_results[generation] = checkpoint
                renderer.record_checkpoint_result(generation, checkpoint)
                safe_print(
                    f"Benchmark G{generation:03d} | "
                    f"overall={checkpoint['overall']:.1f} "
                    f"basic={checkpoint['forage_basic']:.1f} "
                    f"maze={checkpoint['forage_maze']:.1f} "
                    f"escape={checkpoint['predator_escape']:.1f}"
                )

            if renderer and generation % max(1, render_every_generation) == 0:
                safe_print()

            if should_log_generation(generation, total_generations, log_interval):
                safe_print(
                    f"Gen {generation:02d} | best fitness={best_fitness:6.2f} | "
                    f"avg fitness={avg_fitness:6.2f} | best food eaten={best_food} | level={level} | steps={steps_this_generation}"
                )

            next_population = [genes[:] for _, genes, _ in scored[:ELITE_COUNT]]
            mutation_rate, mutation_strength = mutation_schedule(generation)

            top_half = [genes for _, genes, _ in scored[: POPULATION_SIZE // 2]]
            while len(next_population) < POPULATION_SIZE:
                parent_a = random.choice(top_half)
                parent_b = random.choice(top_half)
                child = crossover(parent_a, parent_b)
                child = mutate_with_parameters(child, mutation_rate, mutation_strength)
                next_population.append(child)

            if should_rollback_to_best(avg_fitness_history, best_avg_seen):
                rollback_elites = [genes[:] for genes in best_population_snapshot[:ELITE_COUNT]]
                next_population = rollback_elites
                while len(next_population) < POPULATION_SIZE:
                    base = random.choice(best_population_snapshot[: max(ELITE_COUNT * 2, 8)])
                    next_population.append(
                        mutate_with_parameters(base, mutation_rate * 0.7, mutation_strength * 0.7)
                    )
                safe_print(
                    "Recovery: reverted toward best-known population to stabilize learning."
                )

            population = next_population

            if renderer is not None and renderer.should_stop():
                safe_print("\nRenderer requested stop. Ending run early.")
                break

            min_generation_before_stop = 120 if isinstance(renderer, PygameRenderer) else 25
            if effective_early_stop_mode != "off":
                should_stop, stop_reason = should_stop_early(
                    best_fitness_history,
                    avg_fitness_history,
                    min_generations=min_generation_before_stop,
                )
                if should_stop:
                    if effective_early_stop_mode == "auto":
                        if stop_reason == "plateau":
                            safe_print("\nEarly stop: evolution plateau detected (minimal improvement window).")
                            stop_message = "Early stop: plateau detected. Close window (X) or press Esc."
                        else:
                            safe_print("\nEarly stop: noisy learning with no clear trend detected.")
                            stop_message = "Early stop: noisy no-trend phase. Close window (X) or press Esc."
                        break

                    if generation - last_early_stop_warning_generation >= 20:
                        if stop_reason == "plateau":
                            safe_print("Warning: plateau detected. Continuing because early-stop mode is warn.")
                        else:
                            safe_print("Warning: noisy no-trend phase detected. Continuing because early-stop mode is warn.")
                        last_early_stop_warning_generation = generation

            generation += 1
    finally:
        if isinstance(renderer, PygameRenderer) and not renderer.should_stop():
            renderer.hold_until_closed(stop_message)

        if renderer is not None:
            renderer.close()

    safe_print("\nDone. Next step: add predators and richer species traits.")


def build_snapshot_schedule(max_generations: int) -> list[int]:
    target = max(1, int(max_generations))
    raw = {
        max(1, int(target * 0.10)),
        max(1, int(target * 0.20)),
        max(1, int(target * 0.30)),
        max(1, int(target * 0.40)),
        max(1, int(target * 0.50)),
        max(1, int(target * 0.75)),
        target,
    }
    return sorted(raw)


def launch_training_setup_ui() -> dict[str, float | int] | None:
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("pygame is required for interactive start mode") from error

    pygame.init()
    screen = pygame.display.set_mode((920, 620))
    pygame.display.set_caption("Training Setup")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 26)
    small = pygame.font.SysFont("DejaVu Sans", 20)

    fields = [
        {"label": "Max generations", "key": "max_generations", "value": 4000, "min": 100, "max": 50000, "step": 100},
        {"label": "Steps / generation", "key": "steps_per_generation", "value": 120, "min": 20, "max": 1000, "step": 10},
        {"label": "Food coverage %", "key": "food_coverage_pct", "value": 35, "min": 1, "max": 100, "step": 1},
        {"label": "Number of agents", "key": "population_size", "value": 50, "min": 2, "max": 300, "step": 2},
        {"label": "Grid size", "key": "world_size", "value": 20, "min": 8, "max": 50, "step": 1},
    ]
    selected = 0
    running = True
    accepted = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                    accepted = True
                    running = False
                elif event.key == pygame.K_UP:
                    selected = max(0, selected - 1)
                elif event.key == pygame.K_DOWN:
                    selected = min(len(fields) - 1, selected + 1)
                elif event.key in {pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS}:
                    field = fields[selected]
                    field["value"] = max(field["min"], int(field["value"]) - int(field["step"]))
                elif event.key in {pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}:
                    field = fields[selected]
                    field["value"] = min(field["max"], int(field["value"]) + int(field["step"]))

        screen.fill((18, 20, 26))
        title = font.render("Training setup", True, (236, 239, 244))
        screen.blit(title, (34, 28))
        hint = small.render("UP/DOWN select | LEFT/RIGHT change | ENTER start | ESC quit", True, (165, 172, 186))
        screen.blit(hint, (34, 70))

        y = 140
        for idx, field in enumerate(fields):
            active = idx == selected
            color = (255, 210, 120) if active else (236, 239, 244)
            row = small.render(f"{field['label']}: {field['value']}", True, color)
            screen.blit(row, (52, y))
            y += 52

        prompt = small.render("Press ENTER to launch training then demo", True, (140, 210, 160))
        screen.blit(prompt, (34, 530))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    if not accepted:
        return None

    return {
        "max_generations": int(fields[0]["value"]),
        "steps_per_generation": int(fields[1]["value"]),
        "food_coverage_pct": int(fields[2]["value"]),
        "population_size": int(fields[3]["value"]),
        "world_size": int(fields[4]["value"]),
    }


def interactive_train_and_serve(
    checkpoint_file: str,
    renderer_mode: str,
    fps: float,
    render_step_skip: int,
    fullscreen: bool,
    seed: int,
    log_interval: int,
) -> None:
    setup = launch_training_setup_ui()
    if setup is None:
        safe_print("Interactive start cancelled.")
        return

    global WORLD_SIZE, POPULATION_SIZE, STEPS_PER_GENERATION, FOOD_COUNT
    WORLD_SIZE = int(setup["world_size"])
    POPULATION_SIZE = int(setup["population_size"])
    STEPS_PER_GENERATION = int(setup["steps_per_generation"])
    FOOD_COUNT = WORLD_SIZE * WORLD_SIZE
    rendering_module.WORLD_SIZE = WORLD_SIZE
    rendering_module.STEPS_PER_GENERATION = STEPS_PER_GENERATION

    max_generations = int(setup["max_generations"])
    food_coverage = normalize_food_coverage(float(setup["food_coverage_pct"]))
    checkpoints = build_snapshot_schedule(max_generations)

    train_and_save_snapshots(
        generations=max_generations,
        checkpoints=checkpoints,
        checkpoint_file=checkpoint_file,
        top_k=12,
        seed=seed,
        log_interval=max(100, log_interval),
        stall_window=max_generations + 1,
        food_coverage_init=food_coverage,
    )

    serve_saved_snapshots(
        checkpoint_file=checkpoint_file,
        renderer_mode=renderer_mode,
        fps=fps,
        render_step_skip=render_step_skip,
        fullscreen=fullscreen,
        food_coverage_init=food_coverage,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny evolution game with optional live terminal rendering.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Legacy shortcut for terminal renderer (same as --renderer terminal).",
    )
    parser.add_argument(
        "--renderer",
        choices=["none", "terminal", "pygame"],
        default="none",
        help="Renderer to use: none, terminal, or pygame (default: none).",
    )
    parser.add_argument("--pygame", action="store_true", help="Shortcut for --renderer pygame.")
    parser.add_argument("--fps", type=float, default=20.0, help="Maximum render frames per second (default: 20).")
    parser.add_argument(
        "--render-step-skip",
        type=int,
        default=2,
        help="Render every N simulation steps for speed (default: 2).",
    )
    parser.add_argument(
        "--render-every-generation",
        type=int,
        default=1,
        help="Render one generation out of N (default: 1).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations to run (default: 30 for non-pygame, open-ended for pygame).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    parser.add_argument("--fullscreen", action="store_true", help="Start pygame renderer in fullscreen mode.")
    parser.add_argument(
        "--early-stop",
        choices=["off", "warn", "auto"],
        default=None,
        help="Early-stop behavior: off (never), warn (notify only), auto (stop). Default: warn for pygame, auto otherwise.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print generation stats every N generations (default: 1).",
    )
    parser.add_argument(
        "--showcase-interval-generations",
        type=int,
        default=SHOWCASE_INTERVAL_GENERATIONS,
        help=(
            "In pygame mode, run slow showcase playback every N generations "
            "(default: 200, use 1 for every generation)."
        ),
    )
    parser.add_argument(
        "--train-snapshots",
        action="store_true",
        help="Train offline and save checkpoint snapshots (best/top agents).",
    )
    parser.add_argument(
        "--serve-snapshots",
        action="store_true",
        help="Serve a saved snapshot file for interactive visualization.",
    )
    parser.add_argument(
        "--serve-best-clones",
        type=int,
        default=0,
        help="In serve mode, run N clones of the snapshot best agent instead of mixed top agents (default: 0=disabled).",
    )
    parser.add_argument(
        "--snapshot-file",
        default="models/snapshots.json",
        help="Snapshot file path for train/serve modes (default: models/snapshots.json).",
    )
    parser.add_argument(
        "--snapshot-generations",
        default=",".join(str(value) for value in DEFAULT_SNAPSHOT_GENERATIONS),
        help="Comma-separated generations to save (default: 100,500,1000,2000).",
    )
    parser.add_argument(
        "--snapshot-top-k",
        type=int,
        default=12,
        help="How many top agents to store per snapshot (default: 12).",
    )
    parser.add_argument(
        "--train-generations",
        type=int,
        default=2000,
        help="Total generations for --train-snapshots (default: 2000).",
    )
    parser.add_argument(
        "--food-coverage-init",
        type=float,
        default=1.0,
        help="Initial food coverage (0-1 or 0-100 percent) for each generation/episode (default: 1.0).",
    )
    parser.add_argument(
        "--spawn-mode",
        choices=["random", "corners", "biased-corners"],
        default=DEFAULT_SPAWN_MODE,
        help="Spawn pattern for each generation: random, corners, or biased-corners (default: biased-corners).",
    )
    parser.add_argument(
        "--steps-growth-final",
        type=int,
        default=DEFAULT_STEPS_GROWTH_FINAL,
        help="Final steps per generation for progressive long-run training (default: 320).",
    )
    parser.add_argument(
        "--steps-growth-generations",
        type=int,
        default=DEFAULT_STEPS_GROWTH_GENERATIONS,
        help="Number of generations used to ramp steps from base to --steps-growth-final (default: 500).",
    )
    parser.add_argument(
        "--food-respawn-mode",
        choices=["off", "random", "right-side", "clockwise-quadrants"],
        default=DEFAULT_FOOD_RESPAWN_MODE,
        help="Food respawn behavior during generation: off, random, right-side, clockwise-quadrants (default: right-side).",
    )
    parser.add_argument(
        "--food-respawn-interval-steps",
        type=int,
        default=FOOD_RESPAWN_INTERVAL_STEPS,
        help="Respawn pulse interval in simulation steps (default: 10).",
    )
    parser.add_argument(
        "--food-respawn-batch-ratio",
        type=float,
        default=FOOD_RESPAWN_BATCH_RATIO,
        help="Respawn pulse size as ratio of initial food target (default: 0.18).",
    )
    parser.add_argument(
        "--food-respawn-cycle-steps",
        type=int,
        default=FOOD_RESPAWN_CYCLE_STEPS,
        help="For clockwise-quadrants mode, respawn pulses spent in each quadrant before switching (default: 30).",
    )
    parser.add_argument(
        "--interference-mode",
        choices=["off", "simple-block"],
        default=DEFAULT_INTERFERENCE_MODE,
        help="Optional competitive mechanic: off or simple-block (default: off).",
    )
    parser.add_argument(
        "--blocker-ttl-steps",
        type=int,
        default=BLOCKER_TTL_STEPS,
        help="How many steps a placed blocker lasts in simple-block mode (default: 4).",
    )
    parser.add_argument(
        "--blocker-energy-cost",
        type=float,
        default=BLOCKER_ENERGY_COST,
        help="Energy cost paid by an agent when placing a blocker (default: 0.25).",
    )
    parser.add_argument(
        "--interactive-start",
        action="store_true",
        help="Open setup UI, train with selected params, then launch demo automatically.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.interactive_start:
        selected_renderer = args.renderer
        if args.pygame:
            selected_renderer = "pygame"
        if selected_renderer == "none":
            selected_renderer = "pygame"
        interactive_train_and_serve(
            checkpoint_file=args.snapshot_file,
            renderer_mode=selected_renderer,
            fps=args.fps,
            render_step_skip=args.render_step_skip,
            fullscreen=args.fullscreen,
            seed=args.seed,
            log_interval=max(1, args.log_interval),
        )
        raise SystemExit(0)

    selected_renderer = args.renderer
    if args.pygame:
        selected_renderer = "pygame"

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
            serve_best_clones=max(0, int(args.serve_best_clones)),
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
