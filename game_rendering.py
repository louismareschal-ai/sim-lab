from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from game_constants import POPULATION_SIZE, SHOWCASE_PLAYBACK_SECONDS, STEPS_PER_GENERATION, WORLD_SIZE

if TYPE_CHECKING:
    from main import Creature


def safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        raise SystemExit(0)


def advance_onboarding_state(step: int, total_steps: int, action: str) -> tuple[int, bool]:
    if total_steps <= 0:
        return 0, False

    last_index = total_steps - 1

    if action == "next":
        if step >= last_index:
            return last_index, False
        return min(last_index, step + 1), True

    if action == "back":
        return max(0, step - 1), True

    if action == "close":
        return min(max(0, step), last_index), False

    return min(max(0, step), last_index), True


def checkpoint_requires_resume(generation: int, onboarding_active: bool) -> bool:
    return generation > 0 and not onboarding_active


def checkpoint_should_defer(generation: int, onboarding_active: bool) -> bool:
    return generation > 0 and onboarding_active


def playback_fps_for_steps(total_steps: int, duration_seconds: int) -> float:
    if duration_seconds <= 0:
        return 1.0
    return max(1.0, total_steps / duration_seconds)


class TerminalRenderer:
    RESET = "\033[0m"
    FOOD = "\033[92m"
    BEST = "\033[95m"
    LOW = "\033[91m"
    MID = "\033[93m"
    HIGH = "\033[96m"

    def __init__(self, fps: float, step_skip: int):
        self.fps = max(1.0, fps)
        self.step_skip = max(1, step_skip)
        self.frame_delay = 1.0 / self.fps
        self._last_frame = 0.0

    def _creature_color(self, creature: "Creature") -> str:
        if creature.food_eaten >= 4:
            return self.HIGH
        if creature.food_eaten >= 2:
            return self.MID
        return self.LOW

    def render_generation(
        self,
        generation: int,
        total_generations: int | None,
        step: int,
        creatures: list["Creature"],
        food: set[tuple[int, int]],
        best_fitness: float,
        avg_fitness: float,
        step_info: dict | None = None,
    ) -> None:
        _ = step_info
        if step % self.step_skip != 0:
            return

        now = time.monotonic()
        remaining = self.frame_delay - (now - self._last_frame)
        if remaining > 0:
            time.sleep(remaining)
            now += remaining
        self._last_frame = now

        best_creature = max(creatures, key=lambda c: (c.food_eaten, c.energy))
        creature_positions = {(c.x, c.y): c for c in creatures if c.energy > 0}
        total_label = "∞" if total_generations is None else f"{total_generations:02d}"

        lines = [
            "\033[H\033[2J=== Tiny Evolution Game (Render Mode) ===",
            (
                f"Generation {generation:02d}/{total_label} | "
                f"Step {step + 1:03d}/{STEPS_PER_GENERATION:03d} | "
                f"Best fitness {best_fitness:6.2f} | Avg {avg_fitness:6.2f}"
            ),
        ]

        horizontal = "+" + ("-" * WORLD_SIZE) + "+"
        lines.append(horizontal)

        for y in range(WORLD_SIZE):
            row = ["|"]
            for x in range(WORLD_SIZE):
                pos = (x, y)
                if pos == (best_creature.x, best_creature.y) and best_creature.energy > 0:
                    row.append(f"{self.BEST}@{self.RESET}")
                elif pos in creature_positions:
                    color = self._creature_color(creature_positions[pos])
                    row.append(f"{color}o{self.RESET}")
                elif pos in food:
                    row.append(f"{self.FOOD}.{self.RESET}")
                else:
                    row.append(" ")
            row.append("|")
            lines.append("".join(row))

        lines.append(horizontal)
        safe_print("\n".join(lines), end="", flush=True)

    def should_stop(self) -> bool:
        return False

    def record_generation_result(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        _ = (generation, best_fitness, avg_fitness)

    def record_checkpoint_result(self, generation: int, result: dict[str, float]) -> None:
        _ = (generation, result)

    def poll_events(self) -> None:
        return None

    def close(self) -> None:
        return None


class PygameRenderer:
    def __init__(self, fps: float, step_skip: int, fullscreen: bool = False):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError(
                "pygame is required for GUI rendering. Install it with: pip install pygame"
            ) from error

        self.pygame = pygame
        self.fps = max(1.0, fps)
        self.training_fps = self.fps
        self.step_skip = max(1, step_skip)
        self.fullscreen = fullscreen
        self.frame_delay = 1.0 / self.fps
        self._last_frame = 0.0
        self._should_stop = False
        self.paused = False

        self.snapshot_generations: list[int] = []
        self.active_snapshot_generation: int | None = None
        self.pending_snapshot_generation: int | None = None
        self.snapshot_button_rects: list[tuple[int, object]] = []
        self.generation_history: list[int] = []
        self.best_history: list[float] = []
        self.avg_history: list[float] = []
        self.min_history: list[float] = []
        self.training_curve_locked: bool = False
        self.history_limit = 160
        self.food_coverage_options: list[float] = [0.10, 0.25, 0.50, 0.75, 1.00]
        self.active_food_coverage: float = 1.0
        self.pending_food_coverage: float | None = None
        self.food_coverage_button_rects: list[tuple[float, object]] = []
        self.steps_options: list[int] = [60, 120, 180, 240, 320, 480, 640]
        self.population_options: list[int] = [10, 20, 50, 100]
        self.grid_options: list[int] = [10, 15, 20, 25]
        self.active_steps_per_generation: int = STEPS_PER_GENERATION
        self.active_population_size: int = POPULATION_SIZE
        self.active_world_size: int = WORLD_SIZE
        self.pending_steps_per_generation: int | None = None
        self.pending_population_size: int | None = None
        self.pending_world_size: int | None = None
        self.steps_button_rects: list[tuple[int, object]] = []
        self.population_button_rects: list[tuple[int, object]] = []
        self.grid_button_rects: list[tuple[int, object]] = []
        self.restart_requested = False

        self.focus_agent_id: int | None = None
        self.focus_trail: list[tuple[int, int]] = []

        self.button_pause_rect = None

        pygame.init()
        pygame.display.set_caption("Tiny Evolution Game")
        info = pygame.display.Info()
        if fullscreen:
            self.screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((min(1400, info.current_w), min(950, info.current_h)))

        self.window_width, self.window_height = self.screen.get_size()
        self.margin = max(20, self.window_width // 70)
        self.hud_height = max(90, self.window_height // 9)

        usable_world_height = self.window_height - (self.margin * 2) - self.hud_height
        self.cell_size = max(24, min(72, usable_world_height // WORLD_SIZE))
        self.world_px = self.cell_size * WORLD_SIZE

        self.sidebar_width = max(280, self.window_width - self.world_px - self.margin * 3)
        if self.sidebar_width < 280:
            self.cell_size = max(20, (self.window_width - self.margin * 3 - 280) // WORLD_SIZE)
            self.world_px = self.cell_size * WORLD_SIZE
            self.sidebar_width = max(280, self.window_width - self.world_px - self.margin * 3)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("DejaVu Sans", 19)
        self.small_font = pygame.font.SysFont("DejaVu Sans", 16)

        self.colors = {
            "bg": (16, 18, 23),
            "world": (24, 28, 38),
            "border": (70, 80, 100),
            "grid": (44, 50, 62),
            "food": (82, 184, 118),
            "obstacle": (96, 104, 122),
            "predator": (224, 92, 92),
            "creature": (120, 172, 212),
            "best": (184, 132, 220),
            "text": (236, 239, 244),
            "muted": (165, 172, 186),
        }

        self.onboarding_active = False
        self.checkpoint_resume_required = False

    def _draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int], small: bool = False) -> None:
        font = self.small_font if small else self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_button(self, rect, label: str, active: bool = False) -> None:
        fill = (77, 109, 179) if active else (54, 62, 79)
        self.pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        self.pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=6)
        self._draw_text(label, rect.x + 10, rect.y + 7, self.colors["text"], small=True)

    def _draw_controls(self, x: int, y: int) -> int:
        self._draw_text("Controls", x, y, self.colors["text"])
        y += 22
        self.button_pause_rect = self.pygame.Rect(x, y, 110, 30)
        self.button_restart_rect = self.pygame.Rect(x + 118, y, 110, 30)
        self._draw_button(self.button_pause_rect, "Resume" if self.paused else "Pause", active=self.paused)
        self._draw_button(self.button_restart_rect, "Restart")
        return y + 38

    def _draw_food_coverage_selector(self, x: int, y: int) -> int:
        self._draw_text("Food init", x, y, self.colors["text"])
        y += 22
        self.food_coverage_button_rects = []

        button_w = 70
        button_h = 28
        gap = 8
        max_columns = max(1, min(3, self.sidebar_width // (button_w + gap)))

        for idx, coverage in enumerate(self.food_coverage_options):
            row = idx // max_columns
            col = idx % max_columns
            bx = x + col * (button_w + gap)
            by = y + row * (button_h + gap)
            rect = self.pygame.Rect(bx, by, button_w, button_h)
            active = abs(self.active_food_coverage - coverage) < 1e-6
            self._draw_button(rect, f"{int(coverage * 100)}%", active=active)
            self.food_coverage_button_rects.append((coverage, rect))

        rows = (len(self.food_coverage_options) + max_columns - 1) // max_columns
        y += rows * (button_h + gap)
        self._draw_text("change => instant restart", x, y, self.colors["muted"], small=True)
        return y + 20

    def _draw_env_selector(
        self,
        title: str,
        options: list[int],
        active_value: int,
        x: int,
        y: int,
        button_rects: list[tuple[int, object]],
    ) -> int:
        self._draw_text(title, x, y, self.colors["text"])
        y += 22
        button_rects.clear()

        button_w = 70
        button_h = 28
        gap = 8
        max_columns = max(1, min(3, self.sidebar_width // (button_w + gap)))

        for idx, value in enumerate(options):
            row = idx // max_columns
            col = idx % max_columns
            bx = x + col * (button_w + gap)
            by = y + row * (button_h + gap)
            rect = self.pygame.Rect(bx, by, button_w, button_h)
            active = value == active_value
            self._draw_button(rect, f"{value}", active=active)
            button_rects.append((value, rect))

        rows = (len(options) + max_columns - 1) // max_columns
        y += rows * (button_h + gap)
        return y + 8

    def _draw_snapshot_selector(self, x: int, y: int) -> int:
        self.snapshot_button_rects = []
        if not self.snapshot_generations:
            return y

        self._draw_text("Snapshots", x, y, self.colors["text"])
        y += 22

        button_w = 108
        button_h = 28
        gap = 8
        max_columns = max(1, min(2, self.sidebar_width // (button_w + gap)))

        for idx, generation in enumerate(self.snapshot_generations):
            row = idx // max_columns
            col = idx % max_columns
            bx = x + col * (button_w + gap)
            by = y + row * (button_h + gap)
            rect = self.pygame.Rect(bx, by, button_w, button_h)
            active = generation == self.active_snapshot_generation
            self._draw_button(rect, f"G{generation}", active=active)
            self.snapshot_button_rects.append((generation, rect))

        rows = (len(self.snapshot_generations) + max_columns - 1) // max_columns
        y += rows * (button_h + gap)
        self._draw_text("[ ] or 1-9 to switch", x, y, self.colors["muted"], small=True)
        return y + 20

    def _draw_legend(self, x: int, y: int) -> int:
        self._draw_text("Legend", x, y, self.colors["text"])
        y += 24

        items = [
            (self.colors["food"], "Food"),
            (self.colors["obstacle"], "Wall"),
            (self.colors["creature"], "Agent"),
            (self.colors["best"], "Focus agent"),
            (self.colors["predator"], "Predator"),
        ]

        for color, label in items:
            self.pygame.draw.circle(self.screen, color, (x + 8, y + 8), 6)
            self._draw_text(label, x + 22, y, self.colors["text"], small=True)
            y += 20

        return y

    def _draw_learning_curve(self, x: int, y: int, width: int, height: int) -> int:
        self._draw_text("Learning curve (training: avg with min/max)", x, y, self.colors["text"], small=True)
        y += 20
        rect = self.pygame.Rect(x, y, width, height)
        self.pygame.draw.rect(self.screen, (17, 20, 28), rect, border_radius=6)
        self.pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=6)

        if len(self.generation_history) < 2 or len(self.avg_history) < 2:
            self._draw_text("Need 2+ generations", x + 8, y + 8, self.colors["muted"], small=True)
            return y + height + 6

        series_values = list(self.avg_history)
        if self.best_history:
            series_values.extend(self.best_history)
        if self.min_history:
            series_values.extend(self.min_history)
        minimum = min(series_values)
        maximum = max(series_values)
        span = max(1.0, maximum - minimum)

        min_generation = min(self.generation_history)
        max_generation = max(self.generation_history)
        generation_span = max(1, max_generation - min_generation)

        def map_point(generation: int, value: float) -> tuple[int, int]:
            px = x + 8 + int((generation - min_generation) * (width - 16) / generation_span)
            normalized = (value - minimum) / span
            py = y + height - 8 - int(normalized * (height - 16))
            return px, py

        avg_points = [
            map_point(generation, value)
            for generation, value in zip(self.generation_history, self.avg_history, strict=False)
        ]
        min_points = [
            map_point(generation, value)
            for generation, value in zip(self.generation_history, self.min_history, strict=False)
        ]
        max_points = [
            map_point(generation, value)
            for generation, value in zip(self.generation_history, self.best_history, strict=False)
        ]

        if len(avg_points) >= 2:
            self.pygame.draw.lines(self.screen, self.colors["best"], False, avg_points, 2)

        def draw_dotted(points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
            if len(points) < 2:
                return
            for idx in range(len(points) - 1):
                if idx % 2 == 0:
                    self.pygame.draw.line(self.screen, color, points[idx], points[idx + 1], 1)

        draw_dotted(max_points, (222, 140, 192))
        draw_dotted(min_points, (132, 168, 222))

        self._draw_text(f"min {minimum:.1f}", x + 8, y + height - 18, self.colors["muted"], small=True)
        self._draw_text(f"max {maximum:.1f}", x + 8, y + 4, self.colors["muted"], small=True)
        self._draw_text("x: generation | y: fitness", x + width - 180, y + height - 18, self.colors["muted"], small=True)
        return y + height + 6

    def _handle_events(self) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self._should_stop = True
                return
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_ESCAPE:
                self._should_stop = True
                return
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_SPACE:
                self.paused = not self.paused
            if event.type == self.pygame.KEYDOWN and event.key == self.pygame.K_s:
                self._should_stop = True
                return
            if event.type == self.pygame.KEYDOWN and event.key in {
                self.pygame.K_PLUS,
                self.pygame.K_EQUALS,
                self.pygame.K_KP_PLUS,
            }:
                self.training_fps = min(60.0, self.training_fps + 1.0)
                self.fps = self.training_fps
                self.frame_delay = 1.0 / self.fps
            if event.type == self.pygame.KEYDOWN and event.key in {
                self.pygame.K_MINUS,
                self.pygame.K_KP_MINUS,
            }:
                self.training_fps = max(1.0, self.training_fps - 1.0)
                self.fps = self.training_fps
                self.frame_delay = 1.0 / self.fps

            if event.type == self.pygame.KEYDOWN and self.snapshot_generations:
                numeric_map = {
                    self.pygame.K_1: 0,
                    self.pygame.K_2: 1,
                    self.pygame.K_3: 2,
                    self.pygame.K_4: 3,
                    self.pygame.K_5: 4,
                    self.pygame.K_6: 5,
                    self.pygame.K_7: 6,
                    self.pygame.K_8: 7,
                    self.pygame.K_9: 8,
                }
                if event.key in numeric_map:
                    idx = numeric_map[event.key]
                    if idx < len(self.snapshot_generations):
                        self.pending_snapshot_generation = self.snapshot_generations[idx]

                if event.key in {self.pygame.K_LEFTBRACKET, self.pygame.K_COMMA} and self.active_snapshot_generation in self.snapshot_generations:
                    idx = self.snapshot_generations.index(self.active_snapshot_generation)
                    self.pending_snapshot_generation = self.snapshot_generations[max(0, idx - 1)]
                if event.key in {self.pygame.K_RIGHTBRACKET, self.pygame.K_PERIOD} and self.active_snapshot_generation in self.snapshot_generations:
                    idx = self.snapshot_generations.index(self.active_snapshot_generation)
                    self.pending_snapshot_generation = self.snapshot_generations[min(len(self.snapshot_generations) - 1, idx + 1)]

            if event.type == self.pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if self.button_pause_rect and self.button_pause_rect.collidepoint(mx, my):
                    self.paused = not self.paused
                if self.button_restart_rect and self.button_restart_rect.collidepoint(mx, my):
                    self.restart_requested = True

                for generation, rect in self.snapshot_button_rects:
                    if rect.collidepoint(mx, my):
                        self.pending_snapshot_generation = generation
                        break

                for coverage, rect in self.food_coverage_button_rects:
                    if rect.collidepoint(mx, my):
                        self.pending_food_coverage = coverage
                        self.restart_requested = True
                        break

                for steps_value, rect in self.steps_button_rects:
                    if rect.collidepoint(mx, my):
                        self.pending_steps_per_generation = steps_value
                        self.restart_requested = True
                        break

                for population_value, rect in self.population_button_rects:
                    if rect.collidepoint(mx, my):
                        self.pending_population_size = population_value
                        self.restart_requested = True
                        break

                for grid_value, rect in self.grid_button_rects:
                    if rect.collidepoint(mx, my):
                        self.pending_world_size = grid_value
                        self.restart_requested = True
                        break

    def record_generation_result(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
        if self.training_curve_locked:
            return
        self.generation_history.append(generation)
        self.best_history.append(best_fitness)
        self.avg_history.append(avg_fitness)
        self.min_history.append(avg_fitness)
        if len(self.generation_history) > self.history_limit:
            self.generation_history = self.generation_history[-self.history_limit :]
            self.best_history = self.best_history[-self.history_limit :]
            self.avg_history = self.avg_history[-self.history_limit :]
            self.min_history = self.min_history[-self.history_limit :]

    def record_checkpoint_result(self, generation: int, result: dict[str, float]) -> None:
        _ = (generation, result)

    def mark_checkpoint_pending(self, generation: int) -> None:
        _ = generation

    def requires_checkpoint_resume(self) -> bool:
        return False

    def requires_user_action(self) -> bool:
        return False

    def poll_events(self) -> None:
        self._handle_events()

    def set_showcase_profile(self, is_showcase: bool, total_steps: int) -> None:
        if is_showcase:
            showcase_target_fps = playback_fps_for_steps(total_steps, SHOWCASE_PLAYBACK_SECONDS)
            self.fps = max(1.0, min(self.training_fps, showcase_target_fps))
        else:
            self.fps = self.training_fps
        self.frame_delay = 1.0 / self.fps
        self.step_skip = 1

    def set_snapshot_choices(self, generations: list[int], active_generation: int) -> None:
        self.snapshot_generations = sorted(generations)
        self.active_snapshot_generation = active_generation
        self.pending_snapshot_generation = None

    def consume_snapshot_request(self) -> int | None:
        requested = self.pending_snapshot_generation
        self.pending_snapshot_generation = None
        return requested

    def consume_food_coverage_request(self) -> float | None:
        requested = self.pending_food_coverage
        self.pending_food_coverage = None
        return requested

    def set_active_food_coverage(self, coverage: float) -> None:
        self.active_food_coverage = max(0.01, min(1.0, coverage))

    def set_active_environment(self, world_size: int, population_size: int, steps_per_generation: int) -> None:
        self.active_world_size = int(world_size)
        self.active_population_size = int(population_size)
        self.active_steps_per_generation = int(steps_per_generation)

    def set_training_curve(
        self,
        generations: list[int],
        avg_fitness: list[float],
        max_fitness: list[float],
        min_fitness: list[float],
    ) -> None:
        paired = list(zip(generations, avg_fitness, max_fitness, min_fitness, strict=False))
        if len(paired) < 2:
            self.training_curve_locked = False
            return

        self.generation_history = [int(item[0]) for item in paired]
        self.avg_history = [float(item[1]) for item in paired]
        self.best_history = [float(item[2]) for item in paired]
        self.min_history = [float(item[3]) for item in paired]
        self.training_curve_locked = True

    def consume_environment_request(self) -> dict[str, int]:
        updates: dict[str, int] = {}
        if self.pending_steps_per_generation is not None:
            updates["steps_per_generation"] = int(self.pending_steps_per_generation)
            self.pending_steps_per_generation = None
        if self.pending_population_size is not None:
            updates["population_size"] = int(self.pending_population_size)
            self.pending_population_size = None
        if self.pending_world_size is not None:
            updates["world_size"] = int(self.pending_world_size)
            self.pending_world_size = None
        return updates

    def consume_restart_request(self) -> bool:
        requested = self.restart_requested
        self.restart_requested = False
        return requested

    def set_active_snapshot_generation(self, generation: int) -> None:
        self.active_snapshot_generation = generation

    def is_paused(self) -> bool:
        return self.paused

    def reset_episode_visuals(self) -> None:
        self.focus_agent_id = None
        self.focus_trail = []
        self._last_frame = 0.0

    def render_generation(
        self,
        generation: int,
        total_generations: int | None,
        step: int,
        creatures: list["Creature"],
        food: set[tuple[int, int]],
        best_fitness: float,
        avg_fitness: float,
        step_info: dict | None = None,
    ) -> None:
        self._handle_events()
        if self._should_stop:
            return
        if step % self.step_skip != 0:
            return

        now = time.monotonic()
        remaining = self.frame_delay - (now - self._last_frame)
        if remaining > 0:
            time.sleep(remaining)
            now += remaining
        self._last_frame = now

        step_info = step_info or {}
        obstacles = step_info.get("obstacles", set())
        predators = step_info.get("predators", [])
        food_remaining = int(step_info.get("food_remaining", len(food)))
        food_target = int(step_info.get("food_target", len(food)))
        alive_agents = int(step_info.get("alive_agents", len([c for c in creatures if c.energy > 0])))

        alive_creatures = [creature for creature in creatures if creature.energy > 0]
        if not alive_creatures:
            return

        ranked = sorted(alive_creatures, key=lambda creature: (creature.food_eaten, creature.energy), reverse=True)
        focus = ranked[0]
        if self.focus_agent_id != focus.agent_id:
            self.focus_agent_id = focus.agent_id
            self.focus_trail = []

        world_rect = self.pygame.Rect(self.margin, self.margin, self.world_px, self.world_px)
        sidebar_x = self.margin * 2 + self.world_px
        sidebar_rect = self.pygame.Rect(sidebar_x, self.margin, self.sidebar_width, self.world_px)

        self.screen.fill(self.colors["bg"])
        self.pygame.draw.rect(self.screen, self.colors["world"], world_rect)
        self.pygame.draw.rect(self.screen, self.colors["border"], world_rect, width=2)
        self.pygame.draw.rect(self.screen, (20, 23, 31), sidebar_rect, border_radius=6)
        self.pygame.draw.rect(self.screen, self.colors["border"], sidebar_rect, width=1, border_radius=6)

        for gx in range(WORLD_SIZE + 1):
            line_x = self.margin + gx * self.cell_size
            self.pygame.draw.line(self.screen, self.colors["grid"], (line_x, self.margin), (line_x, self.margin + self.world_px), 1)
        for gy in range(WORLD_SIZE + 1):
            line_y = self.margin + gy * self.cell_size
            self.pygame.draw.line(self.screen, self.colors["grid"], (self.margin, line_y), (self.margin + self.world_px, line_y), 1)

        for obstacle_x, obstacle_y in obstacles:
            ox = self.margin + obstacle_x * self.cell_size
            oy = self.margin + obstacle_y * self.cell_size
            self.pygame.draw.rect(
                self.screen,
                self.colors["obstacle"],
                self.pygame.Rect(ox + 2, oy + 2, self.cell_size - 4, self.cell_size - 4),
                border_radius=3,
            )

        for food_x, food_y in food:
            cx = self.margin + food_x * self.cell_size + self.cell_size // 2
            cy = self.margin + food_y * self.cell_size + self.cell_size // 2
            self.pygame.draw.circle(self.screen, self.colors["food"], (cx, cy), max(4, self.cell_size // 8))

        for creature in alive_creatures:
            cx = self.margin + creature.x * self.cell_size + self.cell_size // 2
            cy = self.margin + creature.y * self.cell_size + self.cell_size // 2
            self.pygame.draw.circle(self.screen, self.colors["creature"], (cx, cy), max(5, self.cell_size // 7))

        for predator_x, predator_y in predators:
            px = self.margin + predator_x * self.cell_size + self.cell_size // 2
            py = self.margin + predator_y * self.cell_size + self.cell_size // 2
            self.pygame.draw.circle(self.screen, self.colors["predator"], (px, py), max(5, self.cell_size // 7), width=2)

        fx = self.margin + focus.x * self.cell_size + self.cell_size // 2
        fy = self.margin + focus.y * self.cell_size + self.cell_size // 2
        self.focus_trail.append((fx, fy))
        if len(self.focus_trail) > 40:
            self.focus_trail = self.focus_trail[-40:]
        if len(self.focus_trail) >= 2:
            self.pygame.draw.lines(self.screen, self.colors["best"], False, self.focus_trail, 2)
        self.pygame.draw.circle(self.screen, self.colors["best"], (fx, fy), max(8, self.cell_size // 5), width=2)

        panel_x = sidebar_x + 12
        panel_y = self.margin + 12
        panel_y = self._draw_controls(panel_x, panel_y)
        panel_y = self._draw_snapshot_selector(panel_x, panel_y)
        panel_y = self._draw_food_coverage_selector(panel_x, panel_y)
        panel_y = self._draw_env_selector(
            "Steps/gen",
            self.steps_options,
            self.active_steps_per_generation,
            panel_x,
            panel_y,
            self.steps_button_rects,
        )
        panel_y = self._draw_env_selector(
            "Agents",
            self.population_options,
            self.active_population_size,
            panel_x,
            panel_y,
            self.population_button_rects,
        )
        panel_y = self._draw_env_selector(
            "Grid",
            self.grid_options,
            self.active_world_size,
            panel_x,
            panel_y,
            self.grid_button_rects,
        )
        panel_y = self._draw_legend(panel_x, panel_y)
        self._draw_learning_curve(panel_x, panel_y + 8, self.sidebar_width - 24, 130)

        hud_y = self.margin + self.world_px + 8
        total_label = "∞" if total_generations is None else str(total_generations)
        self._draw_text(
            f"Generation {generation}/{total_label} | Step {step + 1:03d}/{STEPS_PER_GENERATION:03d}",
            self.margin,
            hud_y,
            self.colors["text"],
        )
        self._draw_text(
            f"Best {best_fitness:6.2f} | Avg {avg_fitness:6.2f} | Food {food_remaining}/{food_target} | Alive {alive_agents}",
            self.margin,
            hud_y + 24,
            self.colors["muted"],
            small=True,
        )

        self._draw_text(
            f"FPS {self.training_fps:4.1f} | Init food {int(self.active_food_coverage * 100)}%",
            self.margin,
            hud_y + 44,
            self.colors["muted"],
            small=True,
        )

        if self.active_snapshot_generation is not None:
            self._draw_text(
                f"Active snapshot: G{self.active_snapshot_generation}",
                self.margin,
                hud_y + 62,
                self.colors["muted"],
                small=True,
            )

        if self.paused:
            paused_badge = self.pygame.Rect(self.margin + self.world_px - 112, self.margin + 10, 100, 28)
            self.pygame.draw.rect(self.screen, (100, 76, 40), paused_badge, border_radius=6)
            self.pygame.draw.rect(self.screen, self.colors["border"], paused_badge, width=1, border_radius=6)
            self._draw_text("PAUSED", paused_badge.x + 18, paused_badge.y + 5, self.colors["text"], small=True)

        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def should_stop(self) -> bool:
        return self._should_stop

    def hold_until_closed(self, message: str) -> None:
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return

        while not self._should_stop:
            self._handle_events()
            self._draw_text(message, self.margin, self.margin + self.world_px + 70, self.colors["muted"], small=True)
            self.pygame.display.flip()
            self.clock.tick(30)

    def close(self) -> None:
        self.pygame.quit()


def build_renderer(renderer_mode: str, fps: float, step_skip: int, fullscreen: bool = False) -> TerminalRenderer | PygameRenderer | None:
    if renderer_mode == "none":
        return None
    if renderer_mode == "terminal":
        return TerminalRenderer(fps=fps, step_skip=step_skip)
    if renderer_mode == "pygame":
        return PygameRenderer(fps=fps, step_skip=step_skip, fullscreen=fullscreen)
    raise ValueError(f"Unsupported renderer mode: {renderer_mode}")
