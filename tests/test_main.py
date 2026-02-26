import contextlib
import io
import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main


class TestCreatureDecisions(unittest.TestCase):
    def test_decide_move_uses_bias_to_choose_right(self) -> None:
        genes = [0.0] * 20
        genes[9] = 5.0
        creature = main.Creature(genes=genes, x=10, y=10)

        move = creature.decide_move(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(move, (1, 0))

    def test_decide_move_uses_food_vector_signal(self) -> None:
        genes = [0.0] * 20
        genes[0] = 1.0
        creature = main.Creature(genes=genes, x=10, y=10)

        move = creature.decide_move(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(move, (-1, 0))


class TestCoreFunctions(unittest.TestCase):
    def test_random_genes_have_expected_shape_and_range(self) -> None:
        genes = main.random_genes()
        self.assertEqual(len(genes), 20)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in genes))

    def test_spawn_food_returns_unique_points_in_bounds(self) -> None:
        random.seed(1)
        food = main.spawn_food(25)
        self.assertEqual(len(food), 25)
        self.assertTrue(all(0 <= x < main.WORLD_SIZE and 0 <= y < main.WORLD_SIZE for x, y in food))

    def test_nearest_food_vector_empty_returns_zero(self) -> None:
        dx, dy = main.nearest_food_vector(0, 0, set())
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_nearest_food_vector_points_to_closest_food(self) -> None:
        food = {(3, 4), (20, 20)}
        dx, dy = main.nearest_food_vector(0, 0, food)
        self.assertAlmostEqual(dx, 0.6, places=2)
        self.assertAlmostEqual(dy, 0.8, places=2)

    def test_wall_features_at_corners(self) -> None:
        fx1, fy1 = main.wall_features(0, 0)
        fx2, fy2 = main.wall_features(main.WORLD_SIZE - 1, main.WORLD_SIZE - 1)
        self.assertAlmostEqual(fx1, -1.0, places=6)
        self.assertAlmostEqual(fy1, -1.0, places=6)
        self.assertAlmostEqual(fx2, 1.0, places=6)
        self.assertAlmostEqual(fy2, 1.0, places=6)

    def test_level_for_generation_thresholds(self) -> None:
        self.assertEqual(main.level_for_generation(1), 1)
        self.assertEqual(main.level_for_generation(20), 1)
        self.assertEqual(main.level_for_generation(21), 2)
        self.assertEqual(main.level_for_generation(120), 2)
        self.assertEqual(main.level_for_generation(121), 3)

    def test_should_stop_early_plateau(self) -> None:
        best_history = [10.0 + (0.2 if i % 2 == 0 else 0.0) for i in range(30)]
        avg_history = [7.5 + (0.1 if i % 2 == 0 else 0.0) for i in range(30)]
        should_stop, reason = main.should_stop_early(best_history, avg_history)
        self.assertTrue(should_stop)
        self.assertEqual(reason, "plateau")

    def test_should_stop_early_not_enough_history(self) -> None:
        should_stop, reason = main.should_stop_early([1.0, 2.0, 3.0], [1.0, 1.5, 2.0])
        self.assertFalse(should_stop)
        self.assertEqual(reason, "")

    def test_simulate_task_once_is_deterministic_for_seed(self) -> None:
        genes = [0.1] * 20
        score_a = main.simulate_task_once(genes, "forage_basic", 123)
        score_b = main.simulate_task_once(genes, "forage_basic", 123)
        self.assertEqual(score_a, score_b)

    def test_evaluate_checkpoint_benchmark_has_expected_keys(self) -> None:
        genes = [0.0] * 20
        result = main.evaluate_checkpoint_benchmark(genes, checkpoint_generation=0, trials=2, use_parallel=False)
        self.assertIn("overall", result)
        self.assertIn("forage_basic", result)
        self.assertIn("forage_maze", result)
        self.assertIn("predator_escape", result)

    def test_select_initial_champion_returns_member(self) -> None:
        population = [[0.0] * 20, [0.2] * 20, [-0.1] * 20]
        champion = main.select_initial_champion(population)
        self.assertIn(champion, population)

    def test_advance_onboarding_state_next_and_finish(self) -> None:
        step, active = main.advance_onboarding_state(0, 4, "next")
        self.assertEqual(step, 1)
        self.assertTrue(active)

        step, active = main.advance_onboarding_state(3, 4, "next")
        self.assertEqual(step, 3)
        self.assertFalse(active)

    def test_advance_onboarding_state_back_and_close(self) -> None:
        step, active = main.advance_onboarding_state(2, 4, "back")
        self.assertEqual(step, 1)
        self.assertTrue(active)

        step, active = main.advance_onboarding_state(1, 4, "close")
        self.assertEqual(step, 1)
        self.assertFalse(active)

    def test_checkpoint_resume_policy(self) -> None:
        self.assertFalse(main.checkpoint_requires_resume(0, onboarding_active=False))
        self.assertTrue(main.checkpoint_requires_resume(100, onboarding_active=False))
        self.assertFalse(main.checkpoint_requires_resume(100, onboarding_active=True))
        self.assertTrue(main.checkpoint_should_defer(100, onboarding_active=True))
        self.assertFalse(main.checkpoint_should_defer(0, onboarding_active=True))

    def test_resolve_early_stop_mode_defaults(self) -> None:
        self.assertEqual(main.resolve_early_stop_mode("pygame", None), "warn")
        self.assertEqual(main.resolve_early_stop_mode("terminal", None), "auto")

    def test_resolve_early_stop_mode_explicit(self) -> None:
        self.assertEqual(main.resolve_early_stop_mode("pygame", "off"), "off")
        self.assertEqual(main.resolve_early_stop_mode("terminal", "warn"), "warn")

    def test_mutation_schedule_decreases_over_time(self) -> None:
        r1, s1 = main.mutation_schedule(1)
        r2, s2 = main.mutation_schedule(100)
        r3, s3 = main.mutation_schedule(200)
        self.assertGreater(r1, r2)
        self.assertGreater(r2, r3)
        self.assertGreater(s1, s2)
        self.assertGreater(s2, s3)

    def test_should_rollback_to_best_threshold(self) -> None:
        history = [10.0] * 20
        self.assertTrue(main.should_rollback_to_best(history, best_avg_seen=15.0))
        self.assertFalse(main.should_rollback_to_best(history, best_avg_seen=13.0))

    def test_playback_fps_for_steps(self) -> None:
        self.assertEqual(main.playback_fps_for_steps(120, 30), 4.0)
        self.assertEqual(main.playback_fps_for_steps(120, 0), 1.0)

    def test_metabolic_drain_increases_over_generation(self) -> None:
        start = main.metabolic_drain_for_step(0, 120)
        mid = main.metabolic_drain_for_step(60, 120)
        end = main.metabolic_drain_for_step(119, 120)
        self.assertLess(start, mid)
        self.assertLess(mid, end)

    def test_creature_fitness_prioritizes_survival_then_food(self) -> None:
        survival_creature = main.Creature(genes=[0.0] * 20, x=0, y=0, survival_steps=90, food_eaten=2, energy=1.0)
        food_only_creature = main.Creature(genes=[0.0] * 20, x=0, y=0, survival_steps=20, food_eaten=8, energy=1.0)
        self.assertGreater(main.creature_fitness(survival_creature), main.creature_fitness(food_only_creature))

    def test_run_checkpoint_benchmark_job_shape(self) -> None:
        generation, result = main.run_checkpoint_benchmark_job([0.0] * 20, 7)
        self.assertEqual(generation, 7)
        self.assertIn("overall", result)

    def test_compute_progress_signal_warmup(self) -> None:
        signal = main.compute_progress_signal([1.0, 2.0], [0.5, 1.0], window=5)
        self.assertFalse(signal["ready"])
        self.assertEqual(signal["score"], 0.0)

    def test_compute_progress_signal_ready(self) -> None:
        best = [float(10 + i) for i in range(20)]
        avg = [float(6 + i * 0.7) for i in range(20)]
        signal = main.compute_progress_signal(best, avg, window=20)
        self.assertTrue(signal["ready"])
        self.assertGreater(signal["avg_delta"], 0.0)

    def test_adapt_curriculum_level_promote_and_relax(self) -> None:
        promote = {"ready": True, "avg_delta": 5.0, "best_delta": 6.0, "stability": 4.0, "score": 20.0}
        relax = {"ready": True, "avg_delta": 0.4, "best_delta": 1.0, "stability": 14.0, "score": 8.0}
        self.assertEqual(main.adapt_curriculum_level(1, promote), 2)
        self.assertEqual(main.adapt_curriculum_level(3, relax), 2)

    def test_parse_checkpoint_generations(self) -> None:
        parsed = main.parse_checkpoint_generations("100, 500,1000,2000,500")
        self.assertEqual(parsed, [100, 500, 1000, 2000])

    def test_snapshot_file_roundtrip(self) -> None:
        snapshots = {
            100: {
                "best_fitness": 42.0,
                "avg_fitness": 20.0,
                "best_food": 4,
                "level": 2,
                "best_genes": [0.1] * 20,
                "top_genes": [[0.1] * 20, [0.2] * 20],
            }
        }
        training_curve = {
            "generations": [1, 2, 3],
            "avg_fitness": [5.0, 7.0, 8.0],
            "max_fitness": [8.0, 11.0, 13.0],
            "min_fitness": [2.0, 3.0, 4.0],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = pathlib.Path(tmp_dir) / "snapshots.json"
            main.save_snapshot_file(
                str(target),
                snapshots,
                total_generations=2000,
                seed=42,
                training_curve=training_curve,
            )
            loaded_snapshots, metadata = main.load_snapshot_file(str(target))

        self.assertIn(100, loaded_snapshots)
        self.assertEqual(metadata["total_generations"], 2000)
        self.assertEqual(metadata["seed"], 42)
        self.assertIn("training_curve", metadata)
        self.assertEqual(metadata["training_curve"]["generations"], [1, 2, 3])
        self.assertEqual(metadata["training_curve"]["avg_fitness"], [5.0, 7.0, 8.0])

    def test_select_default_snapshot_generation_prefers_latest(self) -> None:
        snapshots = {
            100: {"avg_fitness": 21.0},
            400: {"avg_fitness": 36.5},
            1000: {"avg_fitness": 31.2},
        }
        selected = main.core.select_default_snapshot_generation(snapshots)
        self.assertEqual(selected, 1000)

    def test_select_default_snapshot_generation_falls_back_to_latest(self) -> None:
        snapshots = {
            100: {},
            400: {},
            1000: {},
        }
        selected = main.core.select_default_snapshot_generation(snapshots)
        self.assertEqual(selected, 1000)

    def test_build_serve_population_best_clones(self) -> None:
        snapshot = {
            "best_genes": [0.3] * 20,
            "top_genes": [[0.1] * 20, [0.2] * 20],
        }
        population = main.core.build_serve_population(snapshot, target_population_size=50, serve_best_clones=10)
        self.assertEqual(len(population), 10)
        self.assertTrue(all(genes == [0.3] * 20 for genes in population))

    def test_build_serve_population_mixed_top_genes_target_size(self) -> None:
        snapshot = {
            "top_genes": [[0.1] * 20, [0.2] * 20],
        }
        random.seed(42)
        population = main.core.build_serve_population(snapshot, target_population_size=7, serve_best_clones=0)
        self.assertEqual(len(population), 7)
        self.assertTrue(all(len(genes) == 20 for genes in population))

    def test_blocker_probability_from_genes_monotonic(self) -> None:
        low = main.core.blocker_probability_from_genes([-1.0] * 20)
        high_genes = [0.0] * 20
        high_genes[19] = 1.0
        high = main.core.blocker_probability_from_genes(high_genes)
        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_choose_blocker_target_cell_returns_valid_cell(self) -> None:
        actor = main.Creature(genes=[0.0] * 20, x=5, y=5, agent_id=1)
        rival = main.Creature(genes=[0.0] * 20, x=2, y=2, agent_id=2)
        with mock.patch.object(main.core, "WORLD_SIZE", 10):
            target = main.core.choose_blocker_target_cell(
                actor,
                rivals=[rival],
                food={(4, 2)},
                blocked={(2, 2)},
            )
        self.assertIsNotNone(target)
        tx, ty = target
        self.assertGreaterEqual(tx, 0)
        self.assertLess(tx, 10)
        self.assertGreaterEqual(ty, 0)
        self.assertLess(ty, 10)


class TestEvolutionOperators(unittest.TestCase):
    def test_crossover_selects_from_each_parent(self) -> None:
        parent_a = [1.0, 1.0, 1.0, 1.0]
        parent_b = [2.0, 2.0, 2.0, 2.0]

        with mock.patch("main.random.random", side_effect=[0.1, 0.9, 0.2, 0.8]):
            child = main.crossover(parent_a, parent_b)

        self.assertEqual(child, [1.0, 2.0, 1.0, 2.0])

    def test_mutate_no_change_when_rate_zero(self) -> None:
        genes = [0.5, -0.5, 0.0]
        with mock.patch.object(main, "MUTATION_RATE", 0.0):
            mutated = main.mutate(genes)
        self.assertEqual(mutated, genes)

    def test_mutate_updates_all_genes_when_rate_one(self) -> None:
        genes = [0.0, 0.0, 0.0]
        with (
            mock.patch.object(main, "MUTATION_RATE", 1.0),
            mock.patch.object(main, "MUTATION_STRENGTH", 0.5),
            mock.patch("main.random.uniform", return_value=0.2),
        ):
            mutated = main.mutate(genes)

        self.assertEqual(mutated, [0.2, 0.2, 0.2])


class TestGenerationAndLoop(unittest.TestCase):
    def test_quadrant_for_step_clockwise_sequence(self) -> None:
        self.assertEqual(main.core.quadrant_for_step(0, 5), "bottom-right")
        self.assertEqual(main.core.quadrant_for_step(5, 5), "bottom-left")
        self.assertEqual(main.core.quadrant_for_step(10, 5), "top-left")
        self.assertEqual(main.core.quadrant_for_step(15, 5), "top-right")
        self.assertEqual(main.core.quadrant_for_step(20, 5), "bottom-right")

    def test_replenish_food_clockwise_quadrants_respects_active_zone(self) -> None:
        with mock.patch.object(main.core, "WORLD_SIZE", 20):
            for step, expected_zone in [(0, "bottom-right"), (5, "bottom-left"), (10, "top-left"), (15, "top-right")]:
                food: set[tuple[int, int]] = set()
                random.seed(21 + step)
                main.core.replenish_food(
                    food,
                    target_count=20,
                    respawn_mode="clockwise-quadrants",
                    batch_count=20,
                    step=step,
                    cycle_steps=5,
                )

                if expected_zone == "bottom-right":
                    self.assertTrue(all(x >= 10 and y >= 10 for x, y in food))
                elif expected_zone == "bottom-left":
                    self.assertTrue(all(x < 10 and y >= 10 for x, y in food))
                elif expected_zone == "top-left":
                    self.assertTrue(all(x < 10 and y < 10 for x, y in food))
                else:
                    self.assertTrue(all(x >= 10 and y < 10 for x, y in food))

    def test_replenish_food_right_side_spawns_on_right_half(self) -> None:
        food: set[tuple[int, int]] = set()
        with mock.patch.object(main.core, "WORLD_SIZE", 20):
            random.seed(17)
            main.core.replenish_food(food, target_count=20, respawn_mode="right-side", batch_count=20)

        self.assertEqual(len(food), 20)
        self.assertTrue(all(x >= 10 for x, _ in food))

    def test_run_generation_continues_with_respawn_when_initial_food_empty(self) -> None:
        population = [[0.0] * 20 for _ in range(4)]
        seen_steps: list[int] = []

        def callback(step: int, creatures: list[main.Creature], food: set[tuple[int, int]]) -> None:
            _ = (creatures, food)
            seen_steps.append(step)

        with (
            mock.patch.object(main, "WORLD_SIZE", 8),
            mock.patch.object(main, "FOOD_COUNT", 6),
            mock.patch.object(main, "STEPS_PER_GENERATION", 6),
            mock.patch.object(main.core, "spawn_food_with_coverage", return_value=set()),
        ):
            random.seed(19)
            main.run_generation(
                population,
                step_callback=callback,
                food_respawn_mode="right-side",
                food_respawn_interval_steps=1,
                food_respawn_batch_ratio=0.5,
            )

        self.assertEqual(len(seen_steps), 6)

    def test_run_generation_clockwise_clears_food_between_quadrants(self) -> None:
        population = [[0.0] * 20 for _ in range(4)]
        food_snapshots: list[tuple[int, str, set[tuple[int, int]]]] = []

        def callback(
            step: int,
            creatures: list[main.Creature],
            food: set[tuple[int, int]],
            step_info: dict,
        ) -> None:
            _ = creatures
            quadrant = str(step_info.get("food_respawn_quadrant", ""))
            food_snapshots.append((step, quadrant, set(food)))

        with (
            mock.patch.object(main, "WORLD_SIZE", 8),
            mock.patch.object(main, "FOOD_COUNT", 8),
            mock.patch.object(main, "STEPS_PER_GENERATION", 5),
        ):
            random.seed(23)
            main.run_generation(
                population,
                step_callback=callback,
                food_respawn_mode="clockwise-quadrants",
                food_respawn_interval_steps=1,
                food_respawn_batch_ratio=1.0,
                food_respawn_cycle_steps=2,
            )

        self.assertGreaterEqual(len(food_snapshots), 4)

        for _, quadrant, food in food_snapshots:
            if not food:
                continue
            if quadrant == "bottom-right":
                self.assertTrue(all(x >= 4 and y >= 4 for x, y in food))
            elif quadrant == "bottom-left":
                self.assertTrue(all(x < 4 and y >= 4 for x, y in food))
            elif quadrant == "top-left":
                self.assertTrue(all(x < 4 and y < 4 for x, y in food))
            elif quadrant == "top-right":
                self.assertTrue(all(x >= 4 and y < 4 for x, y in food))

    def test_run_generation_corners_spawn_starts_near_corners(self) -> None:
        with mock.patch.object(main.core, "WORLD_SIZE", 12):
            positions = main.core.spawn_positions_for_population(8, "corners")

        corners = {(0, 0), (11, 0), (0, 11), (11, 11)}
        max_corner_distance = max(
            min(abs(x - cx) + abs(y - cy) for cx, cy in corners)
            for x, y in positions
        )
        self.assertLessEqual(max_corner_distance, 1)

    def test_estimate_generation_steps_ramps_to_target(self) -> None:
        self.assertEqual(main.core.estimate_generation_steps(1, 120, 300, 100), 120)
        self.assertEqual(main.core.estimate_generation_steps(100, 120, 300, 100), 300)
        self.assertEqual(main.core.estimate_generation_steps(50, 120, 300, 100), 209)

    def test_spawn_positions_biased_corners_biases_toward_edges(self) -> None:
        with mock.patch.object(main.core, "WORLD_SIZE", 20):
            random.seed(31)
            positions = main.core.spawn_positions_for_population(60, "biased-corners")

        corners = {(0, 0), (19, 0), (0, 19), (19, 19)}
        avg_corner_distance = sum(
            min(abs(x - cx) + abs(y - cy) for cx, cy in corners)
            for x, y in positions
        ) / max(1, len(positions))
        self.assertLess(avg_corner_distance, 7.0)

    def test_run_generation_returns_sorted_scores(self) -> None:
        population = [[0.0] * 20 for _ in range(6)]

        with (
            mock.patch.object(main, "WORLD_SIZE", 8),
            mock.patch.object(main, "FOOD_COUNT", 6),
            mock.patch.object(main, "STEPS_PER_GENERATION", 12),
        ):
            random.seed(2)
            scored = main.run_generation(population)

        self.assertEqual(len(scored), len(population))
        self.assertTrue(all(len(item[1]) == 20 for item in scored))
        self.assertTrue(all(isinstance(item[2], int) for item in scored))
        self.assertTrue(all(scored[i][0] >= scored[i + 1][0] for i in range(len(scored) - 1)))

    def test_run_generation_invokes_step_callback(self) -> None:
        population = [[0.0] * 20 for _ in range(4)]
        seen_steps: list[int] = []

        def callback(step: int, creatures: list[main.Creature], food: set[tuple[int, int]]) -> None:
            seen_steps.append(step)
            self.assertEqual(len(creatures), len(population))
            self.assertIsInstance(food, set)

        with (
            mock.patch.object(main, "WORLD_SIZE", 8),
            mock.patch.object(main, "FOOD_COUNT", 6),
            mock.patch.object(main, "STEPS_PER_GENERATION", 7),
        ):
            random.seed(7)
            main.run_generation(population, step_callback=callback)

        self.assertEqual(len(seen_steps), 7)
        self.assertEqual(seen_steps[0], 0)
        self.assertEqual(seen_steps[-1], 6)

    def test_run_generation_callback_with_step_info(self) -> None:
        population = [[0.0] * 20 for _ in range(4)]
        seen_levels: list[int] = []

        def callback(
            step: int,
            creatures: list[main.Creature],
            food: set[tuple[int, int]],
            step_info: dict,
        ) -> None:
            _ = (step, creatures, food)
            seen_levels.append(step_info["level"])
            self.assertIn("species_counts", step_info)
            self.assertIn("event_counts", step_info)

        with (
            mock.patch.object(main, "WORLD_SIZE", 8),
            mock.patch.object(main, "FOOD_COUNT", 6),
            mock.patch.object(main, "STEPS_PER_GENERATION", 5),
        ):
            random.seed(9)
            main.run_generation(population, step_callback=callback, level=2)

        self.assertEqual(len(seen_levels), 5)
        self.assertTrue(all(level == 2 for level in seen_levels))

    def test_evolve_smoke_prints_generation_line(self) -> None:
        buffer = io.StringIO()
        with (
            mock.patch.object(main, "POPULATION_SIZE", 8),
            mock.patch.object(main, "GENERATIONS", 1),
            mock.patch.object(main, "STEPS_PER_GENERATION", 5),
            mock.patch.object(main, "FOOD_COUNT", 5),
            mock.patch.object(main, "ELITE_COUNT", 2),
            contextlib.redirect_stdout(buffer),
        ):
            random.seed(3)
            main.evolve()

        output = buffer.getvalue()
        self.assertIn("Gen 01", output)
        self.assertIn("Done.", output)

    def test_evolve_render_mode_smoke(self) -> None:
        with (
            mock.patch.object(main, "POPULATION_SIZE", 6),
            mock.patch.object(main, "GENERATIONS", 1),
            mock.patch.object(main, "STEPS_PER_GENERATION", 4),
            mock.patch.object(main, "FOOD_COUNT", 4),
            mock.patch.object(main, "ELITE_COUNT", 2),
            mock.patch("main.time.monotonic", side_effect=[0.0, 1.0, 2.0, 3.0, 4.0]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            random.seed(5)
            main.evolve(render=True, fps=60.0, render_step_skip=1, generations=1, seed=5)


class TestRendererSelection(unittest.TestCase):
    def test_build_renderer_none_returns_none(self) -> None:
        renderer = main.build_renderer("none", fps=20.0, step_skip=1)
        self.assertIsNone(renderer)

    def test_build_renderer_terminal_returns_terminal_renderer(self) -> None:
        renderer = main.build_renderer("terminal", fps=20.0, step_skip=1)
        self.assertIsInstance(renderer, main.TerminalRenderer)


class TestServeSnapshotInteractions(unittest.TestCase):
    def test_serve_handles_partial_environment_update_without_crash(self) -> None:
        class FakePygameRenderer:
            def __init__(self) -> None:
                self._stop = False
                self._env_calls = 0
                self.onboarding_active = False
                self.checkpoint_resume_required = False

            def should_stop(self) -> bool:
                return self._stop

            def poll_events(self) -> None:
                return None

            def close(self) -> None:
                return None

            def hold_until_closed(self, message: str) -> None:
                _ = message
                return None

            def set_snapshot_choices(self, generations: list[int], active_generation: int) -> None:
                _ = (generations, active_generation)

            def set_active_food_coverage(self, coverage: float) -> None:
                _ = coverage

            def set_active_environment(self, world_size: int, population_size: int, steps_per_generation: int) -> None:
                _ = (world_size, population_size, steps_per_generation)

            def set_showcase_profile(self, is_showcase: bool, total_steps: int) -> None:
                _ = (is_showcase, total_steps)

            def set_training_curve(
                self,
                generations: list[int],
                avg_fitness: list[float],
                max_fitness: list[float],
                min_fitness: list[float],
            ) -> None:
                _ = (generations, avg_fitness, max_fitness, min_fitness)

            def reset_episode_visuals(self) -> None:
                return None

            def consume_snapshot_request(self) -> int | None:
                return None

            def consume_food_coverage_request(self) -> float | None:
                return None

            def consume_environment_request(self) -> dict[str, int]:
                self._env_calls += 1
                if self._env_calls == 2:
                    return {"steps_per_generation": 60}
                return {}

            def consume_restart_request(self) -> bool:
                return False

            def render_generation(self, *args, **kwargs) -> None:
                _ = (args, kwargs)
                return None

            def record_generation_result(self, generation: int, best_fitness: float, avg_fitness: float) -> None:
                _ = (generation, best_fitness, avg_fitness)

            def set_active_snapshot_generation(self, generation: int) -> None:
                _ = generation

            def is_paused(self) -> bool:
                return False

        fake_renderer = FakePygameRenderer()

        def fake_run_generation(population, step_callback, level, food_coverage_init, **kwargs):
            _ = (level, food_coverage_init, kwargs)
            creatures = [main.Creature(genes=population[0], x=0, y=0)]
            step_callback(
                0,
                creatures,
                {(0, 0)},
                {
                    "obstacles": set(),
                    "predators": [],
                    "food_remaining": 1,
                    "food_target": 1,
                    "alive_agents": 1,
                },
            )
            fake_renderer._stop = True
            return [(1.0, population[0], 0)]

        snapshots = {
            100: {
                "best_fitness": 1.0,
                "avg_fitness": 1.0,
                "best_food": 0,
                "level": 1,
                "best_genes": [0.0] * 20,
                "top_genes": [[0.0] * 20],
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = pathlib.Path(tmp_dir) / "snapshots.json"
            main.save_snapshot_file(str(target), snapshots, total_generations=100, seed=42)

            with (
                mock.patch.object(main.core, "PygameRenderer", FakePygameRenderer),
                mock.patch.object(main.core, "build_renderer", return_value=fake_renderer),
                mock.patch.object(main.core, "run_generation", side_effect=fake_run_generation),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main.serve_saved_snapshots(
                    checkpoint_file=str(target),
                    renderer_mode="pygame",
                    fps=1.0,
                    render_step_skip=1,
                    fullscreen=False,
                    food_coverage_init=1.0,
                )

        self.assertGreaterEqual(fake_renderer._env_calls, 2)

    def test_build_snapshot_schedule_has_multiple_choices(self) -> None:
        schedule = main.core.build_snapshot_schedule(4000)
        self.assertIn(4000, schedule)
        self.assertGreaterEqual(len(schedule), 6)
        self.assertEqual(schedule, sorted(schedule))


if __name__ == "__main__":
    unittest.main(verbosity=2)
