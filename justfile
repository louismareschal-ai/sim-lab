set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

venv_python := ".venv/bin/python"
game_args := env_var_or_default("GAME_ARGS", "")
train_generations := env_var_or_default("TRAIN_GENERATIONS", "4000")
food_coverage_init := env_var_or_default("FOOD_COVERAGE_INIT", "100")
snapshot_generations := env_var_or_default("SNAPSHOT_GENERATIONS", "100,200,300,400,500,1000,2000,3000,4000")
clockwise_snapshot_file := "models/snapshots_clockwise.json"

default:
    @just --list

setup:
    cd "{{justfile_directory()}}" && (test -x {{venv_python}} || python3 -m venv .venv)
    cd "{{justfile_directory()}}" && {{venv_python}} -m pip install -r requirements.txt

run: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py {{game_args}}

render: train-and-render

start: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --interactive-start --renderer pygame --fps 1 --render-step-skip 1 --early-stop off

test: setup
    cd "{{justfile_directory()}}" && {{venv_python}} -m unittest discover -s tests -p test_main.py -v

check: setup
    cd "{{justfile_directory()}}" && PYTHON_BIN={{venv_python}} bash check.sh

smoke: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --generations 2

scenario-base: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --renderer pygame --early-stop off --spawn-mode random --steps-growth-final 120 --steps-growth-generations 1 --food-coverage-init {{food_coverage_init}} --log-interval 20

scenario-evolve: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --renderer pygame --early-stop off --spawn-mode biased-corners --steps-growth-final 320 --steps-growth-generations 500 --food-coverage-init {{food_coverage_init}} --log-interval 20

scenario-tuned-staged: setup
    cd "{{justfile_directory()}}" && mkdir -p runs
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_tuned_staged.json --log-interval 100 --food-coverage-init 70 --spawn-mode random --steps-growth-final 280 --steps-growth-generations 700 --food-respawn-mode random --food-respawn-interval-steps 12 --food-respawn-batch-ratio 0.20 | tee runs/sweep_tuned_staged.log

scenario-progressive-peak: setup
    cd "{{justfile_directory()}}" && mkdir -p runs
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_progressive.json --log-interval 100 --food-coverage-init 35 --spawn-mode biased-corners --steps-growth-final 320 --steps-growth-generations 500 --food-respawn-mode right-side --food-respawn-interval-steps 10 --food-respawn-batch-ratio 0.18 | tee runs/sweep_progressive.log

sweep-local: setup
    cd "{{justfile_directory()}}" && mkdir -p runs
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_easy.json --log-interval 100 --food-coverage-init 100 --spawn-mode random --steps-growth-final 120 --steps-growth-generations 1 --food-respawn-mode random --food-respawn-interval-steps 10 --food-respawn-batch-ratio 0.20 | tee runs/sweep_easy.log
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_progressive.json --log-interval 100 --food-coverage-init 35 --spawn-mode biased-corners --steps-growth-final 320 --steps-growth-generations 500 --food-respawn-mode right-side --food-respawn-interval-steps 10 --food-respawn-batch-ratio 0.18 | tee runs/sweep_progressive.log
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_gentle_progressive.json --log-interval 100 --food-coverage-init 60 --spawn-mode biased-corners --steps-growth-final 220 --steps-growth-generations 700 --food-respawn-mode random --food-respawn-interval-steps 12 --food-respawn-batch-ratio 0.22 | tee runs/sweep_gentle_progressive.log
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_tuned_staged.json --log-interval 100 --food-coverage-init 70 --spawn-mode random --steps-growth-final 280 --steps-growth-generations 700 --food-respawn-mode random --food-respawn-interval-steps 12 --food-respawn-batch-ratio 0.20 | tee runs/sweep_tuned_staged.log

experience: setup
    cd "{{justfile_directory()}}" && (test -f models/sweep_tuned_staged.json || {{venv_python}} main.py --train-snapshots --train-generations 800 --snapshot-generations 200,400,800 --snapshot-file models/sweep_tuned_staged.json --log-interval 100 --food-coverage-init 70 --spawn-mode random --steps-growth-final 280 --steps-growth-generations 700 --food-respawn-mode random --food-respawn-interval-steps 12 --food-respawn-batch-ratio 0.20)
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file models/sweep_tuned_staged.json --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-coverage-init 70 --spawn-mode random --food-respawn-mode random --food-respawn-interval-steps 12 --food-respawn-batch-ratio 0.20

scenario-corners: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --renderer pygame --early-stop off --spawn-mode corners --steps-growth-final 360 --steps-growth-generations 600 --food-coverage-init {{food_coverage_init}} --log-interval 20

scenario-clockwise: setup
    cd "{{justfile_directory()}}" && (test -f {{clockwise_snapshot_file}} || {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file {{clockwise_snapshot_file}} --log-interval 50 --food-coverage-init {{food_coverage_init}} --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1)
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file {{clockwise_snapshot_file}} --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --food-coverage-init {{food_coverage_init}}

scenario-clockwise-force: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file {{clockwise_snapshot_file}} --log-interval 50 --food-coverage-init {{food_coverage_init}} --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file {{clockwise_snapshot_file}} --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --food-coverage-init {{food_coverage_init}}

scenario-clockwise-best10: setup
    cd "{{justfile_directory()}}" && (test -f {{clockwise_snapshot_file}} || {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file {{clockwise_snapshot_file}} --log-interval 50 --food-coverage-init {{food_coverage_init}} --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1)
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --serve-best-clones 10 --snapshot-file {{clockwise_snapshot_file}} --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --food-coverage-init {{food_coverage_init}}

scenario-clockwise-blockers: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file {{clockwise_snapshot_file}} --log-interval 50 --food-coverage-init {{food_coverage_init}} --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --interference-mode simple-block --blocker-ttl-steps 4 --blocker-energy-cost 0.25
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file {{clockwise_snapshot_file}} --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --interference-mode simple-block --blocker-ttl-steps 4 --blocker-energy-cost 0.25 --food-coverage-init {{food_coverage_init}}

scenario-clockwise-best10-blockers: setup
    cd "{{justfile_directory()}}" && (test -f {{clockwise_snapshot_file}} || {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file {{clockwise_snapshot_file}} --log-interval 50 --food-coverage-init {{food_coverage_init}} --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --interference-mode simple-block --blocker-ttl-steps 4 --blocker-energy-cost 0.25)
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --serve-best-clones 10 --snapshot-file {{clockwise_snapshot_file}} --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-respawn-mode clockwise-quadrants --food-respawn-interval-steps 6 --food-respawn-batch-ratio 0.25 --food-respawn-cycle-steps 1 --interference-mode simple-block --blocker-ttl-steps 4 --blocker-energy-cost 0.25 --food-coverage-init {{food_coverage_init}}

soak-1000: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --renderer none --generations 1000 --early-stop off --log-interval 50

train-2000: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file models/snapshots.json --log-interval 50 --food-coverage-init {{food_coverage_init}}

serve-models: setup
    cd "{{justfile_directory()}}" && test -f models/snapshots.json || (echo "Missing models/snapshots.json. Run: just train-2000" && exit 1)
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file models/snapshots.json --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-coverage-init {{food_coverage_init}}

train-and-render: setup
    cd "{{justfile_directory()}}" && (test -f models/snapshots.json || {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file models/snapshots.json --log-interval 50 --food-coverage-init {{food_coverage_init}})
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file models/snapshots.json --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-coverage-init {{food_coverage_init}}

train-and-render-force: setup
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --train-snapshots --train-generations {{train_generations}} --snapshot-generations {{snapshot_generations}} --snapshot-file models/snapshots.json --log-interval 50 --food-coverage-init {{food_coverage_init}}
    cd "{{justfile_directory()}}" && {{venv_python}} main.py --serve-snapshots --snapshot-file models/snapshots.json --renderer pygame --fps 1 --render-step-skip 1 --showcase-interval-generations 1 --early-stop off --food-coverage-init {{food_coverage_init}}

ci: test smoke
