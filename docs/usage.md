# Usage

Run the CLI with one or more experiment configuration files:

```sh
run_experiment <command> <config.yaml> [<config2.yaml> ...]
```

Equivalent module invocation:

```sh
python -m regret <command> <config.yaml> [<config2.yaml> ...]
```

## Commands

- `validate <config...>`: validate schema and semantic constraints
- `plan <config...>`: print experiment plan without execution
- `run <config...> [--no-plot]`: execute experiments and optionally skip plotting
- `analyze <config...>`: regenerate plots from saved results

## Examples

```sh
run_experiment validate configs/experiments/01_baseline.yaml
run_experiment plan configs/experiments/01_baseline.yaml
run_experiment run configs/experiments/01_baseline.yaml --no-plot
run_experiment analyze configs/experiments/01_baseline.yaml
```
