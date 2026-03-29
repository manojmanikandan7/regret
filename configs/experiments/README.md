# Experiment Configuration Guide

This directory contains YAML configuration files for running benchmarking experiments with Regret.

## Quick Start

```bash
# Validate a config before running
uv run run_experiment validate configs/experiments/01_baseline.yml

# Show execution plan without running
uv run run_experiment plan configs/experiments/01_baseline.yml

# Run the experiment
uv run run_experiment run configs/experiments/01_baseline.yml

# Regenerate plots from existing results
uv run run_experiment analyze configs/experiments/01_baseline.yml
```

To create a new experiment, copy `template_boilerplate.yml` and modify it.

## Configuration Structure

Each config file has four main sections:

### `suite` - Experiment Settings

```yaml
suite:
  name: "my_experiment"           # Unique identifier
  description: "What this tests"  # Purpose of the experiment
  runs: 30                        # Independent repetitions per (algorithm, problem, budget)
  mode: "full"                    # "lite" (faster) or "full" (complete)
  parallel: true                  # Run repetitions in parallel
  profile: false                  # Enable runtime profiling (collects trajectory data)
  budgets: [500, 1000, 2000]      # Evaluation budgets to test
  output:
    raw_root: "results/raw"       # Where to save JSON results
    figures_root: "results/figures"  # Where to save plots
```

### `problems` - Benchmark Functions

```yaml
problems:
  - name: "OneMax-100"            # Display name
    class: "OneMax"               # Must exist in PROBLEM_REGISTRY
    budget_for_plots: 1000        # Which budget to use for history plots (must be in suite.budgets)
    params:
      n: 100                      # Problem-specific parameters
```

**Available problem classes:**
- `OneMax`, `LeadingOnes`, `TwoMax`, `BinVal` - Basic pseudo-boolean
- `Jump`, `Trap`, `Plateau` - Deceptive/multimodal (require `k` parameter)
- `HIFF` - Hierarchical (requires `levels` parameter)
- `NKLandscape` - Tunable epistasis (requires `n`, `k`, `seed`)
- `MaxkSAT`, `PetersenColoringMaxSAT` - Satisfiability

### `algorithms` - Optimizers

```yaml
algorithms:
  - name: "RLS"                   # Display name
    class: "RLS"                  # Must exist in ALGORITHM_REGISTRY
    args:
      defaults: {}                # Default parameters for all problems
      by_problem:                 # Problem-specific overrides
        "Jump-5":
          some_param: value
```

**Available algorithm classes:**
- `RLS`, `RLSExploration` - Local search
- `OnePlusOneEA` - Evolutionary (1+1)
- `MuPlusLambdaEA` - Evolutionary (μ+λ, requires `mu`, `lambda_`)
- `SimulatedAnnealing` - SA with configurable cooling

**Simulated Annealing cooling schedules:**
```yaml
args:
  defaults:
    T_func:
      type: "logarithmic"    # or "linear", "exponential"
      params:
        d: 2.0               # Schedule-specific parameter
```

### `plotting` - Visualization

```yaml
plotting:
  enabled: true                   # Master switch (required)
  budget_for_plots: 1000          # Global default (optional)
  layout:
    per_problem_dir: true         # Organize by problem
    include_n_subdir: true        # Include problem size in path
    structure:
      aggregate: "aggregate"
      history: "history"
      distribution: "distributions"
      profile: "profiles"
  titles:
    include_problem_name: true
    include_problem_size: true
  plots:
    # Include only the plots you want (see below)
```

## Plot Configuration

**Omitted plots are disabled by default.** You only need to include sections for plots you want to generate. There's no need to write `enabled: false` for plots you don't want.

### Available Plots

**Aggregate plots** (across all budgets):

| Plot Type | Description |
|-----------|-------------|
| `regret_curves` | Mean regret over evaluations |
| `convergence_probability` | Fraction of runs finding optimum |
| `comparison_heatmap` | Pairwise algorithm comparison matrix |

**Budget-specific plots** (one per budget):

| Plot Type | Description |
|-----------|-------------|
| `regret_boxplots` | Regret distribution at each budget |
| `performance_profile` | Dolan-Moré performance profiles |

**History/trajectory plots** (at `budget_for_plots`):

| Plot Type | Description |
|-----------|-------------|
| `history_current` | Current fitness value over time |
| `history_best` | Best-so-far fitness over time |
| `regret_instantaneous` | Instantaneous regret (current value) |
| `regret_instantaneous_best` | Instantaneous regret (incumbent) |
| `regret_cumulative` | Cumulative regret (current value) |
| `regret_cumulative_best` | Cumulative regret (incumbent) |
| `ttfo_distribution` | Time-to-first-optimum distribution |

**Runtime profile plots** (require `suite.profile: true`):

| Plot Type | Description |
|-----------|-------------|
| `inverse_runtime_profile_surface` | 3D surface P(τ_v ≤ t) per algorithm |
| `inverse_runtime_profile_curves` | 2D curves of runtime profiles |
| `cr_profile_verification` | Validate CR computation methods |

### Example: Minimal Plotting

To generate only regret curves and boxplots:

```yaml
plotting:
  enabled: true
  plots:
    regret_curves:
      enabled: true
      filename: "regret_curves.pdf"
      log_scale: true
    regret_boxplots:
      enabled: true
      filename: "regret_boxplot_b{budget}.pdf"
```

### Example: With Runtime Profiles

```yaml
suite:
  profile: true  # Required for profile plots
  # ...

plotting:
  enabled: true
  plots:
    inverse_runtime_profile_curves:
      enabled: true
      filename: "inverse_runtime_profile_curves.pdf"
    inverse_runtime_profile_surface:
      enabled: true
      filename: "inverse_runtime_profile_surface_{algorithm}.pdf"
    cr_profile_verification:
      enabled: true
      filename: "cr_profile_verification.pdf"
```

## Validation

The schema enforces several constraints:

1. **Budget consistency**: `budget_for_plots` must be in `suite.budgets`
2. **Registry membership**: `class` values must exist in their registries
3. **Profile dependencies**: Profile plots require `suite.profile: true`
4. **Output paths**: `figures_root` required when `plotting.enabled: true`

Always validate before running:

```bash
uv run run_experiment validate configs/experiments/your_config.yml
```

## File Naming Convention

Configs are numbered for organization:
- `01-07`: Core benchmark suites
- `08`: Runtime profile verification
- `09-14`: Specialized analyses
- `template_boilerplate.yml`: Starting point for new configs
