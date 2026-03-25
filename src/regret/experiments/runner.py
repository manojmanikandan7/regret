import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from regret.core.base import Algorithm, Problem
from regret.core.metrics import compute_statistics, probability_optimal, simple_regret


class ExperimentRunner:
    """Run and manage optimization experiments."""

    def __init__(self, output_dir: str = "results/raw", max_workers: int | None = None):
        """Initialize an experiment runner.

        Args:
            output_dir: Directory where raw JSON result files are written.
            max_workers: Maximum worker processes for parallel execution.
                If None, ProcessPoolExecutor chooses a default.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def run_single(
        self,
        algorithm_class: type[Algorithm],
        problem: Problem,
        budget: int,
        seed: int,
        mode: str = "lite",
        trajectory_stride: int = 1,
        **alg_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one seeded optimization trial.

        Args:
            algorithm_class: Algorithm class to instantiate and run.
            problem: Problem instance optimized by the algorithm.
            budget: Evaluation budget for the run.
            seed: Random seed for reproducibility.
            mode: Output mode. Use "lite" for summary metrics only, or "full"
                to include sampled trajectory history.
            trajectory_stride: Sampling stride for trajectory points in full mode.
                Must be at least 1.
            **alg_kwargs: Additional keyword arguments forwarded to algorithm
                construction.

        Returns:
            A result dictionary containing metrics and metadata for one run.

        Raises:
            ValueError: If trajectory_stride is less than 1.
        """
        if trajectory_stride < 1:
            raise ValueError("trajectory_stride must be >= 1")

        alg = algorithm_class(problem, seed=seed, **alg_kwargs)
        best_value, _best_solution = alg.run(budget)

        # TODO: Include best solution achieved as a bitstring
        result = {
            "regret": simple_regret(best_value, problem.f_star),
            "best_value": best_value,
            "optimal": abs(best_value - problem.f_star) < 1e-9,
            "evaluations": alg.evaluations,
            "seed": seed,
        }
        if mode == "full":
            trajectory = list(alg.history)
            if trajectory_stride > 1:
                downsampled = trajectory[::trajectory_stride]
                # Preserve the terminal state even when the stride skips it.
                if downsampled and downsampled[-1][0] != trajectory[-1][0]:
                    downsampled.append(trajectory[-1])
                trajectory = downsampled
            result["trajectory"] = trajectory
        return result

    def run_experiment(
        self,
        algorithm_class: type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int = 30,
        mode: str = "lite",
        name: str | None = None,
        parallel: bool = False,
        trajectory_stride: int = 1,
        **alg_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run multiple independent trials for one algorithm/problem pair.

        Args:
            algorithm_class: Algorithm class to instantiate per run.
            problem: Problem instance optimized by each run.
            budget: Evaluation budget per run.
            runs: Number of seeded independent runs.
            mode: Output mode passed to each single run.
            name: Optional experiment name used when persisting result JSON.
            parallel: Whether to execute runs via a process pool.
            trajectory_stride: Trajectory sampling stride for full mode.
            **alg_kwargs: Additional keyword arguments forwarded to algorithm
                construction.

        Returns:
            Ordered list of per-run result dictionaries.

        Raises:
            ValueError: If trajectory_stride is less than 1.
        """

        if trajectory_stride < 1:
            raise ValueError("trajectory_stride must be >= 1")

        if parallel:
            results = self._run_parallel(
                algorithm_class,
                problem,
                budget,
                runs,
                mode,
                trajectory_stride=trajectory_stride,
                **alg_kwargs,
            )
        else:
            results = []
            for seed in tqdm(range(runs), desc=f"{algorithm_class.__name__}", leave=False):
                result = self.run_single(
                    algorithm_class,
                    problem,
                    budget,
                    seed,
                    mode,
                    trajectory_stride=trajectory_stride,
                    **alg_kwargs,
                )
                results.append(result)

        # Save results
        if name:
            self._save_results(name, algorithm_class, problem, budget, runs, mode, results)

        return results

    def _run_parallel(
        self,
        algorithm_class: type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int,
        mode: str,
        trajectory_stride: int,
        **alg_kwargs,
    ) -> list[dict[str, Any]]:
        """Run seeded trials in parallel using multiple worker processes.

        Args:
            algorithm_class: Algorithm class to instantiate per worker task.
            problem: Problem instance optimized by each run.
            budget: Evaluation budget per run.
            runs: Number of seeded independent runs.
            mode: Output mode passed to each single run.
            trajectory_stride: Trajectory sampling stride for full mode.
            **alg_kwargs: Additional keyword arguments forwarded to algorithm
                construction.

        Returns:
            Seed-ordered list of per-run result dictionaries.

        Raises:
            RuntimeError: If any worker task fails.
        """
        results_by_seed: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_single,
                    algorithm_class,
                    problem,
                    budget,
                    seed,
                    mode,
                    trajectory_stride,
                    **alg_kwargs,
                ): seed
                for seed in range(runs)
            }
            for future in tqdm(as_completed(futures), total=runs, desc="Parallel runs"):
                seed = futures[future]
                try:
                    results_by_seed[seed] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Parallel run failed for seed={seed}") from e

        # Keep result order deterministic and aligned with seed index.
        return [results_by_seed[seed] for seed in range(runs)]

    def _save_results(
        self,
        name: str,
        algorithm_class: type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int,
        mode: str,
        results: list[dict[str, Any]],
    ):
        """Persist experiment outputs and aggregate statistics to JSON.

        Args:
            name: Base filename (without extension) for the output JSON.
            algorithm_class: Algorithm class used for metadata.
            problem: Problem instance used for metadata and global optimum.
            budget: Evaluation budget used for the experiment.
            runs: Number of independent runs contained in results.
            mode: Output mode used during execution.
            results: Per-run result dictionaries to serialize.
        """
        regrets = np.array([r["regret"] for r in results])

        output = {
            "metadata": {
                "name": name,
                "algorithm": algorithm_class.__name__,
                "problem": problem.__class__.__name__,
                "problem_size": problem.n,
                "budget": budget,
                "runs": runs,
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                **compute_statistics(regrets),
                "prob_optimal": probability_optimal(regrets),
                "global_optimum": problem.f_star,
            },
            "results": results,
        }

        filepath = self.output_dir / f"{name}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, separators=(",", ":"))  # Minified for storage efficiency

    def load_results(self, name: str) -> dict[str, Any]:
        """Load a previously saved experiment JSON result file.

        Args:
            name: Base filename (without extension) of the stored results.

        Returns:
            Parsed experiment result payload.
        """
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


class BatchRunner:
    """Run batches of experiments with multiple algorithms and budgets."""

    def __init__(self, runner: ExperimentRunner | None = None):
        """Initialize a batch runner.

        Args:
            runner: Optional ExperimentRunner instance. If omitted, a default
                ExperimentRunner is created.
        """
        self.runner = runner or ExperimentRunner()

    def run_suite(
        self,
        algorithms: dict[str, type[Algorithm]],
        problem: Problem,
        budgets: list[int],
        runs: int = 30,
        suite_name: str = "suite",
        parallel: bool = False,
        algorithms_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run all algorithm/budget combinations for a single problem.

        Args:
            algorithms: Mapping of display name to algorithm class.
            problem: Problem instance shared by all runs.
            budgets: Evaluation budgets to evaluate.
            runs: Number of seeded runs per (algorithm, budget) pair.
            suite_name: Prefix used when naming persisted experiments.
            parallel: Whether each experiment is executed in parallel.
            algorithms_kwargs: Optional mapping of algorithm names to their
                keyword arguments. If provided, must contain entries for each
                algorithm in the algorithms dict.

        Returns:
            Mapping keyed by (algorithm_name, budget) to list of run results.
        """

        all_results = {}
        algorithms_kwargs = algorithms_kwargs or {}

        for budget in budgets:
            print(f"\n{'=' * 60}")
            print(f"Budget: {budget}")
            print(f"{'=' * 60}")

            for alg_name, alg_class in algorithms.items():
                exp_name = f"{suite_name}_{alg_name}_b{budget}"
                alg_kwargs = algorithms_kwargs.get(alg_name, {})
                results = self.runner.run_experiment(
                    alg_class,
                    problem,
                    budget,
                    runs,
                    name=exp_name,
                    parallel=parallel,
                    **alg_kwargs,
                )

                regrets = np.array([r["regret"] for r in results])
                stats = compute_statistics(regrets)

                print(
                    f"{alg_name:20s} | Mean: {stats['mean']:.4e} | "
                    f"Median: {stats['median']:.4e} | "
                    f"P(opt): {probability_optimal(regrets):.3f}"
                )

                all_results[(alg_name, budget)] = results

        return all_results
