import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Type, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from regret.core.base import Algorithm, Problem
from regret.core.metrics import simple_regret, compute_statistics, probability_optimal


class ExperimentRunner:
    """Run and manage optimization experiments."""

    def __init__(self, output_dir: str = "results/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        algorithm_class: Type[Algorithm],
        problem: Problem,
        budget: int,
        seed: int,
        **alg_kwargs,
    ) -> Dict[str, Any]:
        """Run a single experiment."""
        alg = algorithm_class(problem, seed=seed, **alg_kwargs)
        best_value, best_solution = alg.run(budget)

        return {
            "regret": simple_regret(best_value, problem.f_star),
            "best_value": best_value,
            "optimal": abs(best_value - problem.f_star) < 1e-9,
            "evaluations": alg.evaluations,
            "seed": seed,
        }

    def run_experiment(
        self,
        algorithm_class: Type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int = 30,
        name: str | None = None,
        parallel: bool = False,
        **alg_kwargs,
    ) -> List[Dict[str, Any]]:
        """Run multiple independent trials."""

        if parallel:
            results = self._run_parallel(
                algorithm_class, problem, budget, runs, **alg_kwargs
            )
        else:
            results = []
            for seed in tqdm(
                range(runs), desc=f"{algorithm_class.__name__}", leave=False
            ):
                result = self.run_single(
                    algorithm_class, problem, budget, seed, **alg_kwargs
                )
                results.append(result)

        # Save results
        if name:
            self._save_results(name, algorithm_class, problem, budget, runs, results)

        return results

    def _run_parallel(
        self,
        algorithm_class: Type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int,
        **alg_kwargs,
    ) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.run_single,
                    algorithm_class,
                    problem,
                    budget,
                    seed,
                    **alg_kwargs,
                ): seed
                for seed in range(runs)
            }
            for future in tqdm(as_completed(futures), total=runs, desc="Parallel runs"):
                results.append(future.result())
        return results

    def _save_results(
        self,
        name: str,
        algorithm_class: Type[Algorithm],
        problem: Problem,
        budget: int,
        runs: int,
        results: List[Dict[str, Any]],
    ):
        """Save experiment results to JSON."""
        regrets = np.array([r["regret"] for r in results])

        output = {
            "metadata": {
                "name": name,
                "algorithm": algorithm_class.__name__,
                "problem": problem.__class__.__name__,
                "problem_size": problem.n,
                "budget": budget,
                "runs": runs,
                "timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                **compute_statistics(regrets),
                "prob_optimal": probability_optimal(regrets),
            },
            "results": results,
        }

        filepath = self.output_dir / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

    def load_results(self, name: str) -> Dict[str, Any]:
        """Load saved results."""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, "r") as f:
            return json.load(f)


class BatchRunner:
    """Run batches of experiments with multiple algorithms and budgets."""

    def __init__(self, runner: ExperimentRunner | None = None):
        self.runner = runner or ExperimentRunner()

    def run_suite(
        self,
        algorithms: Dict[str, Type[Algorithm]],
        problem: Problem,
        budgets: List[int],
        runs: int = 30,
        suite_name: str = "suite",
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Run a full suite of experiments."""

        all_results = {}

        for budget in budgets:
            print(f"\n{'=' * 60}")
            print(f"Budget: {budget}")
            print(f"{'=' * 60}")

            for alg_name, alg_class in algorithms.items():
                exp_name = f"{suite_name}_{alg_name}_b{budget}"
                results = self.runner.run_experiment(
                    alg_class, problem, budget, runs, name=exp_name, parallel=parallel
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
