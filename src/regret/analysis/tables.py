"""Table generation utilities for experiment results.

Provides functions to create summary tables from experiment results and export
them to various formats (LaTeX, CSV, Markdown).

All regret statistics in tables refer to **simple regret**: the difference
between the global optimum f* and the best fitness value found at the end
of the optimization run (f* - f_best). Lower is better; zero means optimum found.

Type Aliases (imported from regret._types):
    KeyedResults: Results dict keyed by (algorithm_name, budget) tuples.
    TableStatistics: Dict containing summary statistics for simple regret values.
"""

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regret._types import KeyedResults, TableStatistics
from regret.analysis.statistics import bootstrap_confidence_interval


class TableFormat(str, Enum):
    """Supported table output formats."""

    LATEX = "latex"
    CSV = "csv"
    MARKDOWN = "markdown"


def compute_table_statistics(regrets: np.ndarray, confidence: float = 0.95) -> TableStatistics:
    """Compute summary statistics for a set of simple regret values.

    Simple regret is defined as f* - f_best, where f* is the global optimum
    and f_best is the best fitness found during the run.

    Args:
        regrets: 1D array of simple regret values from multiple runs.
        confidence: Confidence level for bootstrap CI (default: 0.95).

    Returns:
        TableStatistics dict containing:
            - mean: Mean simple regret
            - median: Median simple regret
            - std: Standard deviation (sample, ddof=1)
            - iqr: Interquartile range (Q75 - Q25)
            - ci_lower: Lower bound of bootstrap CI for mean
            - ci_upper: Upper bound of bootstrap CI for mean
            - p_opt: Probability of finding optimum (regret < 1e-9)
            - n_runs: Number of runs
    """
    mean = float(np.mean(regrets))
    median = float(np.median(regrets))
    std = float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0
    q25, q75 = np.percentile(regrets, [25, 75])
    iqr = float(q75 - q25)
    ci_lower, ci_upper = bootstrap_confidence_interval(regrets, confidence=confidence)
    p_opt = float(np.mean(regrets < 1e-9))

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "iqr": iqr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_opt": p_opt,
        "n_runs": len(regrets),
    }


def create_results_table(
    results: KeyedResults,
    budget: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Create a summary table for algorithm comparison at a specific budget.

    All statistics are computed over simple regret values (f* - f_best).

    Args:
        results: Results keyed by (algorithm_name, budget) tuples. Each value
            is a list of run result dicts with 'regret' key (float).
        budget: The budget value to filter results by.
        confidence: Confidence level for bootstrap CI (default: 0.95).

    Returns:
        DataFrame with simple regret statistics for each algorithm,
        sorted by mean (ascending). Lower values indicate better performance.
        Columns: Algorithm, Mean SR, Median SR, Std, IQR, CI (95%), P(opt), Runs.
    """
    algorithms = [alg for alg, b in results.keys() if b == budget]

    rows = []
    for alg in algorithms:
        regrets = np.array([r["regret"] for r in results[(alg, budget)]])
        stats = compute_table_statistics(regrets, confidence=confidence)

        rows.append(
            {
                "Algorithm": alg,
                "Mean SR": f"{stats['mean']:.4e}",
                "Median SR": f"{stats['median']:.4e}",
                "Std": f"{stats['std']:.4e}",
                "IQR": f"{stats['iqr']:.4e}",
                "CI (95%)": f"[{stats['ci_lower']:.4e}, {stats['ci_upper']:.4e}]",
                "P(opt)": f"{stats['p_opt']:.3f}",
                "Runs": stats["n_runs"],
                "_mean_sort": stats["mean"],  # Hidden column for sorting
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_mean_sort").drop(columns=["_mean_sort"]).reset_index(drop=True)
    return df


def create_aggregate_table(
    results_by_problem: dict[str, KeyedResults],
    budget: int,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Create an aggregate table comparing algorithms across all problems.

    For each algorithm, computes mean simple regret averaged across all problems,
    along with the number of problems where it achieved the best mean simple regret.

    Args:
        results_by_problem: Nested dictionary keyed by problem_name, then by
            (algorithm_name, budget) tuples. Each innermost value is a list of
            run result dicts with 'regret' key (float).
        budget: The budget value to filter results by.
        confidence: Confidence level for bootstrap CI (default: 0.95).

    Returns:
        DataFrame with aggregate simple regret statistics across problems,
        sorted by average mean simple regret (ascending).
        Columns: Algorithm, Avg Mean SR, Avg P(opt), Best Count, Problems.
    """
    # Collect per-problem stats for each algorithm
    alg_stats: dict[str, list[dict[str, Any]]] = {}
    problem_names = sorted(results_by_problem.keys())

    for problem_name in problem_names:
        problem_results = results_by_problem[problem_name]
        algorithms = [alg for alg, b in problem_results.keys() if b == budget]

        for alg in algorithms:
            if alg not in alg_stats:
                alg_stats[alg] = []

            regrets = np.array([r["regret"] for r in problem_results[(alg, budget)]])
            stats = compute_table_statistics(regrets, confidence=confidence)
            # Extend stats with problem name for tracking
            extended_stats: dict[str, Any] = {**stats, "problem": problem_name}
            alg_stats[alg].append(extended_stats)

    # Find best algorithm per problem (lowest mean regret)
    best_per_problem: dict[str, str] = {}
    for problem_name in problem_names:
        problem_results = results_by_problem[problem_name]
        best_mean = float("inf")
        best_alg = ""
        for alg, b in problem_results.keys():
            if b != budget:
                continue
            regrets = np.array([r["regret"] for r in problem_results[(alg, b)]])
            mean = float(np.mean(regrets))
            if mean < best_mean:
                best_mean = mean
                best_alg = alg
        best_per_problem[problem_name] = best_alg

    # Build aggregate rows
    rows = []
    for alg, stats_list in alg_stats.items():
        if not stats_list:
            continue

        avg_mean = np.mean([s["mean"] for s in stats_list])
        avg_p_opt = np.mean([s["p_opt"] for s in stats_list])
        best_count = sum(1 for p, best in best_per_problem.items() if best == alg)
        n_problems = len(stats_list)

        rows.append(
            {
                "Algorithm": alg,
                "Avg Mean SR": f"{avg_mean:.4e}",
                "Avg P(opt)": f"{avg_p_opt:.3f}",
                "Best Count": best_count,
                "Problems": n_problems,
                "_avg_mean_sort": avg_mean,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_avg_mean_sort").drop(columns=["_avg_mean_sort"]).reset_index(drop=True)
    return df


def export_table(df: pd.DataFrame, save_path: Path, fmt: TableFormat) -> None:
    """Export a DataFrame to the specified format.

    Args:
        df: DataFrame to export.
        save_path: Output file path (extension will be adjusted if needed).
        fmt: Output format (latex, csv, or markdown).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == TableFormat.LATEX:
        _export_latex(df, save_path)
    elif fmt == TableFormat.CSV:
        _export_csv(df, save_path)
    elif fmt == TableFormat.MARKDOWN:
        _export_markdown(df, save_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _export_latex(df: pd.DataFrame, save_path: Path) -> None:
    """Export DataFrame to LaTeX table format."""
    latex = df.to_latex(index=False, escape=False)
    with open(save_path, "w") as f:
        f.write(latex)


def _export_csv(df: pd.DataFrame, save_path: Path) -> None:
    """Export DataFrame to CSV format."""
    df.to_csv(save_path, index=False)


def _export_markdown(df: pd.DataFrame, save_path: Path) -> None:
    """Export DataFrame to Markdown table format."""
    markdown = df.to_markdown(index=False)
    with open(save_path, "w") as f:
        f.write(markdown if markdown else "")


# Legacy API for backwards compatibility with existing tests
def export_latex_table(df: pd.DataFrame, save_path: str) -> None:
    """Export DataFrame to LaTeX table.

    .. deprecated::
        Use `export_table(df, Path(save_path), TableFormat.LATEX)` instead.
    """
    _export_latex(df, Path(save_path))
    print(f"LaTeX table saved to {save_path}")
