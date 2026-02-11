import numpy as np
import pandas as pd
from typing import Dict, List


def create_results_table(results: Dict[tuple, List[Dict]], budget: int) -> pd.DataFrame:
    """Create a summary table for a specific budget."""

    algorithms = [alg for alg, b in results.keys() if b == budget]

    rows = []
    for alg in algorithms:
        regrets = np.array([r["regret"] for r in results[(alg, budget)]])
        rows.append(
            {
                "Algorithm": alg,
                "Mean": f"{np.mean(regrets):.4e}",
                "Median": f"{np.median(regrets):.4e}",
                "Std": f"{np.std(regrets):.4e}",
                "Min": f"{np.min(regrets):.4e}",
                "Max": f"{np.max(regrets):.4e}",
                "P(opt)": f"{np.mean(regrets < 1e-9):.3f}",
            }
        )

    return pd.DataFrame(rows)


def export_latex_table(df: pd.DataFrame, save_path: str):
    """Export DataFrame to LaTeX table."""
    latex = df.to_latex(index=False, escape=False)

    with open(save_path, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to {save_path}")
