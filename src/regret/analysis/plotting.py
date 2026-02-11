import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# Set publication-quality defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 100


def plot_regret_curves(
    results: dict[tuple, list[dict]],
    save_path: str | None = None,
    log_scale: bool = True,
    title: str | None = None,
):
    """Plot mean regret vs budget for multiple algorithms."""

    # Organize data
    algorithms = set(alg for alg, _ in results.keys())
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        mean_regrets = []
        std_regrets = []

        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]])
                mean_regrets.append(np.mean(regrets))
                std_regrets.append(np.std(regrets))
            else:
                mean_regrets.append(np.nan)
                std_regrets.append(np.nan)

        mean_regrets = np.array(mean_regrets)
        std_regrets = np.array(std_regrets)

        if log_scale:
            ax.loglog(budgets, mean_regrets, marker="o", label=alg, linewidth=2)
        else:
            ax.plot(budgets, mean_regrets, marker="o", label=alg, linewidth=2)

        # Add shaded error region
        ax.fill_between(
            budgets, mean_regrets - std_regrets, mean_regrets + std_regrets, alpha=0.2
        )

    ax.set_xlabel("Budget (evaluations)")
    ax.set_ylabel("Mean Simple Regret")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_regret_boxplots(
    results: dict[tuple, list[dict]],
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
):
    """Create boxplots comparing algorithms at a specific budget."""

    algorithms = [alg for alg, b in results.keys() if b == budget]
    data = [
        np.array([r["regret"] for r in results[(alg, budget)]]) for alg in algorithms
    ]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, label=algorithms, patch_artist=True)

    # Customize colors
    colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Simple Regret")
    ax.set_xlabel("Algorithm")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Regret Distribution at Budget = {budget}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_convergence_probability(
    results: dict[tuple, list[dict]],
    save_path: str | None = None,
    title: str | None = None,
):
    """Plot probability of finding optimum vs budget."""

    algorithms = set(alg for alg, _ in results.keys())
    budgets = sorted(set(budget for _, budget in results.keys()))

    fig, ax = plt.subplots()

    for alg in algorithms:
        probs = []

        for budget in budgets:
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]])
                prob = np.mean(regrets < 1e-9)
                probs.append(prob)
            else:
                probs.append(np.nan)

        ax.semilogx(budgets, probs, marker="o", label=alg, linewidth=2)

    ax.set_xlabel("Budget (evaluations)")
    ax.set_ylabel("P(optimal)")
    ax.set_ylim((-0.05, 1.05))
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_comparison_heatmap(
    results: dict[tuple, list[dict]], save_path: str | None = None
):
    """Create heatmap of mean regrets across algorithms and budgets."""

    algorithms = sorted(set(alg for alg, _ in results.keys()))
    budgets = sorted(set(budget for _, budget in results.keys()))

    data = np.zeros((len(algorithms), len(budgets)))

    for i, alg in enumerate(algorithms):
        for j, budget in enumerate(budgets):
            if (alg, budget) in results:
                regrets = np.array([r["regret"] for r in results[(alg, budget)]])
                data[i, j] = np.log10(np.mean(regrets) + 1e-10)
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(budgets)))
    ax.set_yticks(range(len(algorithms)))
    ax.set_xticklabels([f"{b:.0e}" for b in budgets], rotation=45, ha="right")
    ax.set_yticklabels(algorithms)

    ax.set_xlabel("Budget")
    ax.set_ylabel("Algorithm")
    ax.set_title("Mean Regret (log10 scale)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log10(mean regret)")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_performance_profile(
    results: dict[tuple, list[dict]],
    budget: int,
    save_path: str | None = None,
    title: str | None = None,
):
    """Create performance profile (CDF of regrets)."""

    algorithms = [alg for alg, b in results.keys() if b == budget]

    fig, ax = plt.subplots()

    for alg in algorithms:
        regrets = np.array([r["regret"] for r in results[(alg, budget)]])
        sorted_regrets = np.sort(regrets)
        cdf = np.arange(1, len(sorted_regrets) + 1) / len(sorted_regrets)

        ax.plot(sorted_regrets, cdf, label=alg, linewidth=2)

    ax.set_xlabel("Simple Regret")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xscale("log")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Performance Profile (Budget = {budget})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
