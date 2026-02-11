from regret.algorithms.local_search import RLS
from regret.algorithms.evolutionary import OnePlusOneEA
from regret.problems.pseudo_boolean import OneMax
from regret.experiments.runner import ExperimentRunner
from regret.analysis.plotting import plot_regret_curves

if __name__ == "__main__":
    runner = ExperimentRunner()
    problem = OneMax(n=100)
    budgets = [10**i for i in range(2, 6)]

    algorithms = {"RLS": RLS, "(1+1)-EA": OnePlusOneEA}

    all_results = {}
    for name, AlgClass in algorithms.items():
        for budget in budgets:
            exp_name = f"onemax100_{name}_b{budget}"
            results = runner.run_experiment(
                AlgClass, problem, budget, runs=30, name=exp_name
            )
            all_results[(name, budget)] = results

    plot_regret_curves(all_results, save_path="results/figures/onemax100.pdf")
