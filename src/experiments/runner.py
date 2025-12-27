import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ExperimentRunner:
    def __init__(self, output_dir='results/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(self, algorithm_class, problem, budget, seed):
        alg = algorithm_class(problem, seed=seed)
        best_value, _ = alg.run(budget)
        return {
            'regret': problem.f_star - best_value,
            'best_value': best_value,
            'evaluations': alg.evaluations,
            'seed': seed
        }

    def run_experiment(self, algorithm_class, problem, budget, runs=30, name=None):
        results = []
        for run in range(runs):
            result = self.run_single(algorithm_class, problem, budget, run)
            results.append(result)

        output = {
            'metadata': {
                'algorithm': algorithm_class.__name__,
                'problem': problem.__class__.__name__,
                'problem_size': problem.n,
                'budget': budget,
                'runs': runs,
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }

        if name:
            filepath = self.output_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2)

        return results
