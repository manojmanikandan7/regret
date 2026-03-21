"""
JSON Schema for experiment configuration files.

Defines the structure and constraints for YAML experiment configs.
"""

CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["suite", "problems", "algorithms", "plotting"],
    "properties": {
        "suite": {
            "type": "object",
            "required": ["name", "runs", "budgets", "mode", "output"],
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Name of the experiment suite",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description of the suite",
                },
                "runs": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of independent runs per configuration",
                },
                "mode": {
                    "type": "string",
                    "enum": ["lite", "full"],
                    "description": "Output mode: 'lite' (stats only) or 'full' (with trajectories)",
                },
                "parallel": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to run experiments in parallel",
                },
                "budgets": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "description": "List of evaluation budgets to test",
                },
                "output": {
                    "type": "object",
                    "required": ["raw_root", "figures_root"],
                    "properties": {
                        "raw_root": {"type": "string"},
                        "figures_root": {"type": "string"},
                    },
                    "additionalProperties": False,
                    "description": "Output directory configuration",
                },
            },
            "additionalProperties": False,
        },
        "problems": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "class", "params"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Display name for this configured problem",
                    },
                    "budget_for_plots": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional per-problem override for budget-specific figures",
                    },
                    "class": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Problem class name from PROBLEM_REGISTRY",
                    },
                    "params": {
                        "type": "object",
                        "description": "Problem-specific parameters (e.g., n, k)",
                    },
                },
                "additionalProperties": False,
            },
            "description": "List of problem configurations",
        },
        "algorithms": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "class", "args"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Display name for this configured algorithm",
                    },
                    "class": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Algorithm class name from ALGORITHM_REGISTRY",
                    },
                    "args": {
                        "type": "object",
                        "required": ["defaults"],
                        "properties": {
                            "defaults": {
                                "type": "object",
                                "description": "Default algorithm parameters",
                            },
                            "by_problem": {
                                "type": "object",
                                "description": "Problem-specific parameter overrides",
                            },
                        },
                        "additionalProperties": False,
                        "description": "Algorithm arguments configuration",
                    },
                },
                "additionalProperties": False,
            },
            "description": "List of algorithm configurations",
        },
        "plotting": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "budget_for_plots": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Budget used for budget-specific figures (defaults to max suite budget)",
                },
                "layout": {
                    "type": "object",
                    "properties": {
                        "per_problem_dir": {"type": "boolean"},
                        "include_n_subdir": {"type": "boolean"},
                        "structure": {
                            "type": "object",
                            "properties": {
                                "aggregate": {"type": "string"},
                                "history": {"type": "string"},
                                "distribution": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
                "titles": {
                    "type": "object",
                    "properties": {
                        "include_problem_name": {"type": "boolean"},
                        "include_problem_size": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                "plots": {
                    "type": "object",
                    "properties": {
                        "regret_curves": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "log_scale": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "description": "Uncertainty display mode for aggregate regret curves",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                                "annotate_pairwise": {"type": "boolean"},
                                "comparison_budget": {"type": "integer", "minimum": 1},
                                "paired_runs": {"type": "boolean"},
                            },
                        },
                        "convergence_probability": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "show_confidence_band": {"type": "boolean"},
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                            },
                        },
                        "comparison_heatmap": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                            },
                        },
                        "regret_boxplots": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "show_points": {"type": "boolean"},
                                "annotate_pairwise": {"type": "boolean"},
                                "reference_algorithm": {"type": "string"},
                                "paired_runs": {"type": "boolean"},
                            },
                        },
                        "performance_profile": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "annotate_pairwise": {"type": "boolean"},
                                "reference_algorithm": {"type": "string"},
                                "paired_runs": {"type": "boolean"},
                            },
                        },
                        "history_current": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["current", "best"],
                                },
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "history_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["current", "best"],
                                },
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "regret_instantaneous": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                },
                                "use_best": {"type": "boolean"},
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "regret_cumulative": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                },
                                "use_best": {"type": "boolean"},
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "regret_instantaneous_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                },
                                "use_best": {"type": "boolean"},
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "regret_cumulative_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                },
                                "use_best": {"type": "boolean"},
                                "log_x": {"type": "boolean"},
                                "log_y": {"type": "boolean"},
                                "show_ttfo_markers": {"type": "boolean"},
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                },
                                "n_bootstrap": {"type": "integer", "minimum": 1},
                            },
                        },
                        "ttfo_distribution": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "filename": {"type": "string"},
                                "tolerance": {"type": "number", "minimum": 0},
                                "show_median": {"type": "boolean"},
                                "annotate_pairwise": {"type": "boolean"},
                                "reference_algorithm": {"type": "string"},
                                "paired_runs": {"type": "boolean"},
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
            "description": "Plotting configuration",
        },
    },
    "additionalProperties": False,
}
