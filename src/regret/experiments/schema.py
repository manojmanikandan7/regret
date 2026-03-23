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
                    "description": "Whether to run experiments in parallel (default: true)",
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
                    "required": ["raw_root"],
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
                        "description": "Optional per-problem override for budget-specific figures. Default resolution order: problems[i].budget_for_plots -> plotting.budget_for_plots -> max(suite.budgets)",
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
                                "default": {},
                                "description": "Problem-specific parameter overrides (default: empty mapping)",
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
                "enabled": {
                    "type": "boolean",
                    "description": "Master plotting switch. Must be explicitly provided by config",
                },
                "budget_for_plots": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Budget used for budget-specific figures (defaults to max suite budget)",
                },
                "layout": {
                    "type": "object",
                    "properties": {
                        "per_problem_dir": {
                            "type": "boolean",
                            "default": True,
                            "description": "Create a problem-level subdirectory (default: true)",
                        },
                        "include_n_subdir": {
                            "type": "boolean",
                            "default": True,
                            "description": "Create an n-size subdirectory (default: true)",
                        },
                        "structure": {
                            "type": "object",
                            "properties": {
                                "aggregate": {
                                    "type": "string",
                                    "default": "aggregate",
                                    "description": "Subdirectory for cross-budget plots (default: aggregate)",
                                },
                                "history": {
                                    "type": "string",
                                    "default": "history",
                                    "description": "Subdirectory for trajectory/history plots (default: history)",
                                },
                                "distribution": {
                                    "type": "string",
                                    "description": "Subdirectory for budget-specific distribution/profile-at-budget plots (default: budget_{budget_for_plots})",
                                },
                                "profile": {
                                    "type": "string",
                                    "default": "profiles",
                                    "description": "Subdirectory for runtime profile plots (default: profiles)",
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
                "titles": {
                    "type": "object",
                    "properties": {
                        "include_problem_name": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include problem display name in figure titles (default: true)",
                        },
                        "include_problem_size": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include problem size n in figure titles (default: true)",
                        },
                    },
                    "additionalProperties": False,
                },
                "plots": {
                    "type": "object",
                    "properties": {
                        "regret_curves": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable aggregate regret curve plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "regret_curves.pdf",
                                },
                                "log_scale": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Use log scaling on budget axis (default: true)",
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                    "description": "Uncertainty display mode for aggregate regret curves (default: sem)",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 2000,
                                },
                                "annotate_pairwise": {
                                    "type": "boolean",
                                    "default": False,
                                },
                                "comparison_budget": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "Budget used for pairwise annotation. If omitted, uses the largest available budget",
                                },
                                "paired_runs": {
                                    "type": "boolean",
                                    "default": False,
                                },
                            },
                        },
                        "convergence_probability": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable convergence probability plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "convergence_probability.pdf",
                                },
                                "show_confidence_band": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                            },
                        },
                        "comparison_heatmap": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable pairwise comparison heatmap (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "comparison_heatmap.pdf",
                                },
                            },
                        },
                        "regret_boxplots": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable budget-specific regret boxplots (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "regret_boxplot_b{budget}.pdf",
                                },
                                "show_points": {"type": "boolean", "default": True},
                                "annotate_pairwise": {
                                    "type": "boolean",
                                    "default": False,
                                },
                                "reference_algorithm": {
                                    "type": "string",
                                    "description": "Reference algorithm for pairwise tests. If omitted and pairwise annotation is enabled, the best mean-regret algorithm at the selected budget is used",
                                },
                                "paired_runs": {
                                    "type": "boolean",
                                    "default": False,
                                },
                            },
                        },
                        "performance_profile": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable budget-specific performance profiles (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "performance_profile_b{budget}.pdf",
                                },
                                "annotate_pairwise": {
                                    "type": "boolean",
                                    "default": False,
                                },
                                "reference_algorithm": {
                                    "type": "string",
                                    "description": "Reference algorithm for pairwise tests. If omitted and pairwise annotation is enabled, the best mean-regret algorithm at the selected budget is used",
                                },
                                "paired_runs": {
                                    "type": "boolean",
                                    "default": False,
                                },
                            },
                        },
                        "history_current": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable current-value history trajectory (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_current.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["current", "best"],
                                    "default": "current",
                                },
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": False},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "history_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable best-value history trajectory (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_best.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["current", "best"],
                                    "default": "best",
                                },
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": False},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "regret_instantaneous": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable instantaneous regret trajectory (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_regret_instantaneous.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                    "default": "instantaneous",
                                },
                                "use_best": {"type": "boolean", "default": False},
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": True},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "regret_cumulative": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable cumulative regret trajectory (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_regret_cumulative.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                    "default": "cumulative",
                                },
                                "use_best": {"type": "boolean", "default": False},
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": True},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "regret_instantaneous_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable instantaneous regret over best-so-far values (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_regret_instantaneous_best.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                    "default": "instantaneous",
                                },
                                "use_best": {"type": "boolean", "default": True},
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": True},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "regret_cumulative_best": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable cumulative regret over best-so-far values (default: true, requires known f_star)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_regret_cumulative_best.pdf",
                                },
                                "series": {
                                    "type": "string",
                                    "enum": ["instantaneous", "cumulative"],
                                    "default": "cumulative",
                                },
                                "use_best": {"type": "boolean", "default": True},
                                "log_x": {"type": "boolean", "default": False},
                                "log_y": {"type": "boolean", "default": True},
                                "show_ttfo_markers": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "spread": {
                                    "type": "string",
                                    "enum": [
                                        "none",
                                        "sem",
                                        "sd",
                                        "bootstrap_ci",
                                        "iqr",
                                    ],
                                    "default": "sem",
                                },
                                "confidence": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    "exclusiveMaximum": 1,
                                    "default": 0.95,
                                },
                                "n_bootstrap": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 1000,
                                },
                            },
                        },
                        "ttfo_distribution": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable TTFO distribution plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "history_ttfo_markers.pdf",
                                },
                                "tolerance": {
                                    "type": "number",
                                    "minimum": 0,
                                    "default": 1e-9,
                                },
                                "show_median": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "annotate_pairwise": {
                                    "type": "boolean",
                                    "default": False,
                                },
                                "reference_algorithm": {
                                    "type": "string",
                                    "description": "Reference algorithm for pairwise tests. If omitted and pairwise annotation is enabled, the algorithm with the lowest mean TTFO is used",
                                },
                                "paired_runs": {
                                    "type": "boolean",
                                    "default": False,
                                },
                            },
                        },
                        "runtime_profile_surface": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable per-algorithm runtime profile surface plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "runtime_profile_surface_{algorithm}.pdf",
                                    "description": "Include `{algorithm}` as a placeholder in the filename",
                                },
                            },
                        },
                        "runtime_profile_curves": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable runtime profile curve plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "runtime_profile_curves.pdf",
                                },
                            },
                        },
                        "cr_profile_verification": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable profile identity verification plot (default: true)",
                                },
                                "filename": {
                                    "type": "string",
                                    "default": "cr_profile_verification.pdf",
                                },
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
    "allOf": [
        {
            "if": {
                "properties": {
                    "plotting": {
                        "type": "object",
                        "properties": {
                            "enabled": {"const": True},
                        },
                        "required": ["enabled"],
                    }
                }
            },
            "then": {
                "properties": {
                    "suite": {
                        "type": "object",
                        "properties": {
                            "output": {
                                "type": "object",
                                "required": ["figures_root"],
                            }
                        },
                    }
                }
            },
        }
    ],
    "additionalProperties": False,
}
