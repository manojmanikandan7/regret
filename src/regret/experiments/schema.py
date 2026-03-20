"""
JSON Schema for experiment configuration files.

Defines the structure and constraints for YAML experiment configs.
"""

CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["suite", "problems", "algorithms"],
    "properties": {
        "suite": {
            "type": "object",
            "required": ["name", "runs", "budgets", "mode"],
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
                    "properties": {
                        "raw_root": {"type": "string"},
                        "figures_root": {"type": "string"},
                    },
                    "description": "Output directory configuration",
                },
            },
        },
        "problems": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["class", "params"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Display name (defaults to class name)",
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
            },
            "description": "List of problem configurations",
        },
        "algorithms": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["class", "args"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Display name (defaults to class name)",
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
                "layout": {"type": "object"},
                "titles": {"type": "object"},
                "plots": {"type": "object"},
            },
            "description": "Plotting configuration",
        },
    },
    "additionalProperties": False,
}
