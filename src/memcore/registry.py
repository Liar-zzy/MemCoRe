"""Explicit implementation versions; importing this module never loads models."""

from importlib import import_module


# v1 is the copied KeyGraphV3 baseline. Register a future extension here only
# after it has its own implementation and validation.
VERSIONS = {
    "v1": {
        "evaluator": "memcore.versions.v1.evaluation",
        "prompts": "memcore.versions.v1.prompts.prompt_loader",
    },
}


def get_version(name):
    if name not in VERSIONS:
        raise ValueError(f"Unknown implementation version {name!r}; available: {', '.join(VERSIONS)}")
    return VERSIONS[name]


def load_evaluator(name):
    return import_module(get_version(name)["evaluator"]).evaluate_dataset
