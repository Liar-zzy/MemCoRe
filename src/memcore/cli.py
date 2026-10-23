"""Command line entry point; service dependencies load only for execution."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from .config import BOOL_FIELDS, CHOICES, DEFAULT_CONFIG_PATH, FLOAT_FIELDS, INT_FIELDS, ConfigError, ExperimentConfig, load_config


FIELD_HELP = {
    "version": "Implementation version registered in MemCore",
    "dataset": "LoCoMo dataset JSON path",
    "model": "Override the model in the LLM configuration",
    "output": "Evaluation results directory",
    "ratio": "Fraction of conversations to evaluate, in (0, 1]",
    "backend": "Override the LLM backend",
    "temperature_c5": "Generation temperature used only for category C5",
    "retrieve_k": "Legacy retrieval initial anchor count (default: 10); not the final evidence-pool size; paper retrieval uses round_top_k",
    "round_top_k": "Retrieved notes per round (default: 60); 'none' inherits final_top_k",
    "final_top_k": "Final evidence-pool limit in notes (default: 60)",
    "context_char_limit": "Character limit for evidence context (default: 90000)",
    "mode": "Memory system mode for this baseline",
    "graph_backend": "Graph store backend (inmemory)",
    "llm_config_path": "Memory-system LLM and retrieval embedding configuration YAML path",
    "api_key": "Override the LLM API key (redacted in dry-run output)",
    "base_url": "Override the LLM API base URL",
    "name": "Experiment name; a single nonempty path component",
    "sort_by": "Sort order of retrieval results",
    "prompt_config_path": "Prompt version selection YAML path",
    "use_format_instruction": "Enable format instructions (enabled in the default profile)",
    "retrieval_version": "Memory retrieval variant; paper is the default, v1 is the historical ablation",
    "rrf_topic_weight": "Positive weight for the paper topic RRF channel (default: 1)",
    "rrf_keyword_weight": "Positive weight for the paper keyword RRF channel (default: 1)",
    "keyword_rrf_tie_policy": "Equal-score keyword RRF policy: ordinal or mean_reciprocal",
    "neighbor_score_scale": "Scale for one-hop neighbor scores, in [0, 1] (default: 1)",
    "evidence_pool_version": "Evidence-pool update variant",
    "evidence_selection_version": "Evidence selection: legacy RRF or original-query note cosine",
    "sub_query_version": "Sub-question answer prompt variant; query generation uses prompts.subquery",
    "macer_version": "MACER loop variant",
    "responder_llm_config_path": "Optional separate responder LLM configuration YAML path",
    "render_top_fraction": "Fraction retained in the evidence pool during rendering, in (0, 1]; removes other items",
    "categories": "Comma-separated question categories from 1 to 5; default: all categories",
    "sample_indices": "Comma-separated original conversation indices to evaluate",
    "enable_semantic_gating": "Enable semantic gating (default: enabled)",
    "topic_retrieval_version": "Topic retrieval implementation variant",
    "ablation": "Read-side ablation mode",
    "write_ablation": "Write-side ablation mode",
    "max_steps": "Maximum MACER iterations; 0 answers using the initial retrieved evidence",
    "cache_dir": "Cache root; the evaluator appends cached_mem_key_graph_v3/name",
    "log_dir": "Experiment log directory",
    "metrics_mode": "CPU-only F1/BLEU-1: historical (default) or normalized F1 tokenization",
    "seed": "Optional nonnegative seed for construction and QA option order",
}

# Keep routine usage readable; implementation variants remain configurable in YAML.
COMMON_FIELDS = {
    "dataset", "name", "model", "llm_config_path", "prompt_config_path",
    "base_url", "api_key", "sample_indices", "categories", "ratio",
    "round_top_k", "final_top_k", "max_steps", "context_char_limit",
    "output", "cache_dir", "log_dir", "seed", "metrics_mode",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MemCoRe on LoCoMo or a conversation dataset in LoCoMo format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Default profile: %s\n"
               "Precedence: schema defaults < default profile < --config < explicit CLI options.\n"
               "YAML paths are relative to each YAML file; CLI paths are relative to the current directory." % DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--config", help="Experiment YAML (default: configs/locomo.yaml)")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", "--dry_run", action="store_true", help="Print redacted resolved config without loading runtime")
    actions.add_argument("--check", action="store_true", help="Validate local dataset and service config files")
    for field in fields(ExperimentConfig):
        name = field.name
        flags = list(dict.fromkeys(["--" + name, "--" + name.replace("_", "-")]))
        kwargs = {"dest": name, "default": argparse.SUPPRESS,
                  "help": FIELD_HELP[name] if name in COMMON_FIELDS else argparse.SUPPRESS}
        if name in BOOL_FIELDS:
            parser.add_argument(*flags, action="store_true", **kwargs)
            negative = list(dict.fromkeys(["--no_" + name, "--no-" + name.replace("_", "-")]))
            if name == "enable_semantic_gating":
                negative.extend(["--no_semantic_gating", "--no-semantic-gating"])
            negative_kwargs = dict(kwargs, help=("Disable " + name.replace("_", " ")
                                                if name in COMMON_FIELDS else argparse.SUPPRESS))
            parser.add_argument(*negative, action="store_false", **negative_kwargs)
        else:
            if name in ("final_top_k", "round_top_k"):
                kwargs["type"] = lambda value: None if value.lower() == "none" else int(value)
            elif name in INT_FIELDS:
                kwargs["type"] = int
            elif name in FLOAT_FIELDS:
                kwargs["type"] = float
            if name in CHOICES:
                kwargs["choices"] = CHOICES[name]
            parser.add_argument(*flags, **kwargs)
    return parser


def _check_files(config):
    for key in ("dataset", "llm_config_path", "prompt_config_path", "responder_llm_config_path"):
        value = getattr(config, key)
        if value is not None and not Path(value).is_file():
            raise ConfigError("%s must point to an existing file" % key)


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = vars(parser.parse_args(argv))
    except SystemExit as exc:
        return int(exc.code)
    path = args.pop("config")
    dry_run = args.pop("dry_run")
    check = args.pop("check")
    try:
        config = load_config(path, args)
        if dry_run:
            print(json.dumps(config.to_dict(redact=True), indent=2, ensure_ascii=False))
            return 0
        if check:
            _check_files(config)
            from .runner import validate_inputs
            validate_inputs(config)
            print("Local input validation passed.")
            return 0
        from .runner import run_experiment
        result = run_experiment(config)
        if isinstance(result, dict) and result.get("complete") is False:
            print("memcore: experiment is incomplete; resume with the same configuration", file=sys.stderr)
            return 1
        return 0
    except ModuleNotFoundError as exc:
        module = exc.name or "unknown"
        print("memcore: Missing module '%s'. Install dependencies with: "
              "python -m pip install -r requirements.txt." % module,
              file=sys.stderr)
        return 2
    except (ConfigError, ValueError, OSError) as exc:
        # Never interpolate values from a runtime error that may include credentials.
        if isinstance(exc, ConfigError):
            message = str(exc)
        else:
            message = "Unable to load or validate experiment inputs (%s)" % type(exc).__name__
        print("memcore: " + message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
