"""Lightweight, strictly validated experiment configuration.

No runtime or service dependencies are imported by this module.
"""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .registry import VERSIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "locomo.yaml"


class ConfigError(ValueError):
    """Invalid experiment configuration."""


@dataclass
class ExperimentConfig:
    version: str = "v1"
    dataset: str = str(PROJECT_ROOT / "data" / "locomo10.json")
    model: Optional[str] = None
    output: str = str(PROJECT_ROOT / "results")
    ratio: float = 1.0
    backend: Optional[str] = None
    temperature_c5: float = 0.5
    retrieve_k: int = 10
    round_top_k: Optional[int] = 60
    final_top_k: Optional[int] = 60
    context_char_limit: int = 90000
    evidence_selection_version: str = "legacy_rrf"
    mode: str = "key_graph_v3"
    graph_backend: str = "inmemory"
    llm_config_path: str = str(PROJECT_ROOT / "configs" / "llm.yaml")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    name: str = "locomo_qwen3_8b"
    sort_by: str = "time_asc"
    prompt_config_path: str = str(PROJECT_ROOT / "configs" / "prompts.yaml")
    use_format_instruction: bool = True
    retrieval_version: str = "paper"
    rrf_topic_weight: float = 1.0
    rrf_keyword_weight: float = 1.0
    keyword_rrf_tie_policy: str = "ordinal"
    neighbor_score_scale: float = 1.0
    evidence_pool_version: str = "v2"
    sub_query_version: str = "v2"
    macer_version: str = "v1"
    responder_llm_config_path: Optional[str] = None
    render_top_fraction: float = 1.0
    categories: Optional[List[int]] = None
    sample_indices: Optional[List[int]] = None
    enable_semantic_gating: bool = True
    topic_retrieval_version: str = "v1"
    ablation: str = "none"
    write_ablation: str = "none"
    max_steps: int = 3
    metrics_mode: str = "core_historical_v1"
    seed: Optional[int] = 42
    cache_dir: str = str(PROJECT_ROOT / "cached")
    log_dir: str = str(PROJECT_ROOT / "logs")

    def to_dict(self, redact: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        if redact and result["api_key"] is not None:
            result["api_key"] = "***REDACTED***"
        return result


CHOICES = {
    "version": tuple(VERSIONS),
    "backend": ("openai", "ollama"),
    "mode": ("key_graph_v3",),
    "graph_backend": ("inmemory",),
    "sort_by": ("score", "time", "time_asc"),
    "retrieval_version": ("paper",) + tuple("v%d" % n for n in range(1, 9)),
    "evidence_pool_version": ("v1", "v2"),
    "sub_query_version": ("v1", "v2"),
    "macer_version": ("v1", "v2"),
    "topic_retrieval_version": ("v1", "v2"),
    "ablation": ("none", "no_topic", "no_neighbor", "no_loop", "no_kws"),
    "write_ablation": ("none", "no_consolidation", "no_merge", "no_link_write", "no_denoising"),
    "metrics_mode": ("core_historical_v1", "core_normalized_v1"),
    "evidence_selection_version": ("legacy_rrf", "query_note_cosine_v1"),
    "keyword_rrf_tie_policy": ("ordinal", "mean_reciprocal"),
}
PATH_FIELDS = {
    "dataset", "output", "llm_config_path",
    "prompt_config_path", "responder_llm_config_path", "cache_dir", "log_dir",
}
BOOL_FIELDS = {"use_format_instruction", "enable_semantic_gating"}
INT_FIELDS = {"retrieve_k", "round_top_k", "max_steps", "seed", "final_top_k", "context_char_limit"}
FLOAT_FIELDS = {"ratio", "temperature_c5", "render_top_fraction",
                "rrf_topic_weight", "rrf_keyword_weight", "neighbor_score_scale"}
OPTIONAL_FIELDS = {"model", "backend", "api_key", "base_url", "responder_llm_config_path", "categories", "sample_indices", "seed", "round_top_k", "final_top_k"}


def _categories(value: Any, allow_string: bool) -> Optional[List[int]]:
    if value is None:
        return None
    if allow_string and isinstance(value, str):
        try:
            value = [int(part.strip()) for part in value.split(",")]
        except ValueError:
            raise ConfigError("categories must be comma-separated integers from 1 to 5") from None
    if not isinstance(value, list) or not value or any(type(x) is not int or x not in range(1, 6) for x in value):
        raise ConfigError("categories must be a nonempty list of integers from 1 to 5")
    return list(value)


def _sample_indices(value: Any, allow_string: bool) -> Optional[List[int]]:
    if value is None:
        return None
    if allow_string and isinstance(value, str):
        try:
            value = [int(part.strip()) for part in value.split(",")]
        except ValueError:
            raise ConfigError("sample_indices must be comma-separated nonnegative integers") from None
    if (not isinstance(value, list) or not value or
            any(type(x) is not int or x < 0 for x in value) or len(set(value)) != len(value)):
        raise ConfigError("sample_indices must be a nonempty list of distinct nonnegative integers")
    return list(value)


def _layer(values: Mapping[str, Any], base: Path, cli: bool = False) -> Dict[str, Any]:
    valid = {field.name for field in fields(ExperimentConfig)}
    if any(not isinstance(key, str) for key in values):
        raise ConfigError("configuration keys must be strings")
    unknown = set(values) - valid - {"schema_version"}
    if unknown:
        raise ConfigError("unknown configuration field(s): " + ", ".join(sorted(unknown)))
    if "schema_version" in values and (type(values["schema_version"]) is not int or values["schema_version"] != 1):
        raise ConfigError("schema_version must be the integer 1")
    result = {}
    for key, value in values.items():
        if key == "schema_version":
            continue
        if value is None:
            if key not in OPTIONAL_FIELDS:
                raise ConfigError("%s cannot be null" % key)
        elif key == "categories":
            value = _categories(value, allow_string=cli)
        elif key == "sample_indices":
            value = _sample_indices(value, allow_string=cli)
        elif key in BOOL_FIELDS:
            if type(value) is not bool:
                raise ConfigError("%s must be a boolean" % key)
        elif key in INT_FIELDS:
            if type(value) is not int:
                raise ConfigError("%s must be an integer" % key)
        elif key in FLOAT_FIELDS:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigError("%s must be a finite number" % key)
            value = float(value)
        elif not isinstance(value, str):
            raise ConfigError("%s must be a string" % key)
        if key == "name" and (not value.strip() or value in (".", "..") or any(character in value for character in ("/", "\\", "\x00"))):
            raise ConfigError("name must be a nonempty single path component other than '.' or '..'")
        if key in ("model", "base_url") and value is not None and not value.strip():
            raise ConfigError("%s must be a nonempty string when provided" % key)
        if value is not None and key in CHOICES and value not in CHOICES[key]:
            raise ConfigError("%s must be one of: %s" % (key, ", ".join(CHOICES[key])))
        if value is not None and key in PATH_FIELDS:
            if not value.strip():
                raise ConfigError("%s must be a nonempty path" % key)
            path = Path(value).expanduser()
            value = str((base / path).resolve())
        result[key] = value
    # Validate each layer, so an override cannot silently hide malformed YAML.
    for key in ("ratio", "render_top_fraction"):
        if key in result and not 0 < result[key] <= 1:
            raise ConfigError("%s must be in (0, 1]" % key)
    if "retrieve_k" in result and result["retrieve_k"] <= 0:
        raise ConfigError("retrieve_k must be positive")
    if result.get("round_top_k") is not None and result["round_top_k"] <= 0:
        raise ConfigError("round_top_k must be positive or null to inherit final_top_k")
    if "final_top_k" in result and result["final_top_k"] is not None and result["final_top_k"] <= 0:
        raise ConfigError("final_top_k must be positive or null for the legacy unbounded pool")
    if "context_char_limit" in result and result["context_char_limit"] <= 0:
        raise ConfigError("context_char_limit must be positive")
    if "max_steps" in result and result["max_steps"] < 0:
        raise ConfigError("max_steps must be nonnegative")
    if "seed" in result and result["seed"] is not None and result["seed"] < 0:
        raise ConfigError("seed must be nonnegative")
    if "temperature_c5" in result and result["temperature_c5"] < 0:
        raise ConfigError("temperature_c5 must be nonnegative")
    for key in ("rrf_topic_weight", "rrf_keyword_weight"):
        if key in result and result[key] <= 0:
            raise ConfigError("%s must be positive" % key)
    if "neighbor_score_scale" in result and not 0 <= result["neighbor_score_scale"] <= 1:
        raise ConfigError("neighbor_score_scale must be in [0, 1]")
    return result


def _load_yaml_layer(config_path: Path) -> Dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
    except yaml.YAMLError:
        # Parser diagnostics can echo secret-bearing YAML source lines.
        raise ConfigError("invalid YAML in experiment configuration") from None
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ConfigError("experiment configuration must be a flat YAML mapping")
    return _layer(values, config_path.parent)


def load_config(path=None, overrides=None) -> ExperimentConfig:
    """Resolve defaults < configs/locomo.yaml < user YAML < CLI.

    YAML paths are relative to the YAML file; override paths are relative to
    the caller's current directory. This only validates values, not file existence.
    """
    result = ExperimentConfig().to_dict()
    default_path = DEFAULT_CONFIG_PATH.resolve()
    result.update(_load_yaml_layer(default_path))
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        if config_path != default_path:
            result.update(_load_yaml_layer(config_path))
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("overrides must be a mapping")
        result.update(_layer(overrides, Path.cwd(), cli=True))
    config = ExperimentConfig(**result)
    if config.retrieval_version == "paper" and config.retrieve_k != 10:
        raise ConfigError("retrieve_k only controls legacy retrieval; use round_top_k for paper retrieval")
    if config.round_top_k is not None and (
            config.retrieval_version != "paper" or config.evidence_selection_version != "legacy_rrf"):
        raise ConfigError("round_top_k requires paper retrieval and legacy_rrf evidence selection")
    weighted = (config.rrf_topic_weight != 1.0 or config.rrf_keyword_weight != 1.0
                or config.neighbor_score_scale != 1.0)
    if weighted and (config.retrieval_version != "paper"
                     or config.evidence_selection_version != "legacy_rrf"):
        raise ConfigError("retrieval weights require paper retrieval and legacy_rrf evidence selection")
    if config.keyword_rrf_tie_policy != "ordinal" and (
            config.retrieval_version != "paper"
            or config.evidence_selection_version != "legacy_rrf"):
        raise ConfigError("keyword_rrf_tie_policy requires paper retrieval and legacy_rrf evidence selection")
    if config.evidence_selection_version == "query_note_cosine_v1" and (
            config.retrieval_version != "paper" or config.evidence_pool_version != "v2"
            or config.final_top_k is None or config.render_top_fraction != 1.0):
        raise ConfigError("query_note_cosine_v1 requires paper retrieval, v2 pool, positive final_top_k and render_top_fraction=1")
    return config
