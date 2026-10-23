"""Connect validated configuration to a versioned evaluator."""

from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit

import yaml

from .config import ConfigError
from .registry import get_version, load_evaluator
from .settings import expand_environment, load_yaml, redact


CONSTRUCTION_REVISION = "inmemory-paper-context-centroid-bounded-topic-v0.2.0"
_SECRET_KEYS = {"api_key", "password", "secret", "token", "access_token", "auth_token",
                "authorization", "credentials", "credential", "client_secret"}


def _public_config(value):
    """Record effective model settings without letting credentials define identity."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in _SECRET_KEYS or key_lower.endswith(("_api_key", "_password", "_secret", "_access_token")):
                continue
            if str(key).lower() in {"base_url", "url", "endpoint"} and isinstance(item, str):
                parsed = urlsplit(item)
                host = parsed.hostname or ""
                if parsed.port is not None:
                    host += f":{parsed.port}"
                result[key] = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
            else:
                result[key] = _public_config(item)
        return result
    if isinstance(value, list):
        return [_public_config(item) for item in value]
    return value


def _cache_contract(config, documents):
    dataset = Path(config.dataset).resolve()
    stat = dataset.stat()
    services = documents["llm"]
    return {
        "construction_revision": CONSTRUCTION_REVISION,
        "version": config.version,
        "mode": config.mode,
        "graph_backend": config.graph_backend,
        "construction_policy": {
            "note_embedding": "context_only",
            "refresh_embedding_after_context_update": True,
            "topic_embedding": "all_member_keyword_centroid",
            "topic_min_keywords": 3,
            "topic_max_keywords": 15,
            "leiden_assignment": "final_clusters",
            "topic_partition": "deterministic_affinity_cardinality_postprocess_v1",
            "candidate_selection": "legacy_adaptive_funnel",
        },
        "dataset": {"path": str(dataset), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "llm": _public_config(services["llm"]),
        "embedding": _public_config(services["embedding"]),
        "prompt_selectors": documents["prompts"]["prompts"],
        "enable_semantic_gating": config.enable_semantic_gating,
        "write_ablation": config.write_ablation,
        "construction_seed": config.seed,
    }


def _check_backup_vectors(cache_path, embedding_model):
    expected = 1024 if "Qwen3-Embedding-0.6B" in embedding_model else None
    observed = None
    for backup_path in sorted(cache_path.glob("graph_sample_*.json")):
        try:
            with backup_path.open(encoding="utf-8") as stream:
                backup = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Invalid graph cache {backup_path}: {exc}") from None
        graph = backup.get("graph") if isinstance(backup, dict) else None
        if (not isinstance(backup, dict) or backup.get("schema_version") != 1 or not isinstance(graph, dict)
                or not isinstance(graph.get("nodes"), list)):
            raise ConfigError(f"Invalid graph cache nodes: {backup_path}")
        for index, node in enumerate(graph["nodes"]):
            if not isinstance(node, dict) or not isinstance(node.get("properties"), dict):
                raise ConfigError(f"Invalid graph cache node {index}: {backup_path}")
            if node.get("label") not in {"MemoryNote", "Keyword", "Topic"}:
                continue
            vector = node["properties"].get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ConfigError(f"Missing embedding in graph cache node {index}: {backup_path}")
            if expected is not None and len(vector) != expected:
                raise ConfigError(f"Graph cache embedding dimension {len(vector)} != {expected}: {backup_path}")
            if observed is not None and len(vector) != observed:
                raise ConfigError(f"Inconsistent graph cache embedding dimensions: {backup_path}")
            observed = len(vector)
        if observed is not None and graph.get("embedding_dim") not in (None, observed):
            raise ConfigError(f"Graph cache embedding_dim differs from nodes: {backup_path}")
        if expected is not None and graph.get("embedding_dim") not in (None, expected):
            raise ConfigError(f"Graph cache embedding_dim differs from configured model: {backup_path}")


def _ensure_cache_contract(config, documents):
    cache_path = Path(config.cache_dir) / f"cached_mem_{config.mode}" / config.name
    metadata_path = cache_path / "cache_contract.json"
    current = _cache_contract(config, documents)
    legacy = (list(cache_path.glob("*.pkl")) + list(cache_path.glob("neo4j*.json"))
              + list(cache_path.glob("neo4j*.json.gz")))
    if legacy:
        raise ConfigError("Legacy Neo4j/pickle cache cannot be mixed with in-memory graphs; choose a new --cache-dir or --name")
    if metadata_path.exists():
        try:
            with metadata_path.open(encoding="utf-8") as stream:
                previous = json.load(stream)
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Invalid cache_contract.json: {exc}") from None
        if previous != current:
            raise ConfigError("Cache construction contract differs; choose a new --cache-dir or --name")
    else:
        if list(cache_path.glob("graph_sample_*.json")):
            raise ConfigError("Cache exists without cache_contract.json; choose a new --cache-dir or --name")
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            with metadata_path.open("x", encoding="utf-8") as stream:
                json.dump(current, stream, ensure_ascii=False, indent=2, sort_keys=True)
        except FileExistsError:
            raise ConfigError("Cache contract was created concurrently; rerun to validate it") from None
    _check_backup_vectors(cache_path, documents["llm"]["embedding"]["model"])


def _section(document, key, source, required=()):
    value = document.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: expected a '{key}' mapping")
    for field in required:
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ConfigError(f"{source}: '{key}.{field}' must be a nonempty string")
    return value


def validate_inputs(config):
    """Check local files and service/prompt schemas. Does not contact services."""
    version = get_version(config.version)
    for field in ("dataset", "llm_config_path", "prompt_config_path",
                  "responder_llm_config_path"):
        path = getattr(config, field)
        if path is not None and not Path(path).is_file():
            raise ConfigError(f"{field}: file does not exist: {path}")

    documents = {
        "llm": load_yaml(config.llm_config_path, resolve_environment=False),
        "prompts": load_yaml(config.prompt_config_path),
    }
    # Materialize CLI overrides for the config that the memory constructor
    # actually reads, as well as the retriever controller used by the evaluator.
    llm = _section(documents["llm"], "llm", config.llm_config_path)
    overrides = {}
    for field in ("model", "backend", "api_key", "base_url"):
        value = getattr(config, field)
        if value is not None:
            overrides[field] = value
            llm.pop(field, None)
    documents["llm"] = expand_environment(documents["llm"])
    documents["llm"]["llm"].update(overrides)
    llm = _section(documents["llm"], "llm", config.llm_config_path, ("model", "backend"))
    embedding = _section(documents["llm"], "embedding", config.llm_config_path, ("model", "backend"))
    if llm["backend"] not in {"openai", "ollama"}:
        raise ConfigError("llm.backend must be openai or ollama")
    if embedding["backend"] != "openai":
        raise ConfigError("embedding.backend must be openai for the v1 implementation")
    prompts = import_module(version["prompts"])
    choices = _section(documents["prompts"], "prompts", config.prompt_config_path)
    for category, selection in choices.items():
        if category not in prompts.PROMPT_REGISTRY:
            raise ConfigError(f"Unknown prompt category: {category}")
        if not isinstance(selection, str) or selection not in prompts.PROMPT_REGISTRY[category]:
            raise ConfigError(f"Unknown prompt version for {category}")
    if config.responder_llm_config_path is not None:
        documents["responder"] = load_yaml(config.responder_llm_config_path)
        responder = _section(documents["responder"], "llm", config.responder_llm_config_path,
                             ("backend", "model"))
        if responder["backend"] not in {"openai", "ollama"}:
            raise ConfigError("responder llm.backend must be openai or ollama")
    return documents


def run_experiment(config):
    """Run v1 with the same argument contract from either CLI or YAML."""
    documents = validate_inputs(config)
    _ensure_cache_contract(config, documents)
    # Import before writing results, so missing optional runtime dependencies
    # do not create a misleading experiment record.
    evaluate = load_evaluator(config.version)
    result_dir = Path(config.output) / config.name
    result_dir.mkdir(parents=True, exist_ok=True)
    from .versions.v1.protocol import EXECUTION_REVISION

    snapshot = {
        "experiment": config.to_dict(redact=True),
        "services": redact(documents),
        "execution_revision": EXECUTION_REVISION,
    }
    snapshot_path = result_dir / "run_config.yaml"
    if snapshot_path.exists():
        previous = load_yaml(snapshot_path, resolve_environment=False)
        if previous != snapshot:
            raise ConfigError("This result directory has a different run_config.yaml; choose a new --name or --output")
    else:
        snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # The old core accepts config filenames. Private temporary files let every
    # consumer see the same resolved settings without changing user files.
    with TemporaryDirectory(prefix="memcore-config-") as directory:
        paths = {}
        for role, document in documents.items():
            path = Path(directory) / f"{role}.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
            path.chmod(0o600)
            paths[role] = str(path)
        kwargs = config.to_dict()
        for field in ("schema_version", "version", "responder_llm_config_path"):
            kwargs.pop(field, None)
        kwargs["dataset_path"] = kwargs.pop("dataset")
        kwargs["output_path"] = kwargs.pop("output")
        kwargs["llm_config_path"] = paths["llm"]
        kwargs.pop("neo4j_config_path", None)
        kwargs["prompt_config_path"] = paths["prompts"]
        kwargs["responder_llm_config"] = documents.get("responder", {}).get("llm")
        return evaluate(**kwargs)
