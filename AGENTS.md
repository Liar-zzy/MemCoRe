# Agent guide to MemCoRe

This guide applies to the entire repository. MemCoRe is a research release for
graph memory construction, evidence retrieval, and LoCoMo evaluation. Keep routine
changes small and preserve the reproduction settings unless the task calls for
changing the method.

## Start here

- Read [README.md](README.md) or [README_zh.md](README_zh.md) for setup, model serving,
  and the method figure. The [paper](https://arxiv.org/abs/2602.07885) describes the method.
- Run commands from the repository root. Use Python 3.11 and `python run.py` directly;
  the project does not require package installation.
- The evaluator runs on CPU and calls separate chat and embedding services.
  Graph storage is in-process, with NumPy vector search and JSON snapshots;
  NetworkX/graspologic handle graph analysis and topic clustering. No Neo4j service is needed.
- `versions/v1/`, `key_graph_v3`, and class names ending in `V3` are existing internal
  identifiers. The public project name is **MemCoRe**; keep those identifiers stable
  during routine fixes.

## Setup and verification

Use an existing compatible environment, or create one:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m nltk.downloader punkt punkt_tab
```

These checks do not call model services:

```bash
python run.py --dataset examples/smoke_locomo.json --check
python run.py --dry-run
python -m pytest -q
```

`--check` validates local inputs and service configuration; it does not establish
endpoint connectivity. `--dry-run` prints the resolved experiment configuration.
The tests use fake model responses, including mock HTTP requests.

With both model services configured, run the small example:

```bash
python run.py --dataset examples/smoke_locomo.json --name agent_smoke
python scripts/summarize.py results/agent_smoke
```

This makes real model calls for six conversation turns and five questions.
For LoCoMo, download data with `python scripts/download_data.py`. Then use
`python run.py --sample-indices 0 --name locomo_sample0` for one conversation,
or `python run.py` for the default full run: 10 conversations and 1,986 questions.
Model-serving commands are in the README; use a separate environment for vLLM.

## Configuration and model selection

[configs/locomo.yaml](configs/locomo.yaml) is the default experiment profile.
Precedence is schema defaults, default profile, `--config`, then explicit CLI options.
Paths inside YAML are relative to that YAML file; CLI paths are relative to the
working directory. `--config` selects an experiment profile, while
`--llm-config-path` selects chat and embedding services.

| Chat model | Service configuration | Connection |
| --- | --- | --- |
| Qwen3-8B | [configs/llm.yaml](configs/llm.yaml) | `MEMCORE_LLM_BASE_URL`, optional `MEMCORE_LLM_API_KEY` |
| Qwen3-14B | [configs/llm_qwen3_14b.yaml](configs/llm_qwen3_14b.yaml) | Same variables, with a server serving the 14B model |
| GPT-4o | [configs/llm_gpt4o.yaml](configs/llm_gpt4o.yaml) | `OPENAI_API_KEY`; optional CLI `--base-url` / `--model` overrides |

Select a service file with `--llm-config-path` and use a new `--name` when switching
models. Use the GPT-4o configuration when connecting to OpenAI: changing only
`--model` in the Qwen profile would retain Qwen/vLLM-specific request parameters.
Check the provider's current model availability before a real run.

All three profiles use **Qwen3-Embedding-0.6B**, with 1,024-dimensional vectors,
through `MEMCORE_EMBEDDING_BASE_URL` and optional `MEMCORE_EMBEDDING_API_KEY`.
Keep construction and retrieval on the same embedding model. The chat configuration
controls memory construction, query analysis, and answering by default.
Environment variables must be exported; `.env` is not loaded automatically.

## Reproduction settings to preserve

| Setting | Release default and meaning |
| --- | --- |
| Retrieval | `retrieval_version: paper`, `evidence_selection_version: legacy_rrf` |
| Topic retrieval | Top 3 topic nodes |
| Keyword retrieval | Top 10 keyword nodes **per query keyword/entity** |
| `round_top_k` | 60 notes after fusion and neighbor expansion in each retrieval round |
| `final_top_k` | 60 notes in the accumulated evidence pool across rounds |
| `context_char_limit` | 90,000 **characters**, not tokens |
| `max_steps` | Up to 3 answer/reflection iterations; at most 2 additional retrieval rounds |
| Chat generation | Qwen thinking disabled; request seed 42, top-p 1.0 |
| Metrics | `core_historical_v1`: historical set-based F1 and NLTK BLEU-1 |

Topic and keyword RRF weights are both 1. `retrieve_k` controls historical retrieval
variants; it does not set the evidence budget for `paper` retrieval. Use
`round_top_k` and `final_top_k` for that purpose. Preserve the prompt selections in
[configs/prompts.yaml](configs/prompts.yaml) during reproduction work.

The default Overall metric averages over individual questions across all five
categories. `core_normalized_v1` deliberately changes F1 text normalization;
`full` metrics mode is unavailable. When changing defaults, keep the YAML profile,
configuration schema, and relevant function defaults consistent.

## Code navigation

| Task | Start with |
| --- | --- |
| CLI, option validation, model configuration | [cli.py](src/memcore/cli.py), [config.py](src/memcore/config.py), [settings.py](src/memcore/settings.py) |
| Runtime setup, cache compatibility, saved run configuration | [runner.py](src/memcore/runner.py) |
| Dataset loop, checkpoint/resume, scoring | [evaluation.py](src/memcore/versions/v1/evaluation.py) |
| Note construction, graph evolution, topics, `retrieve_paper_structured` | [key_graph_memory_v3.py](src/memcore/versions/v1/agent/key_graph_memory_v3.py) |
| Iterative answering, evidence pool, context rendering | [key_agent_v3.py](src/memcore/versions/v1/agent/key_agent_v3.py) |
| Chat/embedding requests and structured-output compatibility | [memory_layer.py](src/memcore/versions/v1/agent/memory_layer.py) |
| Graph nodes, edges, vector search, snapshots | [graph_store.py](src/memcore/versions/v1/storage/graph_store.py) |
| Primary metrics and aggregation | [core_metrics.py](src/memcore/versions/v1/utils/core_metrics.py), [utils.py](src/memcore/versions/v1/utils/utils.py) |

The main entry is [run.py](run.py), which loads the CLI and runner before calling
`evaluate_dataset`. Prefer that entry for experiments so configuration expansion
and resume checks are applied consistently. Related tests live under [tests/](tests/).

## Runs, outputs, and changes

- Repeat the same command to resume an interrupted run. Use a new `--name` when
  changing the dataset, model, prompts, or experiment settings. Separate concurrent
  experiments should have separate names.
- Results are under `results/<name>/`, including `run_config.yaml`, per-conversation
  checkpoints, and `final_results.json`. Graphs are under
  `cached/cached_mem_key_graph_v3/<name>/graph_sample_<index>.json`.
- Use `scripts/summarize.py` to inspect completion, question counts, and metrics.
  An incomplete run returns a nonzero exit status. Report partial results as partial;
  passing offline tests does not establish real-model quality or full-run scores.
- For code changes, run the relevant tests and the offline suite before handing off.
  For documentation-only edits, verify paths and commands without rerunning model
  experiments. Keep the English and Chinese README instructions aligned when updating them.
- If changing persisted graph contents or QA checkpoint behavior, inspect compatibility
  handling in `runner.py` and [protocol.py](src/memcore/versions/v1/protocol.py).
- Keep the checkout self-contained. Preserve the existing ignore rules for datasets,
  caches, logs, results, and credentials; keep private machine paths and research
  archives out of release files.
- Try direct model/dataset downloads first. Consider a proxy only after direct access
  fails; do not change global proxy settings unless requested.
- Do not compute or add checksums or hash manifests unless explicitly requested.
