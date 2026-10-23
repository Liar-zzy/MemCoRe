from memcore.versions.v1.agent.memory_layer import LLMController
import os
import json
import argparse
import logging
import math
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import numpy as np
from memcore.versions.v1.data.load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from collections import defaultdict
import random
from tqdm import tqdm
from memcore.versions.v1.utils.utils import aggregate_metrics
from memcore.versions.v1.utils.core_metrics import calculate_core_metrics
from memcore.versions.v1.protocol import EXECUTION_REVISION
from datetime import datetime
import yaml
from memcore.versions.v1.prompts.prompt_loader import PromptLoader
import re
import time as time_module
from memcore.versions.v1.utils.token_tracker import get_construction_tracker, get_retrieval_tracker, enable_tracking

# This version contains the KeyGraphV3 evaluation path.
from memcore.versions.v1.agent.key_graph_memory_v3 import KeyGraphMemoryV3
from memcore.versions.v1.agent.key_agent_v3 import KeyToGAgentV3

KEY_V3_AVAILABLE = True
KEYWORD_AVAILABLE = False

def load_llm_config(config_path: str = "llm_config.yaml") -> Dict:
    """Load LLM configuration from YAML file.
    
    Args:
        config_path: Path to the LLM configuration file
        
    Returns:
        Dictionary containing LLM configuration
    """
    default_config = {
        "backend": "openai",
        "model": "Qwen/Qwen3-8B",
        "api_key": "empty",
        "base_url": None
    }
    
    # Convert relative path to absolute if needed
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                llm_config = config.get("llm", {})
                # Merge with defaults
                result = {**default_config, **llm_config}
                return result
        except Exception as e:
            print(f"Warning: Failed to load LLM config from {config_path}: {e}. Using defaults.")
            return default_config
    else:
        print(f"Warning: LLM config file not found at {config_path}. Using defaults.")
        return default_config

def compute_metric_averages(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算指标的平均值
    
    Args:
        entries: 包含 metrics 字段的结果列表
        
    Returns:
        指标名称到平均值的字典
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for item in entries:
        metrics = item.get("metrics", {})
        for name, value in metrics.items():
            if value is None:
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1

    return {name: totals[name] / counts[name] for name in totals if counts[name]}


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Replace a checkpoint only after its complete JSON has reached disk."""
    import tempfile

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.memcore-', suffix='.tmp', delete=False) as stream:
            temporary_path = stream.name
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validated_sample_checkpoint(data: Dict[str, Any], sample_idx: int,
                                 sample: Any, allow_categories: List[int]) -> Dict[str, Any]:
    """Accept only records that still identify the same source QA positions."""
    if not isinstance(data, dict) or data.get('sample_id') != sample_idx:
        raise ValueError('checkpoint sample_id mismatch')
    entries = data.get('qa_results')
    if not isinstance(entries, list):
        raise ValueError('checkpoint qa_results must be a list')
    expected = [(idx, qa) for idx, qa in enumerate(sample.qa)
                if int(qa.category) in allow_categories]
    expected_by_index = dict(expected)
    legacy = all(isinstance(entry, dict) and 'qa_index' not in entry for entry in entries)
    if legacy and len(entries) > len(expected):
        raise ValueError('legacy checkpoint has extra results')
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('checkpoint contains an invalid QA result')
    if not legacy and any('qa_index' not in entry for entry in entries):
        raise ValueError('checkpoint mixes indexed and legacy QA results')
    restored = {}
    for position, entry in enumerate(entries):
        qa_index = expected[position][0] if legacy else entry['qa_index']
        if type(qa_index) is not int or qa_index not in expected_by_index or qa_index in restored:
            raise ValueError('checkpoint QA index is invalid or duplicated')
        qa = expected_by_index[qa_index]
        if (entry.get('sample_id') != sample_idx or
                entry.get('question') != qa.question or
                entry.get('category') != qa.category or
                entry.get('reference') != qa.final_answer or
                not isinstance(entry.get('metrics'), dict) or
                not {'f1', 'bleu1', 'retrieval_recall', 'retrieval_hit_rate'}.issubset(entry['metrics'])):
            raise ValueError('checkpoint QA identity or metrics mismatch')
        restored[qa_index] = {**entry, 'qa_index': qa_index}
    if data.get('expected_qa') is not None and data['expected_qa'] != len(expected):
        raise ValueError('checkpoint expected QA count mismatch')
    if data.get('complete') is True and len(restored) != len(expected):
        raise ValueError('checkpoint falsely claims completion')
    errors = data.get('errors', [])
    if not isinstance(errors, list):
        raise ValueError('checkpoint errors must be a list')
    history = data.get('error_history', errors)
    if not isinstance(history, list):
        raise ValueError('checkpoint error_history must be a list')
    for error in [*errors, *history]:
        if (not isinstance(error, dict) or type(error.get('qa_index')) is not int
                or error['qa_index'] not in expected_by_index):
            raise ValueError('checkpoint error identity is invalid')
    # Old checkpoints used errors as an append-only list. Preserve every old
    # attempt in history, but only the latest unresolved error per QA is active.
    active_by_index = {error['qa_index']: error for error in errors
                       if error['qa_index'] not in restored}
    errors = [active_by_index[index] for index in sorted(active_by_index)]
    if data.get('complete') is True and errors:
        raise ValueError('checkpoint falsely claims completion with active errors')
    attempts = data.get('cost_attempts')
    if attempts is None:
        attempts = ([{'construction_cost': data.get('construction_cost'),
                      'retrieval_cost': data.get('retrieval_cost'), 'legacy': True}]
                    if data.get('construction_cost') is not None or
                       data.get('retrieval_cost') is not None else [])
    if not isinstance(attempts, list) or any(not isinstance(item, dict) for item in attempts):
        raise ValueError('checkpoint cost_attempts must be a list of mappings')
    return {'qa_results': restored, 'errors': errors,
            'error_history': list(history), 'cost_attempts': attempts}


def _total_token_cost(attempts: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    """Sum only observed tracker counters; raw attempts remain in the checkpoint."""
    total = {key: 0 for key in ('total_input_tokens', 'total_output_tokens',
                               'total_tokens', 'total_wall_time_s', 'call_count')}
    total['by_caller'] = {}
    for attempt in attempts:
        summary = attempt.get(phase)
        if not isinstance(summary, dict):
            continue
        for key in ('total_input_tokens', 'total_output_tokens', 'total_tokens',
                    'total_wall_time_s', 'call_count'):
            total[key] += summary.get(key, 0)
        for caller, values in summary.get('by_caller', {}).items():
            target = total['by_caller'].setdefault(caller, {
                'input_tokens': 0, 'output_tokens': 0, 'wall_time_s': 0, 'count': 0})
            for key in target:
                target[key] += values.get(key, 0)
    total['total_wall_time_s'] = round(total['total_wall_time_s'], 3)
    return total


def _save_sample_checkpoint(path: str, sample_idx: int, expected_qa: int,
                            records: Dict[int, Dict[str, Any]], errors: List[Dict[str, Any]],
                            cost_attempts: List[Dict[str, Any]],
                            failed_qa_index: Optional[int] = None,
                            error_history: Optional[List[Dict[str, Any]]] = None,
                            evidence_budget: Optional[Dict[str, Any]] = None) -> None:
    complete = (len(records) == expected_qa and not errors and failed_qa_index is None
                and not any(record.get('prediction_parse_error') for record in records.values()))
    payload = {
        'sample_id': sample_idx,
        'execution_revision': EXECUTION_REVISION,
        'evidence_budget': evidence_budget,
        'expected_qa': expected_qa,
        'total_qa': len(records),
        'complete': complete,
        'failed_qa_index': failed_qa_index,
        'errors': errors,
        'error_history': list(error_history if error_history is not None else errors),
        'qa_results': [records[index] for index in sorted(records)],
        'cost_attempts': cost_attempts,
        'construction_cost': _total_token_cost(cost_attempts, 'construction_cost'),
        'retrieval_cost': _total_token_cost(cost_attempts, 'retrieval_cost'),
    }
    if complete:
        payload['completed_at'] = datetime.now().isoformat()
    _write_json_atomic(path, payload)


def calculate_retrieval_metrics(gold_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
    """计算检索评估指标 (Recall, Precision, Hit Rate)
    
    Args:
        gold_ids: 黄金标准 ID 列表 (evidence)
        retrieved_ids: 系统检索到的 ID 列表
        
    Returns:
        字典包含:
        - recall: 召回率 = |gold ∩ retrieved| / |gold|
        - precision: 精确率 = |gold ∩ retrieved| / |retrieved|
        - hit_rate: 命中率 (1 if intersection > 0 else 0)
        - intersection_count: 交集大小
        - gold_count: 黄金标准数量
        - retrieved_count: 检索到的数量
    """
    gold_set = set(gold_ids) if gold_ids else set()
    retrieved_set = set(retrieved_ids) if retrieved_ids else set()
    
    intersection = gold_set & retrieved_set
    intersection_count = len(intersection)
    
    # Recall: 召回了多少黄金标准
    recall = intersection_count / len(gold_set) if gold_set else 0.0
    
    # Precision: 检索结果中有多少是正确的
    precision = intersection_count / len(retrieved_set) if retrieved_set else 0.0
    
    # Hit Rate: 是否至少命中一个
    hit_rate = 1.0 if intersection_count > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "hit_rate": hit_rate,
        "intersection_count": intersection_count,
        "gold_count": len(gold_set),
        "retrieved_count": len(retrieved_set)
    }

def clean_prediction(prediction: str) -> str:
    """去除prediction中的Option A/B/C/D: 前缀"""
    if not prediction:
        return prediction
    # 匹配 "Option A: ", "Option B: " 等格式
    pattern = r'^Option\s+[A-Za-z]:\s*'
    cleaned = re.sub(pattern, '', prediction)
    return cleaned

class advancedMemAgent:
    def __init__(self, model=None, backend=None, retrieve_k=10, temperature_c5=0.5,
                 mode="key_graph_v3", graph_backend: str = "inmemory",
                 llm_config_path="configs/llm.yaml", api_key=None, base_url=None,
                 logger: logging.Logger = None, sort_by: str = 'score',
                 prompt_config_path: str = None, use_format_instruction: bool = False,
                 retrieval_version: str = 'paper', evidence_pool_version: str = 'v2',
                 sub_query_version: str = 'v2', macer_version: str = 'v1',
                 responder_llm_config: dict = None,
                 render_top_fraction: float = 1.0,
                 enable_semantic_gating: bool = True,
                 topic_retrieval_version: str = 'v1',
                 max_steps: int = 3, final_top_k: Optional[int] = 60,
                 seed: Optional[int] = None, context_char_limit: int = 90000,
                 evidence_selection_version: str = 'legacy_rrf',
                 rrf_topic_weight: float = 1.0,
                 rrf_keyword_weight: float = 1.0,
                 neighbor_score_scale: float = 1.0,
                 keyword_rrf_tie_policy: str = 'ordinal',
                 round_top_k: Optional[int] = None):
        """Initialize the memory agent.
        
        Args:
            model: LLM model name (overrides config if provided)
            backend: LLM backend ('openai' or 'ollama') (overrides config if provided)
            retrieve_k: Legacy retrieval anchor count; paper retrieval uses round_top_k
            temperature_c5: Temperature for category 5 questions
            mode: Mode to use (amem/agraph/agraphrag/keyword_tog/key_graph_v2)
            graph_backend: Graph store backend; only inmemory is supported
            llm_config_path: Path to LLM configuration file (for Memory System)
            api_key: API key (overrides config if provided)
            base_url: Base URL (overrides config if provided)
            logger: 外部传入的 logger 实例
            prompt_config_path: Prompt配置文件路径，用于选择不同版本的prompts
            responder_llm_config: Responder LLM配置字典（用于双模型架构）
                - 如果提供，Agent将使用独立的LLM进行答案生成、反思和子查询
                - 如果为None，Agent将复用Memory System的LLM
            render_top_fraction: 渲染时只保留前 N% 分数的证据 (0.0-1.0)
                - 1.0: 保留全部（默认）
                - 0.33: 只保留前 1/3 分数的证据
            enable_semantic_gating: 是否启用 MemDyn 语义门控（默认 True）
                - True: MemDyn 模式，使用语义门控和加权拓扑
                - False: Baseline 模式，使用二值连接
            topic_retrieval_version: Topic 检索版本 ('v1' 或 'v2')
                - 'v1': 基础向量相似度检索
                - 'v2': 三角导航策略，结合向量相似度和关键词重叠
        """
        self.external_logger = logger
        self.prompt_config_path = prompt_config_path
        self.use_format_instruction = use_format_instruction
        self.retrieval_version = retrieval_version
        self.evidence_pool_version = evidence_pool_version
        self.sub_query_version = sub_query_version
        self.macer_version = macer_version
        self.responder_llm_config = responder_llm_config
        self.render_top_fraction = render_top_fraction
        self.enable_semantic_gating = enable_semantic_gating
        self.topic_retrieval_version = topic_retrieval_version
        self.max_steps = max_steps
        self.final_top_k = final_top_k
        if retrieval_version == 'paper' and retrieve_k != 10:
            raise ValueError('retrieve_k only controls legacy retrieval; use round_top_k for paper retrieval')
        if round_top_k is not None:
            if type(round_top_k) is not int or round_top_k <= 0:
                raise ValueError('round_top_k must be a positive integer or None')
            if retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf':
                raise ValueError('round_top_k requires paper retrieval and legacy_rrf evidence selection')
        self.round_top_k = round_top_k
        if type(context_char_limit) is not int or context_char_limit <= 0:
            raise ValueError("context_char_limit must be a positive integer")
        self.context_char_limit = context_char_limit
        self.evidence_selection_version = evidence_selection_version
        self.rrf_topic_weight = rrf_topic_weight
        self.rrf_keyword_weight = rrf_keyword_weight
        self.neighbor_score_scale = neighbor_score_scale
        self.keyword_rrf_tie_policy = keyword_rrf_tie_policy
        if graph_backend != "inmemory":
            raise ValueError("The v1 evaluator supports only the inmemory graph backend")
        # Load LLM config from file
        llm_config = load_llm_config(llm_config_path)
        
        # Use provided values or fall back to config
        backend = backend or llm_config.get("backend", "openai")
        model = model or llm_config.get("model", "Qwen/Qwen3-8B")
        api_key = api_key if api_key is not None else llm_config.get("api_key")
        base_url = base_url if base_url is not None else llm_config.get("base_url")
        
        # Initialize flag for keyword agent
        self.use_keyword_agent = False
        self.keyword_agent = None
        
        if mode == "key_graph_v3":
            # Each agent owns its in-memory graph store and can load one snapshot.
            if not KEY_V3_AVAILABLE:
                raise ImportError("KeyGraphMemoryV3 modules are not available. Please ensure key_graph_memory_v3.py and key_agent_v3.py are present in the agent/ directory.")
            
            # Disable auto-maintenance during batch loading (will trigger manually)
            self.memory_system = KeyGraphMemoryV3(
                llm_config_path=llm_config_path,
                use_neo4j=False,
                graph_backend=graph_backend,
                seed=seed,
                evo_threshold=100,
                topic_maintenance_threshold=9999,  # Disable auto-trigger for batch loading
                logger=self.external_logger,
                prompt_config_path=self.prompt_config_path,
                enable_semantic_gating=self.enable_semantic_gating,
                topic_retrieval_version=self.topic_retrieval_version
            )
            self.keyword_agent = KeyToGAgentV3(
                memory_system=self.memory_system,
                log_level=logging.INFO,
                logger=self.external_logger,
                prompt_config_path=self.prompt_config_path,
                use_format_instruction=self.use_format_instruction,
                retrieval_version=self.retrieval_version,
                evidence_pool_version=self.evidence_pool_version,
                sub_query_version=self.sub_query_version,
                responder_llm_config=self.responder_llm_config,
                render_top_fraction=self.render_top_fraction,
                final_top_k=self.final_top_k,
                round_top_k=self.round_top_k,
                context_char_limit=self.context_char_limit,
                evidence_selection_version=self.evidence_selection_version,
                rrf_topic_weight=self.rrf_topic_weight,
                rrf_keyword_weight=self.rrf_keyword_weight,
                neighbor_score_scale=self.neighbor_score_scale,
                keyword_rrf_tie_policy=self.keyword_rrf_tie_policy
            )
            self.use_keyword_agent = True
        self.retriever_llm = LLMController(backend=backend, model=model, api_key=api_key, base_url=base_url)
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.mode = mode
        self.sort_by = sort_by  # 检索结果排序方式: 'score', 'time', 'time_asc'

    def add_memory(self, content, time=None, source_id=None, write_ablation_flags: Dict[str, bool] = None):
        self.memory_system.add_note(content, time=time, source_id=source_id, write_ablation_flags=write_ablation_flags)

    def retrieve_memory(self, content, k=10):
        if self.mode == "key_graph_v3" and isinstance(self.memory_system, KeyGraphMemoryV3):
            # Use hybrid retrieval for KeyGraphMemoryV2 (Keyword + Topic paths)
            return self.memory_system.retrieve_topic_aware(
                query=content, 
                k_keywords=k,
                k_topics=max(5, k//2),
                k_final=k,
                enable_graph_expansion=True
            )
        else:
            print('Error')
        # elif self.use_keyword_agent:
        #     # Use topic-aware retrieval for keyword agent
        #     return self.memory_system.retrieve_topic_aware(content, k=k, max_topic_neighbors=3)
    
    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        # print("response:{}".format(response))
        return response
    
    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def _truncate_context(self, context: str, max_chars: int = 30000) -> str:
        """Truncate context to avoid exceeding token limits."""
        if len(context) <= max_chars:
            return context
        # Truncate and add indicator
        return context[:max_chars] + "\n\n[Context truncated due to length...]"
    
    def answer_question(self, question: str, category: int, answer: str, 
                        ablation_flags: Dict[str, bool] = None) -> str:
        """Generate answer for a question given the conversation context.
        
        Args:
            question: The question to answer
            category: Question category (1-5)
            answer: Ground truth answer for evaluation
            ablation_flags: Optional dict with ablation settings:
                - 'no_topic': Disable Topic-based retrieval
                - 'no_neighbor': Disable Neighbor expansion
                - 'no_loop': Single-step retrieval-generation (no MACER loop)
                - 'no_kws': Disable Keyword-based retrieval
        
        Returns:
            Tuple of (response, category_info, context_info, history_info, retrieved_source_ids)
            - For keyword agent modes, retrieved_source_ids is a list of source IDs from the evidence pool
            - For other modes, retrieved_source_ids is an empty list
        """
        
        # Call MACER loop for reasoning with category-aware prompts
        if self.macer_version == 'v2':
            macer_result = self.keyword_agent.macer_loop_v2(
                question=question,
                max_steps=self.max_steps,
                initial_k=self.retrieve_k,
                evolution_k=min(5, self.retrieve_k),
                max_topic_neighbors=3,
                category=category,
                ground_truth_answer=answer,
                temperature_c5=self.temperature_c5,
                sort_by=self.sort_by,
                ablation_flags=ablation_flags
            )
        else:
            macer_result = self.keyword_agent.macer_loop(
                question=question,
                max_steps=self.max_steps,
                initial_k=self.retrieve_k,
                evolution_k=min(5, self.retrieve_k),
                max_topic_neighbors=3,
                category=category,
                ground_truth_answer=answer,
                temperature_c5=self.temperature_c5,
                sort_by=self.sort_by,
                ablation_flags=ablation_flags
            )
        
        # Extract answer from MACER result
        prediction = macer_result.get("answer", "Unknown")
        history_str = macer_result.get("final_history_str", "Unknown")
        context = macer_result.get("final_context", "Unknown")
        retrieved_source_ids = macer_result.get("retrieved_source_ids", [])
        self.last_evidence_trace = macer_result.get("evidence_trace", [])
        self.last_seen_source_ids = macer_result.get("seen_source_ids", retrieved_source_ids)
        self.last_presented_source_ids = macer_result.get("presented_source_ids", retrieved_source_ids)
        self.last_partial_source_ids = macer_result.get("partial_source_ids", [])
        user_prompt = macer_result.get("prompt","empty")

        prediction = clean_prediction(prediction)
        
        # Format as JSON for compatibility
        response = json.dumps({"answer": prediction})
        
        # Return with MACER metadata including retrieved_source_ids
        return (
            response,
            f"Category {category}",
            context,
            history_str,
            retrieved_source_ids,
            user_prompt
        )

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def evaluate_dataset(dataset_path: str, model: str = None, output_path: Optional[str] = None, \
    ratio: float = 1.0, backend: str = None, temperature_c5: float = 0.5, \
    retrieve_k: int = 10, mode: str = "key_graph_v3", graph_backend: str = "inmemory", \
    llm_config_path: str = "configs/llm.yaml", api_key: str = None, \
    base_url: str = None, name: str = "default", sort_by: str = 'score', \
    prompt_config_path: str = None, use_format_instruction: bool = False, \
    retrieval_version: str = 'paper', evidence_pool_version: str = 'v2', \
    sub_query_version: str = 'v2', macer_version: str = 'v1',
    responder_llm_config: dict = None, render_top_fraction: float = 1.0,
    categories: List[int] = None, enable_semantic_gating: bool = True,
    topic_retrieval_version: str = 'v1', ablation: str = 'none',
    write_ablation: str = 'none', cache_dir: str = None,
    max_steps: int = 3, log_dir: str = None, metrics_mode: str = 'core_historical_v1',
    seed: Optional[int] = None, final_top_k: Optional[int] = 60,
    sample_indices: Optional[List[int]] = None,
    context_char_limit: int = 90000, rrf_topic_weight: float = 1.0,
    rrf_keyword_weight: float = 1.0, neighbor_score_scale: float = 1.0,
    keyword_rrf_tie_policy: str = 'ordinal',
    round_top_k: Optional[int] = None,
    **selection_options):
    """Evaluate the agent on the LoComo dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
    """
    if mode != "key_graph_v3":
        raise ValueError("The v1 evaluator supports only key_graph_v3")
    if graph_backend != "inmemory":
        raise ValueError("The v1 evaluator supports only the inmemory graph backend")
    if final_top_k is not None and (type(final_top_k) is not int or final_top_k <= 0):
        raise ValueError("final_top_k must be positive or None")
    if type(context_char_limit) is not int or context_char_limit <= 0:
        raise ValueError("context_char_limit must be a positive integer")
    if set(selection_options) - {'evidence_selection_version'}:
        raise ValueError('Unsupported evaluator selection option')
    evidence_selection_version = selection_options.get('evidence_selection_version', 'legacy_rrf')
    if retrieval_version == 'paper' and retrieve_k != 10:
        raise ValueError('retrieve_k only controls legacy retrieval; use round_top_k for paper retrieval')
    if round_top_k is not None:
        if type(round_top_k) is not int or round_top_k <= 0:
            raise ValueError('round_top_k must be a positive integer or None')
        if retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf':
            raise ValueError('round_top_k requires paper retrieval and legacy_rrf evidence selection')
    evidence_budget = {
        'retrieval_version': retrieval_version,
        'evidence_selection_version': evidence_selection_version,
        'round_top_k': (round_top_k if round_top_k is not None else final_top_k)
                       if retrieval_version == 'paper' and evidence_selection_version == 'legacy_rrf'
                       else None,
        'legacy_retrieve_k': retrieve_k if retrieval_version != 'paper' else None,
        'final_top_k': final_top_k,
        'context_char_limit': context_char_limit,
    }
    if evidence_selection_version not in ('legacy_rrf', 'query_note_cosine_v1'):
        raise ValueError("Unsupported evidence selection version")
    if evidence_selection_version == 'query_note_cosine_v1' and (
            retrieval_version != 'paper' or evidence_pool_version != 'v2'
            or final_top_k is None or render_top_fraction != 1.0):
        raise ValueError("Cosine selection requires paper retrieval, v2 pool, bounded K and full rendering")
    for key, value in (('rrf_topic_weight', rrf_topic_weight),
                       ('rrf_keyword_weight', rrf_keyword_weight)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if (type(neighbor_score_scale) not in (int, float)
            or not math.isfinite(neighbor_score_scale)
            or not 0 <= neighbor_score_scale <= 1):
        raise ValueError('neighbor_score_scale must be finite in [0, 1]')
    weighted = (rrf_topic_weight != 1.0 or rrf_keyword_weight != 1.0
                or neighbor_score_scale != 1.0)
    if weighted and (retrieval_version != 'paper'
                     or evidence_selection_version != 'legacy_rrf'):
        raise ValueError('Retrieval weights require paper retrieval and legacy_rrf evidence selection')
    if keyword_rrf_tie_policy not in ('ordinal', 'mean_reciprocal'):
        raise ValueError('Unsupported keyword RRF tie policy')
    if keyword_rrf_tie_policy != 'ordinal' and (
            retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf'):
        raise ValueError('Keyword RRF tie policy requires paper retrieval and legacy_rrf evidence selection')
    if metrics_mode not in ('core_historical_v1', 'core_normalized_v1'):
        raise ValueError(f"Unsupported metrics_mode: {metrics_mode}")
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
    # Load LLM config to get model and backend for log filename
    llm_config = load_llm_config(llm_config_path)
    effective_backend = backend or llm_config.get("backend", "openai")
    effective_model = model or llm_config.get("model", "Qwen/Qwen3-8B")
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"{effective_model}_{effective_backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(log_dir or "logs", mode, name, log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    logger.info(f"graph_backend: {graph_backend}")
    logger.info(f"enable_semantic_gating: {enable_semantic_gating}")
    logger.info(f"topic_retrieval_version: {topic_retrieval_version}")
    if weighted:
        logger.info("retrieval_weights: topic=%s keyword=%s neighbor_scale=%s",
                    rrf_topic_weight, rrf_keyword_weight, neighbor_score_scale)
    if keyword_rrf_tie_policy != 'ordinal':
        logger.info('keyword_rrf_tie_policy: %s', keyword_rrf_tie_policy)
    
    # Construct ablation_flags dictionary from ablation string (Read-Side Ablation)
    ablation_flags = {
        'no_topic': ablation == 'no_topic',
        'no_neighbor': ablation == 'no_neighbor',
        'no_loop': ablation == 'no_loop',
        'no_kws': ablation == 'no_kws'
    }
    logger.info(f"Read ablation mode: {ablation}")
    if ablation != 'none':
        logger.info(f"Read ablation flags: {ablation_flags}")
    
    # Construct write_ablation_flags dictionary from write_ablation string (Write-Side Ablation)
    write_ablation_flags = {
        'no_consolidation': write_ablation == 'no_consolidation',  # W1: Skip _evolve_memory entirely
        'no_merge': write_ablation == 'no_merge',                  # W2: Skip neighbor_updates
        'no_link_write': write_ablation == 'no_link_write',        # W3: Skip new_links
        'no_denoising': write_ablation == 'no_denoising'           # W4: Use raw content as context
    }
    logger.info(f"Write ablation mode: {write_ablation}")
    if write_ablation != 'none':
        logger.info(f"Write ablation flags: {write_ablation_flags}")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    if sample_indices is not None:
        if (not isinstance(sample_indices, list) or not sample_indices or
                any(type(index) is not int or index < 0 or index >= len(samples)
                    for index in sample_indices) or len(set(sample_indices)) != len(sample_indices)):
            raise ValueError("sample_indices must be distinct indices in the selected dataset")
        selected_samples = [(index, samples[index]) for index in sample_indices]
    else:
        selected_samples = list(enumerate(samples))
    
    # Store results
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    # 使用用户指定的类别，如果未指定则使用全部类别
    allow_categories = categories if categories else [1, 2, 3, 4, 5]
    logger.info(f"Evaluating categories: {allow_categories}")
    
    # Setup results directory and progress tracking
    results_dir = os.path.join(output_path or "results", name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 立即保存prompt配置到results目录（防止测试过程中修改配置导致无法追溯）
    prompt_loader = PromptLoader(prompt_config_path)
    prompt_config_summary = prompt_loader.get_config_summary()
    if prompt_config_path and os.path.exists(prompt_config_path):
        import shutil
        config_backup_path = os.path.join(results_dir, 'prompt_config_used.yaml')
        shutil.copy(prompt_config_path, config_backup_path)
        logger.info(f"Prompt config saved to {config_backup_path}")
        logger.info(f"Prompt config: {prompt_config_summary}")
    
    # Load LLM config to get model and backend for progress file naming
    llm_config = load_llm_config(llm_config_path)
    effective_backend = backend or llm_config.get("backend", "openai")
    effective_model = model or llm_config.get("model", "Qwen/Qwen3-8B")
    # Create progress file to track completed samples
    progress_file = os.path.join(results_dir, f"progress_key_advanced.json")
    completed_samples = set()
    sample_results_cache = {}
    expected_total_questions = sum(
        int(qa.category) in allow_categories for _, sample in selected_samples for qa in sample.qa)
    # The progress file is only a hint. Check the actual QA records even when an
    # old progress file says a sample is complete, and recover partial records.
    for sample_idx, sample in selected_samples:
        if seed is not None:
            sample_seed = seed + sample_idx * 100000
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2 ** 32))
        sample_result_file = os.path.join(results_dir, f"sample_{sample_idx}_{name}_results.json")
        if not os.path.exists(sample_result_file):
            sample_result_file = os.path.join(results_dir, f"sample_{sample_idx}_results.json")
        if not os.path.exists(sample_result_file):
            continue
        try:
            with open(sample_result_file, 'r', encoding='utf-8') as stream:
                saved = json.load(stream)
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable sample checkpoint {sample_idx}: {e}")
            continue
        # Do not overwrite or combine old predictions after read-side changes,
        # including when this evaluator is called without the CLI runner.
        # The CLI runner additionally compares the complete resolved run_config.
        if isinstance(saved, dict) and (
                saved.get('execution_revision') != EXECUTION_REVISION
                or saved.get('evidence_budget') != evidence_budget):
            raise ValueError('Checkpoint execution revision or evidence budget differs; use a new run name')
        try:
            checkpoint = _validated_sample_checkpoint(saved, sample_idx, sample, allow_categories)
            sample_results_cache[sample_idx] = checkpoint
            expected_count = sum(int(qa.category) in allow_categories for qa in sample.qa)
            if (len(checkpoint['qa_results']) == expected_count and not checkpoint['errors']
                    and not any(entry.get('prediction_parse_error')
                                for entry in checkpoint['qa_results'].values())):
                completed_samples.add(sample_idx)
            logger.info(f"Recovered {len(checkpoint['qa_results'])}/{expected_count} QAs for sample {sample_idx}")
        except Exception as e:
            logger.warning(f"Ignoring invalid sample checkpoint {sample_idx}: {e}")
    for checkpoint in sample_results_cache.values():
        for entry in checkpoint['qa_results'].values():
            results.append(entry)
            all_metrics.append(entry['metrics'])
            all_categories.append(entry['category'])
            category_counts[entry['category']] += 1
            total_questions += 1
    
    # Evaluate each sample
    i = 0
    error_num = sum(bool(entry.get('prediction_parse_error')) for entry in results)
    active_errors_by_sample = {index: list(checkpoint['errors'])
                               for index, checkpoint in sample_results_cache.items()}
    # Load LLM config to get model and backend for directory naming
    llm_config = load_llm_config(llm_config_path)
    effective_backend = backend or llm_config.get("backend", "openai")
    effective_model = model or llm_config.get("model", "Qwen/Qwen3-8B")
    # 使用自定义 cache_dir 或默认路径
    if cache_dir:
        memories_dir = os.path.join(cache_dir, "cached_mem_{}".format(mode), name)
    else:
        memories_dir = os.path.join('cached', "cached_mem_{}".format(mode), name)
    os.makedirs(memories_dir, exist_ok=True)
    logger.info(f"Cache directory: {memories_dir}")
    
    # A graph snapshot is the only reusable construction cache. Legacy Neo4j
    # exports and pickle files are deliberately incompatible with this path.
    if (list(Path(memories_dir).glob("neo4j*.json")) or
            list(Path(memories_dir).glob("*.pkl"))):
        raise RuntimeError("Legacy Neo4j/pickle cache cannot be mixed with in-memory graphs")

    for sample_idx, sample in selected_samples:
        if sample_idx in completed_samples:
            logger.info(f"Sample {sample_idx} already completed. Skipping.")
            continue
        if seed is not None:
            sample_seed = seed + sample_idx * 100000
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2 ** 32))
        graph_cache_file = os.path.join(memories_dir, f"graph_sample_{sample_idx}.json")
        agent = advancedMemAgent(
            model=model, backend=backend, retrieve_k=retrieve_k,
            temperature_c5=temperature_c5, mode=mode,
            graph_backend=graph_backend, llm_config_path=llm_config_path,
            api_key=api_key, base_url=base_url, logger=logger, sort_by=sort_by,
            prompt_config_path=prompt_config_path,
            use_format_instruction=use_format_instruction,
            retrieval_version=retrieval_version,
            evidence_pool_version=evidence_pool_version,
            sub_query_version=sub_query_version, macer_version=macer_version,
            responder_llm_config=responder_llm_config,
            render_top_fraction=render_top_fraction,
            enable_semantic_gating=enable_semantic_gating,
            topic_retrieval_version=topic_retrieval_version,
            max_steps=max_steps, final_top_k=final_top_k,
            round_top_k=round_top_k,
            context_char_limit=context_char_limit,
            evidence_selection_version=evidence_selection_version,
            rrf_topic_weight=rrf_topic_weight,
            rrf_keyword_weight=rrf_keyword_weight,
            neighbor_score_scale=neighbor_score_scale,
            keyword_rrf_tie_policy=keyword_rrf_tie_policy,
            seed=(seed + sample_idx * 100000) if seed is not None else None,
        )

        enable_tracking()
        construction_tracker = get_construction_tracker()
        retrieval_tracker = get_retrieval_tracker()
        construction_tracker.reset()
        retrieval_tracker.reset()
        agent.memory_system.llm_controller.llm._token_tracker = construction_tracker
        agent.memory_system._pair_log_path = os.path.join(results_dir, "pair_log.jsonl")

        if os.path.exists(graph_cache_file):
            # Loading and fresh construction both use the same memory-system
            # interface; a malformed snapshot is an error, never a silent rebuild.
            agent.memory_system.load_graph(graph_cache_file)
            logger.info(f"Loaded in-memory graph for sample {sample_idx}")
        else:
            logger.info(f"Constructing in-memory graph for sample {sample_idx}")
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + ": " + turn.text
                    agent.add_memory(
                        conversation_tmp, time=turns.date_time,
                        source_id=turn.dia_id,
                        write_ablation_flags=write_ablation_flags,
                    )
            if hasattr(agent.memory_system, 'build_topic_layer'):
                logger.info("Batch loading complete. Building Topic Layer manually...")
                agent.memory_system.build_topic_layer(
                    min_keywords_per_topic=3,
                    max_keywords_for_summary=10,
                )
            # A failed save must abort before QA, or resume could reuse a
            # partially constructed graph with a misleading checkpoint.
            agent.memory_system.save_graph(graph_cache_file)
            logger.info(f"Saved in-memory graph for sample {sample_idx}")

        # E1: Save construction cost stats for this sample
        construction_stats = construction_tracker.get_summary()
        if construction_stats["call_count"] > 0:
            logger.info(f"[E1] Construction cost for sample {sample_idx}: "
                        f"tokens={construction_stats['total_tokens']}, "
                        f"calls={construction_stats['call_count']}, "
                        f"time={construction_stats['total_wall_time_s']:.1f}s")

        # Switch tracker: from now on, LLM calls go to retrieval tracker
        if mode == "key_graph_v3" and hasattr(agent, 'memory_system'):
            agent.memory_system.llm_controller.llm._token_tracker = retrieval_tracker
        retrieval_tracker.reset()

        logger.info(f"\nProcessing sample {sample_idx + 1}/{len(samples)}")

        # Keep the original dataset index as the durable QA identity.
        recovered = sample_results_cache.get(sample_idx, {})
        sample_records = dict(recovered.get('qa_results', {}))
        sample_errors = list(recovered.get('errors', []))
        sample_error_history = list(recovered.get('error_history', []))
        cost_attempts = [*recovered.get('cost_attempts', []),
                         {'construction_cost': construction_stats,
                          'retrieval_cost': retrieval_tracker.get_summary()}]
        expected_qa = sum(int(qa.category) in allow_categories for qa in sample.qa)
        sample_result_file = os.path.join(results_dir, f"sample_{sample_idx}_{name}_results.json")
        failed_qa_index = None
        
        for qa_index, qa in enumerate(sample.qa):
            if int(qa.category) not in allow_categories or qa_index in sample_records:
                continue
            # Query-intent diagnostics belong to this QA only. An answer may
            # fail before intent analysis and must not inherit a prior trace.
            if hasattr(agent, 'memory_system') and hasattr(agent.memory_system, 'last_query_intent_trace'):
                agent.memory_system.last_query_intent_trace = None
            if seed is not None:
                random.seed(seed + sample_idx * 100000 + qa_index)
            try:
                    
                # Generate prediction
                answer_result = agent.answer_question(qa.question, qa.category, qa.final_answer, ablation_flags)
                    
                prediction, _, context, history_str, retrieved_source_ids, user_prompt = answer_result
                evidence_trace = getattr(agent, 'last_evidence_trace', [])
                seen_source_ids = getattr(agent, 'last_seen_source_ids', retrieved_source_ids)
                presented_source_ids = getattr(agent, 'last_presented_source_ids', retrieved_source_ids)
                partial_source_ids = getattr(agent, 'last_partial_source_ids', [])
                    
                prediction_parse_error = False
                try:
                    prediction = json.loads(prediction)["answer"]
                except:
                    prediction = prediction
                    logger.info(f"Failed to parse prediction as JSON: {prediction}")
                    prediction_parse_error = True
                    
                # Log results
                logger.info(f"\nQuestion {total_questions + 1}: {qa.question}")
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Reference: {qa.final_answer}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Final_history_str: {history_str}")
                logger.info(f"Final_context: {context}")
                logger.info(f"user_prompt: {user_prompt}")

                # Calculate answer metrics
                f1_version = ('historical_v1' if metrics_mode == 'core_historical_v1'
                              else 'normalized_v1')
                metrics = calculate_core_metrics(prediction, qa.final_answer, f1_version=f1_version)
                    
                # Calculate retrieval metrics (Recall/Precision/Hit Rate)
                gold_evidence = qa.evidence if hasattr(qa, 'evidence') and qa.evidence else []
                retrieval_metrics = calculate_retrieval_metrics(gold_evidence, retrieved_source_ids)
                seen_retrieval_metrics = calculate_retrieval_metrics(
                    gold_evidence, seen_source_ids)
                presented_retrieval_metrics = calculate_retrieval_metrics(
                    gold_evidence, presented_source_ids)
                    
                # Merge retrieval metrics into metrics dict
                metrics["retrieval_recall"] = retrieval_metrics["recall"]
                metrics["retrieval_precision"] = retrieval_metrics["precision"]
                metrics["retrieval_hit_rate"] = retrieval_metrics["hit_rate"]
                metrics["seen_retrieval_recall"] = seen_retrieval_metrics["recall"]
                metrics["seen_retrieval_precision"] = seen_retrieval_metrics["precision"]
                metrics["seen_retrieval_hit_rate"] = seen_retrieval_metrics["hit_rate"]
                metrics["presented_retrieval_recall"] = presented_retrieval_metrics["recall"]
                metrics["presented_retrieval_precision"] = presented_retrieval_metrics["precision"]
                metrics["presented_retrieval_hit_rate"] = presented_retrieval_metrics["hit_rate"]
                    
                # Log retrieval metrics
                logger.info(f"Evidence (Gold): {gold_evidence}")
                logger.info(f"Retrieved Source IDs: {retrieved_source_ids}")
                logger.info(f"Retrieval Metrics: Recall={retrieval_metrics['recall']:.4f}, "
                           f"Precision={retrieval_metrics['precision']:.4f}, "
                           f"Hit Rate={retrieval_metrics['hit_rate']:.0f} "
                           f"(Intersection: {retrieval_metrics['intersection_count']}/{retrieval_metrics['gold_count']})")
                    
                # Store individual result
                result = {
                    "sample_id": sample_idx,
                    "qa_index": qa_index,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                    "evidence": gold_evidence,
                    "retrieved_source_ids": retrieved_source_ids,
                    "seen_source_ids": seen_source_ids,
                    "presented_source_ids": presented_source_ids,
                    "partial_source_ids": partial_source_ids,
                    "evidence_trace": evidence_trace,
                    "retrieval_metrics": retrieval_metrics,
                    "seen_retrieval_metrics": seen_retrieval_metrics,
                    "presented_retrieval_metrics": presented_retrieval_metrics,
                    "user_prompt": user_prompt,
                    "final_context": context,
                    "final_history_str": history_str,
                    "prediction_parse_error": prediction_parse_error,
                }
                    
                # The sample checkpoint is the authoritative resume source.
                candidate_records = {**sample_records, qa_index: result}
                candidate_errors = [error for error in sample_errors
                                    if error['qa_index'] != qa_index]
                cost_attempts[-1]['retrieval_cost'] = retrieval_tracker.get_summary()
                _save_sample_checkpoint(sample_result_file, sample_idx, expected_qa,
                                        candidate_records, candidate_errors, cost_attempts,
                                        error_history=sample_error_history,
                                        evidence_budget=evidence_budget)
                sample_records = candidate_records
                sample_errors = candidate_errors
                error_num += int(prediction_parse_error)
                results.append(result)
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                category_counts[qa.category] += 1
                total_questions += 1

                # Retain the historical per-QA artifact as a convenience.
                sample_dir = os.path.join(results_dir, f'sample{sample_idx}_{name}')
                try:
                    _write_json_atomic(os.path.join(sample_dir, f'qa{total_questions}.json'), result)
                except Exception as artifact_error:
                    logger.warning(f"Could not save individual QA artifact: {artifact_error}")
                    
                # Log progress
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")
            except Exception as e:
                failed_qa_index = qa_index
                failure = {'qa_index': qa_index, 'error_type': type(e).__name__,
                           'at': datetime.now().isoformat()}
                intent_trace = getattr(getattr(agent, 'memory_system', None),
                                       'last_query_intent_trace', None)
                if (isinstance(intent_trace, list) and intent_trace
                        and isinstance(intent_trace[-1], dict)
                        and intent_trace[-1].get('retry_reason')):
                    diagnostic_fields = ('attempt', 'caller', 'format', 'output_characters',
                                         'finish_reason', 'retry_reason')
                    failure['query_intent_attempts'] = [
                        {key: attempt.get(key) for key in diagnostic_fields}
                        for attempt in intent_trace if isinstance(attempt, dict)]
                sample_errors = [error for error in sample_errors
                                 if error['qa_index'] != qa_index] + [failure]
                sample_error_history.append(failure)
                logger.error(f"Error processing sample {sample_idx} QA {qa_index}: {e}")
                logger.exception(e)
                cost_attempts[-1]['retrieval_cost'] = retrieval_tracker.get_summary()
                _save_sample_checkpoint(sample_result_file, sample_idx, expected_qa,
                                        sample_records, sample_errors, cost_attempts,
                                        failed_qa_index, sample_error_history, evidence_budget)
                break

        # E1: Collect retrieval cost stats for this sample
        retrieval_stats = retrieval_tracker.get_summary()
        cost_attempts[-1]['retrieval_cost'] = retrieval_stats
        num_qa = len(sample_records) - len(recovered.get('qa_results', {}))
        if retrieval_stats["call_count"] > 0:
            logger.info(f"[E1] Retrieval cost for sample {sample_idx}: "
                        f"tokens={retrieval_stats['total_tokens']}, "
                        f"calls={retrieval_stats['call_count']}, "
                        f"time={retrieval_stats['total_wall_time_s']:.1f}s, "
                        f"queries={num_qa}")
            if num_qa > 0:
                logger.info(f"[E1] Per-query avg: "
                            f"tokens={retrieval_stats['total_tokens']/num_qa:.0f}, "
                            f"time={retrieval_stats['total_wall_time_s']/num_qa:.2f}s")

        try:
            _save_sample_checkpoint(sample_result_file, sample_idx, expected_qa,
                                    sample_records, sample_errors, cost_attempts,
                                    failed_qa_index, sample_error_history, evidence_budget)
            active_errors_by_sample[sample_idx] = sample_errors
            if (len(sample_records) == expected_qa and not sample_errors
                    and failed_qa_index is None and not any(
                        entry.get('prediction_parse_error')
                        for entry in sample_records.values())):
                completed_samples.add(sample_idx)
            else:
                completed_samples.discard(sample_idx)
            progress_data = {
                "model": effective_model, "backend": effective_backend, "ratio": ratio,
                "completed_samples": sorted(completed_samples),
                "last_updated": datetime.now().isoformat(),
            }
            _write_json_atomic(progress_file, progress_data)
            sample_qa_results = [sample_records[index] for index in sorted(sample_records)]
            if sample_qa_results:
                overall_avg = compute_metric_averages(sample_qa_results)
                by_category = {}
                for cat in {item['category'] for item in sample_qa_results}:
                    by_category[int(cat)] = compute_metric_averages(
                        [item for item in sample_qa_results if item['category'] == cat])
                average_metrics_data = {
                    "sample_id": sample_idx, "total_qa": len(sample_qa_results),
                    "expected_qa": expected_qa,
                    "complete": sample_idx in completed_samples,
                    "overall_metrics_average": overall_avg,
                    "metrics_average_by_category": by_category,
                }
                _write_json_atomic(os.path.join(results_dir, f"sample_{sample_idx}_average_metrics.json"),
                                   average_metrics_data)
        except Exception as e:
            logger.error(f"Error saving sample {sample_idx} checkpoint: {e}")
            raise
        
    execution_error_count = sum(len(errors) for errors in active_errors_by_sample.values())
    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    # Prepare final results
    # Get effective model and backend for final results
    llm_config = load_llm_config(llm_config_path)
    effective_model = model or llm_config.get("model", "Qwen/Qwen3-8B")
    effective_backend = backend or llm_config.get("backend", "openai")
    final_results = {
        "model": effective_model,
        "backend": effective_backend,
        "dataset": dataset_path,
        "metrics_mode": metrics_mode,
        "execution_revision": EXECUTION_REVISION,
        "seed": seed,
        "graph_backend": graph_backend,
        "retrieval_version": retrieval_version,
        "final_top_k": final_top_k,
        "evidence_budget": evidence_budget,
        **({"round_top_k": round_top_k} if round_top_k is not None else {}),
        "context_char_limit": context_char_limit,
        **({"rrf_topic_weight": rrf_topic_weight,
            "rrf_keyword_weight": rrf_keyword_weight,
            "neighbor_score_scale": neighbor_score_scale} if weighted else {}),
        **({"evidence_selection_version": evidence_selection_version}
           if evidence_selection_version != 'legacy_rrf' else {}),
        **({"keyword_rrf_tie_policy": keyword_rrf_tie_policy}
           if keyword_rrf_tie_policy != 'ordinal' else {}),
        "total_questions": total_questions,
        "expected_questions": expected_total_questions,
        "complete": (len(completed_samples) == len(selected_samples)
                     and total_questions == expected_total_questions
                     and execution_error_count == 0 and error_num == 0),
        "completed_samples": sorted(completed_samples),
        "prediction_parse_errors": error_num,
        "execution_errors": execution_error_count,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "prompt_config": prompt_config_summary,
        "prompt_config_path": prompt_config_path
        # "individual_results": results
    }
    logger.info(f"Error number: {error_num}")
    # Save results
    _write_json_atomic(os.path.join(results_dir, 'final_results.json'), final_results)
    logger.info(f"Results saved to {os.path.join(results_dir, 'final_results.json')}")
    
    # Save metrics summary CSV
    import csv
    csv_path = os.path.join(results_dir, 'metrics_summary.csv')
    csv_rows = []
    
    # 每个category的f1和bleu1
    for cat in ['1', '2', '3', '4', '5']:
        cat_key = f'category_{cat}'
        if cat_key in aggregate_results:
            f1 = round(aggregate_results[cat_key]['f1']['mean'], 4)
            bleu1 = round(aggregate_results[cat_key]['bleu1']['mean'], 4)
            csv_rows.append([f'cat{cat}', f1, bleu1, '', ''])
    
    # overall的f1, bleu1, retrieval_recall, retrieval_hit_rate
    if 'overall' in aggregate_results:
        overall = aggregate_results['overall']
        f1 = round(overall['f1']['mean'], 4)
        bleu1 = round(overall['bleu1']['mean'], 4)
        ret_recall = round(overall['retrieval_recall']['mean'], 4)
        ret_hit = round(overall['retrieval_hit_rate']['mean'], 4)
        csv_rows.append(['avg', f1, bleu1, ret_recall, ret_hit])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'f1', 'bleu1', 'retrieval_recall', 'retrieval_hit_rate'])
        writer.writerows(csv_rows)
    logger.info(f"Metrics summary saved to {csv_path}")
    
    # Log summary
    logger.info("\nEvaluation Summary:")
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")
    
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")
    
    return final_results
