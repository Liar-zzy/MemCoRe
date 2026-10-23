from typing import List, Dict, Optional, Set, Tuple
import networkx as nx
import itertools
import json
import os
from pathlib import Path
import tempfile
from copy import deepcopy
import math
import numpy as np
import yaml
import uuid
import logging
from collections import defaultdict
from memcore.versions.v1.agent.memory_layer import (
    AgenticMemorySystem, 
    MemoryNote, 
    LLMController,
    EmbeddingController,
    get_json_completion_with_retry,
)
from datetime import datetime
from memcore.versions.v1.prompts.topic_prompts import topic_description_v0
from memcore.versions.v1.prompts.prompt_loader import PromptLoader
from memcore.versions.v1.prompts.judgment_prompts import judgment_prompts_v2
from memcore.versions.v1.utils.rrf import reciprocal_rank_fusion
from memcore.versions.v1.utils.topic_partition import bounded_keyword_partition
from memcore.versions.v1.storage.graph_store import InMemoryGraphStore

import warnings
warnings.filterwarnings("ignore")

from graspologic.partition import hierarchical_leiden

def load_llm_config(config_path: str = "configs/llm.yaml") -> Dict:
    """加载 LLM 和 Embedding 配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default values.")
        return {}

class KeyGraphMemoryV3(AgenticMemorySystem):
    """
    基于 Amem 语义和 Keyword 话题的混合内存系统。
    
    结合两种图结构：
    1. Semantic Graph: (:MemoryNote)-[:RELATED_TO]->(:MemoryNote) - 基于 Amem 的语义关联
    2. Keyword Topic Graph: 
       - (:MemoryNote)-[:HAS_KEYWORD]->(:Keyword)
       - (:Keyword)-[:CO_OCCURS]->(:Keyword)
    """
    MEMORY_REVISION = 'paper-context-centroid-bounded-topic-v2'
    
    def __init__(self, 
                 llm_config_path: str = "configs/llm.yaml",
                 neo4j_config_path: str = "neo4j_config.yaml",
                 use_neo4j: bool = False,
                 evo_threshold: int = 100,
                 topic_maintenance_threshold: int = 5,
                 logger: Optional[logging.Logger] = None,
                 prompt_config_path: str = None,
                 enable_semantic_gating: bool = True,
                 topic_retrieval_version: str = 'v1',
                 graph_backend: str = 'inmemory',
                 seed: Optional[int] = None
                 ):
        """
        初始化 KeyGraphMemoryV3。
        
        Args:
            llm_config_path: LLM 和 Embedding 配置文件路径
            neo4j_config_path: Legacy parameter, ignored by the in-memory backend
            use_neo4j: Legacy parameter; True is rejected.
            evo_threshold: 记忆演化阈值
            topic_maintenance_threshold: Topic 层自动维护阈值（每添加 N 条 Note 触发一次更新，默认 5）
            logger: 可选的 logger 实例，如果不提供则使用默认 logger
            enable_semantic_gating: 是否启用 MemDyn 语义门控（默认 True），False 则使用 Baseline 模式
            topic_retrieval_version: Topic 检索版本 ('v1' 或 'v2')
                - 'v1': 基础向量相似度检索 (_retrieve_via_topic_vector)
                - 'v2': 三角导航策略，结合向量相似度和关键词重叠 (_retrieve_via_topic_vector_v2)
        """
        # 初始化 logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # 加载 LLM 配置
        config = load_llm_config(llm_config_path)
        llm_config = config.get('llm', {})
        embedding_config = config.get('embedding', {})
        
        if use_neo4j:
            raise ValueError("KeyGraphMemoryV3 no longer supports use_neo4j=True")
        if graph_backend != 'inmemory':
            raise ValueError("graph_backend must be 'inmemory'")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer or None')
        # This version never uses the parent's SentenceTransformer retriever.
        # Avoid constructing or downloading one just to create a graph memory.
        self.memories = {}
        self.retriever = None
        self.evo_threshold = evo_threshold
        self.llm_controller = LLMController(
            llm_config.get('backend', 'openai'), llm_config.get('model', 'gpt-4o-mini'),
            llm_config.get('api_key'), llm_config.get('base_url'),
            llm_config.get('extra_body'))
        
        # 初始化 Embedding Controller
        self.embedding_controller = EmbeddingController(
            backend=embedding_config.get('backend', 'openai'),
            model=embedding_config.get('model', 'text-embedding-3-small'),
            api_key=embedding_config.get('api_key'),
            base_url=embedding_config.get('base_url')
        )
        self.logger.info(f"Initialized EmbeddingController with model: {self.embedding_controller.model}")
        
        self.store = InMemoryGraphStore()
        self.graph_backend = graph_backend
        self.use_neo4j = False
        self.last_retrieval_trace = None
        self.last_query_intent_trace = None
        self.last_topic_clustering_trace = None
        self.graph_seed = seed
        
        # Topic 层自动维护机制
        self.maintenance_threshold = topic_maintenance_threshold
        self.maintenance_counter = 0
        self.logger.info(f"Topic maintenance: enabled (threshold: {self.maintenance_threshold} notes)")
        
        # 初始化Prompt加载器
        self.prompt_loader = PromptLoader(prompt_config_path)
        
        # MemDyn 语义门控开关
        self.enable_semantic_gating = enable_semantic_gating
        self.logger.info(f"MemDyn Semantic Gating: {'ENABLED' if enable_semantic_gating else 'DISABLED (Baseline Mode)'}")

        # E4: Pair logging for JSD correlation analysis
        self._pair_log_path = None
        
        # Topic 检索版本配置
        self.topic_retrieval_version = topic_retrieval_version
        self.logger.info(f"Topic Retrieval Version: {topic_retrieval_version}")
        
    


    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的 embedding 向量。
        这是一个便捷方法，直接调用 embedding_controller。
        
        Args:
            text: 输入文本
            
        Returns:
            embedding 向量
        """
        return self.embedding_controller.get_embedding(text)
    
    def add_note(self, content: str, time: str = None, use_v2: bool = True, source_id: str = None, 
                 write_ablation_flags: Dict[str, bool] = None, **kwargs) -> str:
        """
        添加新的记忆笔记，集成记忆演化和稀疏连接。
        
        实现逻辑：
        1. 手动创建 MemoryNote（使用 LLM 生成 metadata）
        2. 计算 Note 和 Keywords 的 Embeddings
        3. 记忆演化：向量召回 + LLM 综合判断（更新邻居 + 建立连接）
        4. 写入图存储（节点 + 关系 + embeddings + 邻居更新）
        
        Args:
            content: 记忆内容
            time: 时间戳 (可选)
            use_v2: 是否使用 v2 版本的 LLM prompt（更严格的 metadata 生成）
            source_id: 外部来源 ID（如数据集中的 dia_id），用于评估时的对应 (可选)
            write_ablation_flags: 写入侧消融标志 (可选)
                - 'no_consolidation' (W1): 跳过 _evolve_memory（无 Merging/Linking）
                - 'no_merge' (W2): 跳过 neighbor_updates（无 Context 合并）
                - 'no_link_write' (W3): 跳过 new_links（无物理连接）
                - 'no_denoising' (W4): 用原始 content 作为 context（无 LLM 去噪）
            **kwargs: 其他参数
            
        Returns:
            note_id: 新添加的记忆 ID
        """
        import uuid
        from datetime import datetime
        
        # 处理消融标志
        if write_ablation_flags is None:
            write_ablation_flags = {}
        
        # Step 1: source_id is the durable construction identity when supplied.
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError('source_id must be a nonempty string when provided')
        note_id = 'note:' + source_id if source_id is not None else str(uuid.uuid4())
        if note_id in self.memories or self.store.get_node(note_id) is not None:
            raise ValueError(f'duplicate memory source_id: {source_id}')
        current_time = datetime.now().strftime("%I:%M %p on %d %B, %Y")
        timestamp = time or current_time
        
        self.logger.info(f"[KeyGraphMemoryV3] Creating note {note_id[:8]}...")
        
        # 记录消融模式
        if any(write_ablation_flags.values()):
            active_flags = [k for k, v in write_ablation_flags.items() if v]
            self.logger.info(f"[KeyGraphMemoryV3] Write ablation active: {active_flags}")
        
        # Step 1.1: 使用 LLM 生成 metadata
        self.llm_controller.llm._current_caller = "analyze_content"
        metadata = self._generate_metadata_with_llm(content, use_v2=use_v2)
        
        # W4 (no_denoising): 覆盖 context 为原始 content
        if write_ablation_flags.get('no_denoising'):
            self.logger.info(f"[Ablation W4] Skipping LLM denoising, using raw content as context")
            metadata['context'] = content
        
        # Step 1.2: 创建 MemoryNote 对象（不触发父类的 process_memory）
        note = MemoryNote(
            content=content,
            id=note_id,
            keywords=metadata.get("keywords", []),
            links=[],  # 暂时为空，稍后通过演化填充
            importance_score=1.0,
            retrieval_count=0,
            timestamp=timestamp,
            last_accessed=current_time,
            context=metadata.get("context", "General"),
            evolution_history=[],
            category="Uncategorized",
            llm_controller=None  # 不需要，因为我们已经生成了 metadata
        )
        
        # 添加 source_id 属性（用于评估时的外部 ID 对应）
        note.source_id = source_id
        
        self.logger.info(f"✓ Note created:")
        self.logger.info(f"- Keywords: {note.keywords}")
        self.logger.info(f"- Content: {note.content}")
        self.logger.info(f"- Context: {note.context}")
        
        # Step 2: 计算 Embeddings
        embeddings_dict = self._compute_embeddings_for_note(note)
        
        # Step 3: 记忆演化（向量召回 + LLM 综合判断）
        # 只有在图中已有节点时才尝试演化
        evolution_result = None
        
        # W1 (no_consolidation): 完全跳过演化
        if write_ablation_flags.get('no_consolidation'):
            self.logger.info(f"[Ablation W1] Skipping _evolve_memory entirely (no consolidation)")
            evolution_result = None
        elif self.memories:
            evolution_result = self._evolve_memory(note, embeddings_dict['note'])
            
            # 应用演化结果
            if evolution_result:
                # W2 (no_merge): 清空 neighbor_updates
                if write_ablation_flags.get('no_merge'):
                    self.logger.info(f"[Ablation W2] Clearing neighbor_updates (no merge)")
                    evolution_result['neighbor_updates'] = []
                
                # W3 (no_link_write): 清空 new_links
                if write_ablation_flags.get('no_link_write'):
                    self.logger.info(f"[Ablation W3] Clearing new_links (no link)")
                    evolution_result['new_links'] = []
                
                # 建立连接
                if evolution_result.get('new_links'):
                    note.links = evolution_result['new_links']

                # E4: Log pair data for JSD correlation analysis
                if hasattr(self, '_pair_log_path') and self._pair_log_path and evolution_result.get('new_links'):
                    self._log_evolution_pairs(note, evolution_result)
        else:
            self.logger.debug(f"  ✓ First note, no evolution needed")
        
        # Step 4: 写入图存储（包括邻居更新）
        self._save_to_store(note, embeddings_dict)
        self.memories[note_id] = note
        
        # Step 5: 应用邻居更新（如果有）
        if evolution_result and evolution_result.get('neighbor_updates'):
            self._update_neighbors(evolution_result['neighbor_updates'])
        
        # Step 6: 自动维护机制 - Topic 层更新
        self.maintenance_counter += 1
        if self.maintenance_counter >= self.maintenance_threshold:
            self.logger.info(f"[Maintenance] Triggering topic layer update (Counter: {self.maintenance_counter}/{self.maintenance_threshold})...")
            try:
                self.build_topic_layer()
                self.maintenance_counter = 0
                self.logger.info(f"[Maintenance] ✓ Topic layer updated successfully")
            except Exception as e:
                raise RuntimeError('Topic layer maintenance failed') from e
        
        return note_id
    
    def _generate_metadata_with_llm(self, content: str, use_v2: bool = True) -> Dict:
        """
        使用 LLM 生成 MemoryNote 的 metadata（keywords, context）。
        
        Args:
            content: 记忆内容
            use_v2: 是否使用 v2 版本的 prompt（更严格）
            
        Returns:
            metadata: {keywords: [...], context: "..."}
        """
        if use_v2:
            return MemoryNote.analyze_content_v2(content, self.llm_controller, self.prompt_loader)
        else:
            return MemoryNote.analyze_content(content, self.llm_controller)
    
    def _log_evolution_pairs(self, note: MemoryNote, evolution_result: Dict):
        """E4: Log evolution pairs for JSD correlation analysis."""
        if not self._pair_log_path:
            return
        try:
            # Build lookup from current memories (filter non-MemoryNote entries)
            memory_lookup = {m.id: m for m in self.memories.values() if hasattr(m, 'id')}
            for link in evolution_result.get('new_links', []):
                target_id = link.get('target_id', '')
                target_note = memory_lookup.get(target_id)
                if not target_note:
                    continue
                strength = link.get('connection_strength', 0.0)
                record = {
                    "new_note_content": note.content[:500],
                    "new_note_context": note.context[:300] if note.context else "",
                    "candidate_id": target_id,
                    "candidate_content": target_note.content[:500],
                    "candidate_context": target_note.context[:300] if target_note.context else "",
                    "connection_strength": strength,
                    "relation_type": link.get('relation_type', 'RELATED_TO'),
                }
                with open(self._pair_log_path, 'a') as f:
                    import json as _json
                    f.write(_json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.warning(f"[E4] Failed to log pair: {e}")

    def _evolve_memory(self, note: MemoryNote, note_embedding: List[float]) -> Optional[Dict]:
        """Hybrid vector/keyword recall followed by the original adaptive LLM gate."""
        if not note_embedding:
            return None
        try:
            candidates_map = {}
            for note_id, score in self.store.search('MemoryNote', note_embedding, 5,
                                                    exclude_ids=(note.id,)):
                props = self.store.get_node(note_id)['properties']
                candidates_map[note_id] = {
                    'note_id': note_id, 'content': props.get('content', ''),
                    'context': props.get('context', ''),
                    'timestamp': props.get('timestamp', ''),
                    'keywords': props.get('keywords', []),
                    'original_vector_score': score, 'score': score,
                    'keyword_overlap': 0, 'shared_keywords': [], 'sources': ['vector']}

            overlaps = defaultdict(set)
            for keyword in sorted(set(note.keywords)):
                keyword_id = self._keyword_id(keyword)
                for edge in self.store.edges(keyword_id, types='HAS_KEYWORD', direction='in'):
                    if edge['source'] != note.id:
                        overlaps[edge['source']].add(keyword)
            keyword_ranked = sorted(overlaps.items(), key=lambda item: (-len(item[1]), item[0]))[:5]
            for note_id, shared in keyword_ranked:
                props = self.store.get_node(note_id)['properties']
                if note_id in candidates_map:
                    candidate = candidates_map[note_id]
                    candidate['keyword_overlap'] = len(shared)
                    candidate['shared_keywords'] = sorted(shared)
                    candidate['sources'].append('keyword')
                    candidate['score'] = max(candidate['score'], 0.9)
                else:
                    candidates_map[note_id] = {
                        'note_id': note_id, 'content': props.get('content', ''),
                        'context': props.get('context', ''),
                        'timestamp': props.get('timestamp', ''),
                        'keywords': props.get('keywords', []),
                        'original_vector_score': 0.0, 'score': 0.9,
                        'keyword_overlap': len(shared),
                        'shared_keywords': sorted(shared), 'sources': ['keyword']}
            # ==== Step 2: 为仅向量召回的结果补充 keyword_overlap 和 shared_keywords ====
            for note_id, data in candidates_map.items():
                if data['keyword_overlap'] == 0 and note.keywords:
                    # 计算重叠数和共享关键词
                    other_keywords = set(data['keywords'])
                    current_keywords = set(note.keywords)
                    shared = current_keywords & other_keywords
                    data['keyword_overlap'] = len(shared)
                    data['shared_keywords'] = sorted(shared)
            
            all_candidates = list(candidates_map.values())
            # print(f"    Fusion: {len(all_candidates)} unique candidates after merge")
            
            # 打印合并统计
            vector_only = sum(1 for c in all_candidates if c['sources'] == ['vector'])
            keyword_only = sum(1 for c in all_candidates if c['sources'] == ['keyword'])
            both_sources = sum(1 for c in all_candidates if 'vector' in c['sources'] and 'keyword' in c['sources'])
            # print(f"      - Vector only: {vector_only}, Keyword only: {keyword_only}, Both: {both_sources}")
            
            # ==== Step 3: Adaptive Funnel - 宽进严出策略 ====
            # 阈值配置
            STRONG_SCORE_THRESHOLD = 0.8       # 强候选阈值
            WEAK_SCORE_MIN = 0.6               # 弱候选最低阈值
            KEYWORD_OVERLAP_THRESHOLD = 3      # 关键词重叠阈值（需 >= 3）
            MIN_STRONG_COUNT = 3               # 最少强候选数量
            MAX_LLM_CANDIDATES = 5             # 送入 LLM 的最大候选数
            
            # 分类候选集
            strong_candidates = []   # 必进 LLM：Vector Score > 0.8
            weak_candidates = []     # 待定：Vector Score 在 0.6-0.8 之间
            
            for c in all_candidates:
                score = c['score']
                keyword_overlap = c['keyword_overlap']
                
                if score > STRONG_SCORE_THRESHOLD:
                    # Strong Candidates: 必进 LLM
                    strong_candidates.append(c)
                elif score >= WEAK_SCORE_MIN:
                    # Weak Candidates: 待定
                    weak_candidates.append(c)
                elif keyword_overlap >= KEYWORD_OVERLAP_THRESHOLD:
                    # 虽然向量分低，但关键词高度重叠（>= 3），也纳入弱候选
                    weak_candidates.append(c)
                # else: Noise (丢弃) - Vector Score < 0.6 且无足够关键词重叠
            
            # 按分数排序
            strong_candidates.sort(key=lambda x: (-x['score'], x['note_id']))
            weak_candidates.sort(key=lambda x: (-x['score'], x['note_id']))
            
            # print(f"    Adaptive Funnel:")
            # print(f"      - Strong (score>{STRONG_SCORE_THRESHOLD}): {len(strong_candidates)}")
            # print(f"      - Weak ({WEAK_SCORE_MIN}<=score<={STRONG_SCORE_THRESHOLD} or overlap>={KEYWORD_OVERLAP_THRESHOLD}): {len(weak_candidates)}")
            # print(f"      - Noise (discarded): {len(all_candidates) - len(strong_candidates) - len(weak_candidates)}")
            
            # 构建 LLM 输入列表
            llm_candidates = strong_candidates.copy()
            
            # 如果 Strong 数量不足 MIN_STRONG_COUNT，从 Weak 中补齐
            if len(llm_candidates) < MIN_STRONG_COUNT:
                slots_to_fill = MIN_STRONG_COUNT - len(llm_candidates)
                llm_candidates.extend(weak_candidates[:slots_to_fill])
            
            # 限制送入 LLM 的最大候选数量
            llm_candidates = llm_candidates[:MAX_LLM_CANDIDATES]

            # if llm_candidates:
            #     print(f"    LLM candidates breakdown:")
            #     for c in llm_candidates:
            #         tier = "Strong" if c['score'] > STRONG_SCORE_THRESHOLD else "Weak"
            #         info_parts = [f"vector={c['score']:.3f}"]
            #         if c['keyword_overlap'] > 0:
            #             info_parts.append(f"overlap={c['keyword_overlap']}")
            #         print(f"      - {c['note_id'][:8]}: [{tier}] {', '.join(info_parts)} (source={c['source']})")
            
            if not llm_candidates:
                print(f"    No candidates for evolution (all filtered as noise), skipping")
                return None
            
            # 为后续 LLM 判断准备候选集
            candidates_A = llm_candidates
            candidates_B = [
                c for c in llm_candidates 
                if c['score'] > STRONG_SCORE_THRESHOLD or c['keyword_overlap'] >= KEYWORD_OVERLAP_THRESHOLD
            ]
            
            # print(f"    Final: candidates_A={len(candidates_A)}, candidates_B={len(candidates_B)}")
            
            # ==== Step 4: LLM 综合判断 ====
            evolution_result = self._llm_evolution_judgment(note, candidates_A, candidates_B)
            
            # if evolution_result:
            #     updates_count = len(evolution_result.get('neighbor_updates', []))
            #     links_count = len(evolution_result.get('new_links', []))
                # print(f"    ✓ Evolution result: {updates_count} updates, {links_count} links")
            
            return evolution_result
            
        except Exception as e:
            raise RuntimeError('Memory evolution failed') from e

    def _llm_evolution_judgment(self, 
                                note: MemoryNote, 
                                candidates_A: List[Dict],
                                candidates_B: List[Dict]) -> Optional[Dict]:
        """
        根据 enable_semantic_gating 开关执行不同的演化判断逻辑。
        
        Mode A (True): MemDyn (Semantic Gating + Weighted Topology)
        Mode B (False): Baseline (Binary Connection + Average Smoothing)
        
        Args:
            note: 当前新笔记
            candidates_A: 演化候选集（score > 0.75）
            candidates_B: 连接候选集（score > 0.85）
            
        Returns:
            evolution_result: {
                'neighbor_updates': [...],
                'new_links': [...],  # 统一为对象列表格式
            } 或 None
        """
        # 1. 准备数据（通用）
        candidates_json = []
        for c in candidates_A:
            # 生成 match_reason：解释为什么这个候选被召回
            match_reasons = []
            sources = c.get('sources', [])
            shared_keywords = c.get('shared_keywords', [])
            original_vector_score = c.get('original_vector_score', 0.0)
            
            if 'vector' in sources and original_vector_score > 0.7:
                match_reasons.append(f"High Semantic Similarity (score={original_vector_score:.2f})")
            elif 'vector' in sources:
                match_reasons.append(f"Semantic Similarity (score={original_vector_score:.2f})")
            
            if 'keyword' in sources and shared_keywords:
                keywords_str = ', '.join(shared_keywords[:5])  # 最多显示5个关键词
                match_reasons.append(f"Shared Keywords [{keywords_str}]")
            
            # 如果没有明确原因，给一个默认说明
            if not match_reasons:
                match_reasons.append("Potential relevance detected")
            
            match_reason = "; ".join(match_reasons)
            
            candidates_json.append({
                'id': c['note_id'],
                'content': c['content'][:200],  # 截断以节省 tokens
                'context': c['context'][:150] if c['context'] else '',
                'score': round(c['score'], 3),
                'match_reason': match_reason,  # 新增字段：匹配原因
                'is_linking_candidate': c in candidates_B
            })
        
        import json
        candidates_str = json.dumps(candidates_json, ensure_ascii=False, indent=2)
        content = note.content
        context = note.context
        keywords = note.keywords

        # === 分支开始 ===
        if self.enable_semantic_gating:
            # [Mode A: MemDyn] 获取支持门控的 Prompt
            # 优先使用 config 中的 judgment 版本（如 v3_gating_corrected），回退到 v3_gating
            cfg_judgment = self.prompt_loader.get_config_summary().get('judgment', 'v3_gating')
            if 'gating' in cfg_judgment:
                prompt = self.prompt_loader.get_judgment_prompt()(
                    content, context, keywords, candidates_str
                )
            else:
                prompt = self.prompt_loader.get_judgment_prompt(version='v3_gating')(
                    content, context, keywords, candidates_str
                )
            
            # 使用复杂的 Schema（带权重和类型）
            response_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evolution_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "neighbor_updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "rationale": {"type": "string"},
                                        "new_context": {"type": "string"}
                                    },
                                    "required": ["id", "rationale", "new_context"],
                                    "additionalProperties": False
                                }
                            },
                            "new_links": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_id": {"type": "string"},
                                        "relation_type": {"type": "string", "enum": ["SUPPORTS", "CONFLICTS", "RELATED_TO"]},
                                        "connection_strength": {"type": "number"},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["target_id", "relation_type", "connection_strength"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["neighbor_updates", "new_links"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        else:
            # [Mode B: Baseline] 获取普通的 Prompt（A-mem 风格，使用 v2）
            prompt = self.prompt_loader.get_judgment_prompt(version='v2')(
                content, context, keywords, candidates_str
            )
            
            # 使用简单的 Schema（只关注连不连，不关注权重）
            response_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evolution_result_baseline",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "neighbor_updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "new_context": {"type": "string"}
                                    },
                                    "required": ["id", "new_context"],
                                    "additionalProperties": False
                                }
                            },
                            "new_links": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["neighbor_updates", "new_links"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        # === 分支结束 ===
        
        try:
            self.llm_controller.llm._current_caller = "judgment"
            result = get_json_completion_with_retry(
                self.llm_controller.llm,
                prompt,
                response_format=response_schema,
                temperature=0.0,
                max_tokens=8192,
                attempts=3,
            )
            
            # 验证返回的 ID 确实在候选集中
            valid_ids = {c['note_id'] for c in candidates_A}
            valid_link_ids = {c['note_id'] for c in candidates_B}
            
            # 过滤无效的 updates
            result['neighbor_updates'] = [
                u for u in result['neighbor_updates'] 
                if u['id'] in valid_ids
            ]
            
            # 数据归一化：统一 new_links 格式为对象列表
            if not self.enable_semantic_gating:
                # Baseline 模式：把字符串列表转为对象列表
                normalized_links = []
                for link_id in result.get('new_links', []):
                    if link_id in valid_link_ids:
                        normalized_links.append({
                            'target_id': link_id,
                            'relation_type': 'RELATED_TO',
                            'connection_strength': 1.0
                        })
                result['new_links'] = normalized_links
            else:
                # MemDyn 模式：过滤无效的 links
                result['new_links'] = [
                    link for link in result.get('new_links', [])
                    if link.get('target_id') in valid_link_ids
                ]
            
            return result
            
        except Exception as e:
            raise RuntimeError('LLM evolution judgment failed') from e
    
    @staticmethod
    def _keyword_id(keyword):
        return 'keyword:' + keyword

    def _save_to_store(self, note: MemoryNote, embeddings_dict: Dict):
        """Write a note, its keywords and directed typed relationships."""
        self.store.upsert_node(note.id, 'MemoryNote', {
            'content': note.content, 'context': note.context,
            'timestamp': note.timestamp, 'keywords': list(note.keywords),
            'embedding': embeddings_dict['note'],
            'source_id': getattr(note, 'source_id', None),
            'importance_score': note.importance_score,
            'retrieval_count': note.retrieval_count})
        keyword_embeddings = embeddings_dict.get('keywords', {})
        for keyword in sorted(set(note.keywords)):
            keyword_id = self._keyword_id(keyword)
            if self.store.get_node(keyword_id) is None:
                self.store.upsert_node(keyword_id, 'Keyword', {
                    'name': keyword, 'embedding': keyword_embeddings[keyword]})
            self.store.upsert_edge(note.id, keyword_id, 'HAS_KEYWORD')
        for first, second in itertools.combinations(sorted(set(note.keywords)), 2):
            source, target = self._keyword_id(first), self._keyword_id(second)
            existing = next((edge for edge in self.store.edges(source, types='CO_OCCURS')
                             if edge['target'] == target), None)
            count = (existing['properties'].get('count', 0) if existing else 0) + 1
            self.store.upsert_edge(source, target, 'CO_OCCURS', {'count': count})
        for link in note.links:
            link = ({'target_id': link, 'relation_type': 'RELATED_TO',
                     'connection_strength': 1.0} if isinstance(link, str) else link)
            target = link.get('target_id') if isinstance(link, dict) else None
            if target is None or self.store.get_node(target) is None:
                continue
            if self.enable_semantic_gating:
                relation = link.get('relation_type', 'RELATED_TO')
                properties = {'weight': link.get('connection_strength', 0.5),
                              'source': 'memdyn'}
            else:
                relation, properties = 'RELATED_TO', {'source': 'baseline'}
            self.store.upsert_edge(note.id, target, relation, properties)

    def _update_neighbors(self, neighbor_updates: List[Dict]):
        """Merge context and refresh the note vector before changing either view."""
        staged = []
        for update in neighbor_updates or []:
            note_id, context = update['id'], update['new_context']
            node = self.store.get_node(note_id)
            if node is None:
                continue
            if not isinstance(context, str):
                raise ValueError('merged context must be a string')
            embedding = self.get_embedding(context)
            vector = np.asarray(embedding, dtype=np.float64)
            if (vector.ndim != 1 or vector.size != self.store.embedding_dim
                    or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
                raise ValueError('merged context embedding is invalid')
            properties = node['properties']
            properties['context'] = context
            properties['embedding'] = vector.tolist()
            staged.append((note_id, context, properties))
        for note_id, context, properties in staged:
            self.store.upsert_node(note_id, 'MemoryNote', properties)
            if note_id in self.memories:
                self.memories[note_id].context = context

    def _compute_embeddings_for_note(self, note: MemoryNote) -> Dict[str, List[float]]:
        """
        为 Note 和其 Keywords 计算 embeddings。
        批量计算以减少 API 调用次数。
        
        Args:
            note: 记忆笔记对象
            
        Returns:
            embeddings_dict: {
                'note': [embedding_vector],
                'keywords': {
                    'keyword1': [embedding_vector],
                    'keyword2': [embedding_vector],
                    ...
                }
            }
        """
        
        embeddings_dict = {
            'note': None,
            'keywords': {}
        }
        
        try:
            # Paper Eq. 3: a note is indexed by its distilled context.
            embeddings_dict['note'] = self.get_embedding(note.context)
            print(f"    ✓ Note embedding computed (dim: {len(embeddings_dict['note'])})")
            
            # 2. 计算 Keywords 的 embeddings
            # 去重后批量计算（如果有多个相同的 keyword）
            if note.keywords:
                unique_keywords = sorted(set(note.keywords))
                print(f"    Computing embeddings for {len(unique_keywords)} unique keywords...")
                
                for keyword in unique_keywords:
                    # 为每个 keyword 单独计算 embedding
                    # 注意：这里可以优化为批量调用，但需要 API 支持
                    kw_embedding = self.get_embedding(keyword)
                    embeddings_dict['keywords'][keyword] = kw_embedding
                
                print(f"    ✓ Keywords embeddings computed")
            
        except Exception as e:
            raise RuntimeError("Unable to construct complete note embeddings") from e
        
        return embeddings_dict
    
    def _create_note_and_keywords(self, note: MemoryNote, embeddings_dict: Dict):
        """Compatibility helper for write paths that only need note/keyword nodes."""
        self._save_to_store(note, embeddings_dict)



    def _analyze_query_intent(self, query: str, allow_empty_keywords: bool = False) -> Dict:
        """
        Analyze user query using LLM to extract intent.
        
        Returns a dictionary with:
        - topic_desc: A brief description of the macro domain (10-15 words)
        - keywords: List of 3-5 core entities or concepts
        - time_sorting: Time sorting intent ("latest", "oldest", or "none")
        
        Args:
            query: User query string
            
        Returns:
            Dict with 'topic_desc' (str), 'keywords' (List[str]), and 'time_sorting' (str) keys
        """
        print(f"  [1] Analyzing query intent with LLM...")
        
        query_intent_func = self.prompt_loader.get_query_intent_prompt()
        prompt = query_intent_func(query)
        
        response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_intent",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "topic_desc": {
                                    "type": "string",
                                    "description": "Brief description of the macro domain (10-15 words)"
                                },
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of 3-5 core entities or concepts",
                                    "minItems": 3,
                                    "maxItems": 5
                                }
                                # ,
                                # "time_sorting": {
                                #     "type": "string",
                                #     "enum": ["latest", "oldest", "none"],
                                #     "description": "Time sorting intent: 'latest' for recent items, 'oldest' for earliest items, 'none' for no time preference"
                                # }
                            },
                            # "required": ["topic_desc", "keywords", "time_sorting"],
                            "required": ["topic_desc", "keywords"],
                            "additionalProperties": False
                        }
                    }
                }
        controller = self.llm_controller.llm
        previous_caller = getattr(controller, '_current_caller', None)
        self.last_query_intent_trace = []
        last_error = None
        try:
            for attempt in range(3):
                active_prompt = prompt
                active_format = response_format
                if attempt >= 1:
                    active_prompt += ("\nReturn exactly one compact JSON object. "
                                      "Do not add blank lines, explanations, or markdown.")
                if attempt == 2:
                    active_format = {'type': 'json_object'}
                    active_prompt += ("\nThe JSON object must satisfy this schema: "
                                      + json.dumps(response_format['json_schema']['schema'],
                                                   ensure_ascii=False, separators=(',', ':')))
                controller._current_caller = 'query_intent'
                response = None
                failure_reason = None
                try:
                    response = controller.get_completion(
                        active_prompt, response_format=active_format, max_tokens=512)
                    if not isinstance(response, str) or not response.strip():
                        failure_reason = 'empty_response'
                        raise ValueError('empty query intent response')
                    result = json.loads(response)
                    if not isinstance(result, dict):
                        failure_reason = 'invalid_object'
                        raise ValueError('query intent must be a JSON object')
                    if (not isinstance(result.get('topic_desc'), str)
                            or not result['topic_desc'].strip()):
                        failure_reason = 'invalid_topic_desc'
                        raise ValueError('query intent topic_desc must be nonempty')
                    keywords = result.get('keywords')
                    if (not isinstance(keywords, list)
                            or any(not isinstance(keyword, str) or not keyword.strip()
                                   for keyword in keywords)
                            or (not keywords and not allow_empty_keywords)):
                        failure_reason = 'invalid_keywords'
                        raise ValueError('query intent keywords must be a string list')
                    if result.get('time_sorting') not in ('latest', 'oldest', 'none'):
                        result['time_sorting'] = 'none'
                    retry_reason = None
                except Exception as error:
                    last_error = error
                    retry_reason = ('json_decode_error' if isinstance(error, json.JSONDecodeError)
                                    else failure_reason or type(error).__name__)
                    result = None
                metadata = getattr(controller, 'last_completion_metadata', None)
                metadata = metadata if isinstance(metadata, dict) else {}
                self.last_query_intent_trace.append({
                    'attempt': attempt + 1,
                    'caller': 'query_intent',
                    'format': active_format['type'],
                    'output_characters': (len(response) if isinstance(response, str)
                                          else metadata.get('output_characters', 0)),
                    'finish_reason': metadata.get('finish_reason'),
                    'retry_reason': retry_reason,
                })
                if result is not None:
                    print(f"      ✓ Intent analyzed:")
                    print(f"        Topic: {result['topic_desc']}")
                    print(f"        Keywords: {', '.join(result['keywords'])}")
                    print(f"        Time Sorting: {result['time_sorting']}")
                    return result
            raise RuntimeError('Query intent analysis failed after 3 bounded attempts') from last_error
        finally:
            controller._current_caller = previous_caller
    
    def _apply_time_weight(self, notes_data: Dict, sort_mode: str) -> Dict:
        """
        根据时间意图调整分数。
        
        Args:
            notes_data: {note_id: {'note': obj, 'score': float, ...}}
            sort_mode: 'latest' | 'oldest'
            
        Returns:
            更新后的 notes_data（原地修改）
        """
        from datetime import datetime
        
        if sort_mode == 'none':
            return notes_data
        
        # 时间戳格式："%I:%M %p on %d %B, %Y" (例如 "10:30 AM on 15 December, 2025")
        time_format = "%Y-%m-%d %H:%M:%S"
        time2_format = "%I:%M %p on %d %B, %Y"
        now = datetime.now()
        
        print(f"      Applying time weight (mode: {sort_mode})...")
        
        for note_id, data in notes_data.items():
            note = data.get('note')
            if note is None:
                continue
            
            original_score = data.get(
                'final_score',
                max(data.get('topic_score', 0), data.get('keyword_score', 0)),
            )
            
            # 解析时间戳
            note_time = None
            try:
                if hasattr(note, 'timestamp') and note.timestamp:
                    note_time = datetime.strptime(note.timestamp, time_format)
                    # print(f"Parsed timestamp: {note_time}")
            except ValueError as e:
                # 如果解析失败，尝试其他常见格式或使用默认值
                try:
                    # 尝试备用格式
                    note_time = datetime.strptime(note.timestamp, time2_format)
                except:
                    # print(f"Error parsing timestamp: {e}")
                    # print(f"Note timestamp: {note.timestamp}")
                    self.logger.warning(f"Error parsing timestamp: {e} Note timestamp: {note.timestamp}")
            
            # 如果仍无法解析，使用当前时间作为默认值
            if note_time is None:
                note_time = now
            
            # 计算时间差（天数）
            days_diff = abs((now - note_time).days)
            
            # 根据 sort_mode 计算时间因子
            if sort_mode == 'latest':
                # 最近的笔记获得更高的加权
                # 衰减公式：time_factor = 1.0 / (1.0 + 0.1 * days_diff)
                # 当 days_diff = 0 时，time_factor = 1.0
                # 当 days_diff = 10 时，time_factor ≈ 0.5
                # 当 days_diff = 90 时，time_factor ≈ 0.1
                time_factor = 1.0 / (1.0 + 0.1 * days_diff)
            elif sort_mode == 'oldest':
                # 越早的笔记获得更高的加权
                # 使用 sigmoid 变体：更老的获得更高分数
                # time_factor = 1.0 - 1.0 / (1.0 + 0.1 * days_diff)
                # 当 days_diff = 0 时，time_factor = 0
                # 当 days_diff = 10 时，time_factor ≈ 0.5
                # 当 days_diff = 90 时，time_factor ≈ 0.9
                time_factor = 1.0 - 1.0 / (1.0 + 0.1 * days_diff)
            else:
                time_factor = 0.5  # 中性
            
            # 新分数：new_score = original_score * (1.0 + time_factor)
            # 这让最符合时间意图的笔记获得最高 2 倍的加权
            new_score = original_score * (1.0 + time_factor)
            data['final_score'] = new_score
            
            # 记录时间调整信息
            data['time_factor'] = time_factor
            data['days_diff'] = days_diff
        
        return notes_data
    
    def retrieve_topic_aware_structured(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc',
                                       enable_topic_path: bool = True,
                                       enable_keyword_path: bool = True) -> List[Dict]:
        """
        Intent-driven graph navigation retrieval with time-aware scoring.
        返回结构化的检索结果列表，而不是格式化字符串。
        
        与 retrieve_topic_aware 相同的检索逻辑，但返回结构化数据以支持去重和后续处理。
        
        Retrieval Flow:
        Step A: Analyze query intent with LLM and compute embeddings
        Step B: Dual-path graph recall via the typed graph store (with expanded k for time-aware queries)
            - Path 1 (Topic): Search topic_vector_index → find Notes
            - Path 2 (Keyword): Search keyword_vector_index → find Notes
        Step C: Fusion and ranking
        Step C.5: Apply time-aware weighting (if time_sorting != 'none')
        Step D: Logical expansion via RELATED_TO relationships
        Step E: Sort results by score or time
        
        Args:
            query: User query string
            k_final: Final number of anchor notes to return (default 10)
            k_topics: Number of topics to retrieve (default 3)
            k_keywords: Number of keywords to retrieve (default 5)
            enable_logical_expansion: Whether to expand via RELATED_TO (default True)
            sort_by: Sort order for final results (default 'score')
                - 'score': Sort by score descending (highest first)
                - 'time': Sort by timestamp descending (newest first)
                - 'time_asc': Sort by timestamp ascending (oldest first)
            
        Returns:
            List of dictionaries, each containing:
            - id: note ID
            - content: note content
            - score: final retrieval score
            - timestamp: timestamp string
            - parsed_time: parsed datetime object (for sorting)
            - source_label: source type ('Retrieval' or 'Neighbor')
            - sources: list of source descriptions
            - source_id: external source ID for evaluation
        """
        print(f"\n[Retrieve Structured] Query: {query[:60]}...")
        
        # Step A: Analyze & Vectorize
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')
        self.logger.info(f"Topic: {topic_desc}, Keywords: {keywords}, Time Sorting: {time_sorting}")
        
        print(f"  [2] Computing embeddings...")
        try:
            vec_topic = self.get_embedding(topic_desc)
            keywords_str = ' '.join(keywords) if isinstance(keywords, list) else keywords
            vec_keyword = self.get_embedding(keywords_str)
            # print(f"      ✓ Topic embedding: dim={len(vec_topic)}")
            # print(f"      ✓ Keyword embedding: dim={len(vec_keyword)}")
        except Exception as e:
            # print(f"      ✗ Error computing embeddings: {e}")
            return []
        
        # Step B: Dual-path graph recall
        k_topics_actual = k_topics * 2 if time_sorting != 'none' else k_topics
        k_keywords_actual = k_keywords * 2 if time_sorting != 'none' else k_keywords
        
        if time_sorting != 'none':
            # print(f"      (Time-aware mode: expanding retrieval k from {k_topics}/{k_keywords} to {k_topics_actual}/{k_keywords_actual})")
            self.logger.info(f"Time-aware mode: expanding retrieval k from {k_topics}/{k_keywords} to {k_topics_actual}/{k_keywords_actual}")
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}
        
        # Log ablation state
        self.logger.info(f"[Ablation] enable_topic_path={enable_topic_path}, enable_keyword_path={enable_keyword_path}")
        
        # Path 1: Topic-based recall (skip if enable_topic_path=False)
        if enable_topic_path:
            # print(f"  [3] Path 1: Topic-based recall (top-{k_topics_actual} topics)...")
            self.logger.info(f"Path 1: Topic-based recall (top-{k_topics_actual} topics, version={self.topic_retrieval_version})...")
            topic_notes = self._retrieve_via_topic_vector_dispatch(vec_topic, keywords, k_topics_actual)
            # print(f"      ✓ Found {len(topic_notes)} notes via topics")
            self.logger.info(f"Found {len(topic_notes)} notes via topics")
            
            for note_id, score, source_info in topic_notes:
                if note_id not in notes_scores:
                    notes_scores[note_id] = {
                        'note': self._get_note(note_id),
                        'topic_score': 0.0,
                        'keyword_score': 0.0,
                        'sources': []
                    }
                notes_scores[note_id]['topic_score'] += score
                notes_scores[note_id]['sources'].append(f"[Topic] {source_info}")
        else:
            self.logger.info(f"Path 1: Topic-based recall [SKIPPED - Ablation: no_topic]")
        
        # Path 2: Keyword-based recall (skip if enable_keyword_path=False)
        if enable_keyword_path:
            # print(f"  [4] Path 2: Keyword-based recall (top-{k_keywords_actual} keywords)...")
            self.logger.info(f"Path 2: Keyword-based recall (top-{k_keywords_actual} keywords)...")
            keyword_notes = self._retrieve_via_keyword_vector(vec_keyword, k_keywords_actual)
            # print(f"      ✓ Found {len(keyword_notes)} notes via keywords")
            self.logger.info(f"Found {len(keyword_notes)} notes via keywords")
            
            for note_id, score, source_info in keyword_notes:
                if note_id not in notes_scores:
                    notes_scores[note_id] = {
                        'note': self._get_note(note_id),
                        'topic_score': 0.0,
                        'keyword_score': 0.0,
                        'sources': []
                    }
                notes_scores[note_id]['keyword_score'] += score
                notes_scores[note_id]['sources'].append(f"[Keyword] {source_info}")
        else:
            self.logger.info(f"Path 2: Keyword-based recall [SKIPPED - Ablation: no_kws]")
        
        # Filter invalid notes
        notes_scores = {
            nid: data for nid, data in notes_scores.items() 
            if data['note'] is not None
        }
        
        # print(f"  [5] Fusion: {len(notes_scores)} unique notes")
        self.logger.info(f"Fusion: {len(notes_scores)} unique notes")
        
        # Step C: Reciprocal Rank Fusion over independently ranked channels.
        channel_rankings = [
            sorted(
                (note_id for note_id, data in notes_scores.items() if data[field] > 0),
                key=lambda note_id: notes_scores[note_id][field],
                reverse=True,
            )
            for field in ('topic_score', 'keyword_score')
        ]
        fused = reciprocal_rank_fusion(channel_rankings, k0=60, limit=k_final)
        for note_id, score in fused:
            notes_scores[note_id]['final_score'] = score
        
        # Step C.5: Apply time-aware weighting (if time_sorting != 'none')
        # if time_sorting != 'none':
        #     print(f"  [5.5] Applying time-aware weighting (mode: {time_sorting})...")
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
        
        # Sort and select top-k_final as anchor notes
        sorted_notes = [(note_id, notes_scores[note_id]) for note_id, _ in fused]
        
        # print(f" ✓ Selected top-{len(sorted_notes)} anchor notes")
        self.logger.info(f"Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical expansion via RELATED_TO
        expanded_notes = {}
        
        # Add anchor notes
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
        
        if enable_logical_expansion:
            # print(f"  [6] Logical expansion via RELATED_TO relationships...")
            self.logger.info(f"Logical expansion via RELATED_TO relationships...")
            
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5,
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
            
            # print(f" ✓ Expanded to {len(expanded_notes)} total notes")
            self.logger.info(f"Expanded to {len(expanded_notes)} total notes")
        else:
            self.logger.info(f"Logical expansion [SKIPPED - Ablation: no_neighbor]")
        
        # Step E: Convert to structured list
        structured_results = []
        
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if note is None:
                continue
            
            source_type = data.get('source_type', 'unknown')
            score = data.get('score', 0.0)
            sources = data.get('sources', [])
            
            timestamp = getattr(note, 'timestamp', '')
            parsed_time = self._parse_timestamp(timestamp)
            source_id = getattr(note, 'source_id', None)
            
            source_label = 'Retrieval' if source_type == 'anchor' else 'Neighbor'
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': score,
                'timestamp': timestamp,
                'parsed_time': parsed_time,
                'source_label': source_label,
                'sources': sources,
                'source_id': source_id,
                'keywords': note.keywords
            })
        
        # Step F: Sort results based on sort_by parameter
        if sort_by == 'score':
            # Sort by score descending (highest first)
            structured_results.sort(key=lambda x: x['score'], reverse=True)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by score DESC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by score DESC)")
        elif sort_by == 'time':
            # Sort by time descending (newest first)
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by time DESC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by time DESC)")
        elif sort_by == 'time_asc':
            # Sort by time ascending (oldest first)
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by time ASC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by time ASC)")
        else:
            # print(f"  [7] Structured results: {len(structured_results)} items (unsorted)")
            self.logger.info(f"Structured results: {len(structured_results)} items (unsorted)")
        
        return structured_results
    
    def retrieve_topic_aware(self, 
                            query: str, 
                            k_final: int = 10,
                            k_topics: int = 3,
                            k_keywords: int = 5,
                            enable_logical_expansion: bool = True,
                            sort_by: str = 'time_asc') -> str:
        """
        Intent-driven graph navigation retrieval with time-aware scoring.
        返回格式化的上下文字符串（向后兼容）。
        
        此方法现在调用 retrieve_topic_aware_structured 获取结构化数据，
        然后格式化为字符串输出。
        
        Retrieval Flow:
        Step A: Analyze query intent with LLM and compute embeddings
        Step B: Dual-path graph recall via the typed graph store (with expanded k for time-aware queries)
            - Path 1 (Topic): Search topic_vector_index → find Notes
            - Path 2 (Keyword): Search keyword_vector_index → find Notes
        Step C: Fusion and ranking
        Step C.5: Apply time-aware weighting (if time_sorting != 'none')
        Step D: Logical expansion via RELATED_TO relationships
        Step E: Sort and format output with source attribution
        
        Args:
            query: User query string
            k_final: Final number of anchor notes to return (default 10)
            k_topics: Number of topics to retrieve (default 3)
            k_keywords: Number of keywords to retrieve (default 5)
            enable_logical_expansion: Whether to expand via RELATED_TO (default True)
            sort_by: Sort order for results (default 'score')
                - 'score': Sort by score descending (highest first)
                - 'time': Sort by timestamp descending (newest first)
                - 'time_asc': Sort by timestamp ascending (oldest first)
            
        Returns:
            Formatted context string with source attribution
        """
        print(f"\n[Retrieve] Query: {query[:60]}...")
        
        # 调用结构化检索方法
        structured_results = self.retrieve_topic_aware_structured(
            query=query,
            k_final=k_final,
            k_topics=k_topics,
            k_keywords=k_keywords,
            enable_logical_expansion=enable_logical_expansion,
            sort_by=sort_by
        )
        
        if not structured_results:
            return "No results found."
        
        # 格式化输出
        lines = []
        for result in structured_results:
            # 格式: (Score: {score:.2f}) [{source_label}] [Time: {timestamp}] {content}
            line = f"(Score: {result['score']:.2f}) [Time: {result['timestamp']}] [Content]: {result['content']} [Context]: {result['context']} [Keywords]: {result['keywords']}"
            lines.append(line)
        
        # print(f"  [7] Formatting results: {len(lines)} lines")
        self.logger.info(f"Formatting results: {len(lines)} lines")
        return "\n".join(lines)
    
    def retrieve_topic_aware_structured_v3(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc') -> List[Dict]:
        """
        [v2 Experimental] Intent-driven graph navigation retrieval with "Split & Merge" Keyword Strategy.
        
        改进点：
        - 采用 "Split & Merge" 策略：对每个提取出的 keyword 单独计算向量并检索，然后融合结果。
        - 避免了 v1 版本中将多个 keyword 拼接成一个字符串导致向量语义"被稀释"的问题。
        - 提供了更精确的来源追踪 (Source Trace)，例如 "via 'Python'"。
        
        Args:
            (Same as v1)
            
        Returns:
            (Same as v1) List of structured result dicts.
        """
        print(f"\n[Retrieve v3 (Split&Merge)] Query: {query[:60]}...")
        
        # Step A: Analyze Intent
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')
        self.logger.info(f"[v3] Topic: {topic_desc}, Keywords: {keywords}, Time Sorting: {time_sorting}")
        
        # Step A.2: Compute Embeddings (Independent Vectorization)
        print(f"  [2] Computing embeddings (Independent)...")
        topic_vec = None
        keyword_vecs = {}  # {keyword_text: embedding_list}
        
        try:
            # 1. Topic Embedding
            vec_topic = self.get_embedding(topic_desc)
            
            # 2. Keyword Embeddings (Loop)
            # 确保 keywords 是列表
            if not isinstance(keywords, list):
                keywords = [keywords]
            
            for kw in keywords:
                if kw and kw.strip():
                    # 为每个关键词单独计算向量
                    vec = self.get_embedding(kw)
                    keyword_vecs[kw] = vec
            
            print(f"      ✓ Computed vectors for {len(keyword_vecs)} unique keywords")
            
        except Exception as e:
            print(f"      ✗ Error computing embeddings: {e}")
            return []
        
        # Determine actual K values (expand for time-aware mode)
        factor = 2 if time_sorting != 'none' else 1
        # factor = 1
        # k_topics_actual = k_topics * factor
        # k_keywords_total_budget = k_keywords * factor
        k_topics_actual = k_topics * factor
        k_keywords_total_budget = k_keywords * factor
        
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}

        per_keyword_k = k_keywords_total_budget
        
        # Step B.1: Path 1 - Topic-based recall (Same as v1)
        self.logger.info(f"Path 1: Topic-based recall (top-{k_topics_actual}, version={self.topic_retrieval_version})...")
        topic_notes = self._retrieve_via_topic_vector_dispatch(vec_topic, keywords, k_topics_actual)
        
        for note_id, score, source_info in topic_notes:
            if note_id not in notes_scores:
                notes_scores[note_id] = {
                    'note': self._get_note(note_id),
                    'topic_score': 0.0,
                    'keyword_score': 0.0,
                    'sources': []
                }
            # 累加分数（因为 Topic 往往代表宏观背景，多次命中说明相关性强）
            notes_scores[note_id]['topic_score'] += score
            notes_scores[note_id]['sources'].append(f"[Topic] {source_info}")
        
        # Step B.2: Path 2 - Keyword-based recall (Split & Merge)
        self.logger.info(f"Path 2: Keyword-based recall (Iterating {len(keyword_vecs)} keywords)...")
        
        # 动态分配 K 值：如果关键词很多，每个关键词分配较少的 K，避免召回爆炸
        # 但至少保证每个关键词能召回 2 个
        # num_kws = len(keyword_vecs)
        # if num_kws > 0:
        #     per_keyword_k = max(2, int(k_keywords_total_budget / num_kws * 1.5)) # 稍微放宽一点预算 *1.5
        # else:
        #     per_keyword_k = k_keywords_total_budget
        per_keyword_k = k_keywords_total_budget
            
        # 临时存储所有 keyword 召回结果，用于去重
        # note_id -> {'score': max_score, 'sources': [src1, src2]}
        kw_temp_results = {}
        
        for kw_text, kw_vec in keyword_vecs.items():
            # 复用底层的单向量检索方法
            sub_results = self._retrieve_via_keyword_vector(kw_vec, per_keyword_k)
            
            for note_id, score, source_desc in sub_results:
                # 构造更精细的来源描述
                # source_desc 原本是 "Concept Keywords: Python, Coding"
                # 我们加上 "via 'Python'"
                specific_source = f"{source_desc} (via target '{kw_text}')"
                
                if note_id not in kw_temp_results:
                    kw_temp_results[note_id] = {
                        'score': score,
                        'sources': [specific_source]
                    }
                else:
                    # 如果同一个笔记被多个目标关键词召回
                    # 1. 分数取最大值 (代表最强的一次匹配)
                    kw_temp_results[note_id]['score'] += score
                    # 2. 来源合并
                    kw_temp_results[note_id]['sources'].append(specific_source)
        
        # 将合并后的 Keyword 结果注入主表
        self.logger.info(f"Found {len(kw_temp_results)} unique notes via keywords")
        
        for note_id, data in kw_temp_results.items():
            score = data['score']
            sources = data['sources']
            
            if note_id not in notes_scores:
                # 如果 Topic 路径没找到，需要先获取 Note 对象
                note_obj = self._get_note(note_id)
                if not note_obj:
                    continue
                    
                notes_scores[note_id] = {
                    'note': note_obj,
                    'topic_score': 0.0,
                    'keyword_score': 0.0,
                    'sources': []
                }
            
            # 记录 Keyword 分数
            # 如果之前已经有 keyword_score (极少情况)，取最大值
            # notes_scores[note_id]['keyword_score'] += max(notes_scores[note_id]['keyword_score'], score)
            notes_scores[note_id]['keyword_score'] += score
            
            # 添加来源
            for src in sources:
                notes_scores[note_id]['sources'].append(f"[Keyword] {src}")
        
        # Step C: Fusion & Ranking (Same as v1)
        # 过滤无效笔记
        notes_scores = {nid: data for nid, data in notes_scores.items() if data['note'] is not None}
        
        # 计算最终分数
        for note_id, data in notes_scores.items():
            # 融合公式：0.4 Topic + 0.6 Keyword
            data['final_score'] = data['topic_score'] * 0.4 + data['keyword_score'] * 0.6
        
        # Step C.5: Time weighting (Optional, keeping consistent with v1 comment out)
        # if time_sorting != 'none':
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
            
        # Select Top-K Anchors
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        self.logger.info(f"[v2] Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical Expansion (Same as v1)
        expanded_notes = {}
        
        # Add anchors
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
            
        if enable_logical_expansion:
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5, # 邻居降权
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
        
        # Step E: Format Structured Results (Same as v1)
        structured_results = []
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if not note: continue
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': data.get('score', 0.0),
                'timestamp': getattr(note, 'timestamp', ''),
                'parsed_time': self._parse_timestamp(getattr(note, 'timestamp', '')),
                'source_label': 'Retrieval' if data.get('source_type') == 'anchor' else 'Neighbor',
                'sources': data.get('sources', []),
                'source_id': getattr(note, 'source_id', None),
                'keywords': note.keywords
            })
            
        # Sorting
        if sort_by == 'score':
            structured_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'time':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            
        return structured_results

    def retrieve_topic_aware_structured_v4(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc') -> List[Dict]:
        """
        [v2 Experimental] Intent-driven graph navigation retrieval with "Split & Merge" Keyword Strategy.
        
        改进点：
        - 采用 "Split & Merge" 策略：对每个提取出的 keyword 单独计算向量并检索，然后融合结果。
        - 避免了 v1 版本中将多个 keyword 拼接成一个字符串导致向量语义"被稀释"的问题。
        - 提供了更精确的来源追踪 (Source Trace)，例如 "via 'Python'"。
        
        Args:
            (Same as v1)
            
        Returns:
            (Same as v1) List of structured result dicts.
        """
        print(f"\n[Retrieve v3 (Split&Merge)] Query: {query[:60]}...")
        
        # Step A: Analyze Intent
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')
        self.logger.info(f"[v3] Topic: {topic_desc}, Keywords: {keywords}, Time Sorting: {time_sorting}")
        
        # Step A.2: Compute Embeddings (Independent Vectorization)
        print(f"  [2] Computing embeddings (Independent)...")
        topic_vec = None
        keyword_vecs = {}  # {keyword_text: embedding_list}
        
        try:
            # 1. Topic Embedding
            vec_topic = self.get_embedding(topic_desc)
            
            # 2. Keyword Embeddings (Loop)
            # 确保 keywords 是列表
            if not isinstance(keywords, list):
                keywords = [keywords]
            
            for kw in keywords:
                if kw and kw.strip():
                    # 为每个关键词单独计算向量
                    vec = self.get_embedding(kw)
                    keyword_vecs[kw] = vec
            
            print(f"      ✓ Computed vectors for {len(keyword_vecs)} unique keywords")
            
        except Exception as e:
            print(f"      ✗ Error computing embeddings: {e}")
            return []
        
        # Determine actual K values (expand for time-aware mode)
        factor = 2 if time_sorting != 'none' else 1
        # factor = 1
        # k_topics_actual = k_topics * factor
        # k_keywords_total_budget = k_keywords * factor
        k_topics_actual = k_topics * factor
        k_keywords_total_budget = k_keywords * factor
        
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}
        
        # Step B.1: Path 1 - Topic-based recall (Same as v1)
        self.logger.info(f"Path 1: Topic-based recall (top-{k_topics_actual}, version={self.topic_retrieval_version})...")
        topic_notes = self._retrieve_via_topic_vector_dispatch(vec_topic, keywords, k_topics_actual)
        
        for note_id, score, source_info in topic_notes:
            if note_id not in notes_scores:
                notes_scores[note_id] = {
                    'note': self._get_note(note_id),
                    'topic_score': 0.0,
                    'keyword_score': 0.0,
                    'sources': []
                }
            # 累加分数（因为 Topic 往往代表宏观背景，多次命中说明相关性强）
            notes_scores[note_id]['topic_score'] += score
            notes_scores[note_id]['sources'].append(f"[Topic] {source_info}")
        
        # Step B.2: Path 2 - Keyword-based recall (Split & Merge)
        self.logger.info(f"Path 2: Keyword-based recall (Iterating {len(keyword_vecs)} keywords)...")
        
        # 动态分配 K 值：如果关键词很多，每个关键词分配较少的 K，避免召回爆炸
        # 但至少保证每个关键词能召回 2 个
        # num_kws = len(keyword_vecs)
        # if num_kws > 0:
        #     per_keyword_k = max(2, int(k_keywords_total_budget / num_kws * 1.5)) # 稍微放宽一点预算 *1.5
        # else:
        #     per_keyword_k = k_keywords_total_budget
        per_keyword_k = k_keywords_total_budget
            
        # 临时存储所有 keyword 召回结果，用于去重
        # note_id -> {'score': max_score, 'sources': [src1, src2]}
        kw_temp_results = {}
        
        for kw_text, kw_vec in keyword_vecs.items():
            # 复用底层的单向量检索方法
            sub_results = self._retrieve_via_keyword_vector(kw_vec, per_keyword_k)
            
            for note_id, score, source_desc in sub_results:
                specific_source = f"{source_desc} (via target '{kw_text}')"
                
                if note_id not in kw_temp_results:
                    kw_temp_results[note_id] = {
                        'max_score': score,
                        'matched_queries': {kw_text},
                        'sources': [specific_source]
                    }
                else:
                    # 策略核心：MAX 基准 + 记录命中词
                    kw_temp_results[note_id]['max_score'] = max(kw_temp_results[note_id]['max_score'], score)
                    kw_temp_results[note_id]['matched_queries'].add(kw_text)
                    kw_temp_results[note_id]['sources'].append(specific_source)
        
        # 3. 计算最终 Keyword Score (Base + Bonus) 并注入主表
        self.logger.info(f"Found {len(kw_temp_results)} unique notes via keywords")
        
        # 覆盖度奖励系数 (Alpha)
        COVERAGE_ALPHA = 0.15
        
        for note_id, data in kw_temp_results.items():
            max_score = data['max_score']
            match_count = len(data['matched_queries'])
            sources = data['sources']
            
            # 【核心公式】：Final = Max * (1 + Alpha * (Count - 1))
            # 例子：命中 2 个词 -> Bonus 15%; 命中 3 个词 -> Bonus 30%
            coverage_boost = 1.0 + COVERAGE_ALPHA * (match_count - 1)
            final_keyword_score = max_score * coverage_boost
            
            # 注入主表
            if note_id not in notes_scores:
                note_obj = self._get_note(note_id)
                if not note_obj: continue
                notes_scores[note_id] = {
                    'note': note_obj,
                    'topic_score': 0.0,
                    'keyword_score': 0.0,
                    'sources': []
                }
            
            # 更新 Keyword 分数 (取 max 是为了防止异常情况，通常直接赋值)
            notes_scores[note_id]['keyword_score'] = max(notes_scores[note_id]['keyword_score'], final_keyword_score)
            
            # 添加来源
            for src in sources:
                notes_scores[note_id]['sources'].append(f"[Keyword] {src}")
        
        # Step C: Fusion & Ranking (Same as v1)
        # 过滤无效笔记
        notes_scores = {nid: data for nid, data in notes_scores.items() if data['note'] is not None}
        
        # 计算最终分数
        for note_id, data in notes_scores.items():
            # 融合公式：0.4 Topic + 0.6 Keyword
            data['final_score'] = data['topic_score'] * 0.4 + data['keyword_score'] * 0.6
        
        # Step C.5: Time weighting (Optional, keeping consistent with v1 comment out)
        # if time_sorting != 'none':
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
            
        # Select Top-K Anchors
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        self.logger.info(f"[v2] Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical Expansion (Same as v1)
        expanded_notes = {}
        
        # Add anchors
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
            
        if enable_logical_expansion:
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5, # 邻居降权
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
        
        # Step E: Format Structured Results (Same as v1)
        structured_results = []
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if not note: continue
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': data.get('score', 0.0),
                'timestamp': getattr(note, 'timestamp', ''),
                'parsed_time': self._parse_timestamp(getattr(note, 'timestamp', '')),
                'source_label': 'Retrieval' if data.get('source_type') == 'anchor' else 'Neighbor',
                'sources': data.get('sources', []),
                'source_id': getattr(note, 'source_id', None),
                'keywords': note.keywords
            })
            
        # Sorting
        if sort_by == 'score':
            structured_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'time':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            
        return structured_results

    def retrieve_topic_aware_structured_v5(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc') -> List[Dict]:
        """
        [v2 Experimental] Intent-driven graph navigation retrieval with "Split & Merge" Keyword Strategy.
        
        改进点：
        - 采用 "Split & Merge" 策略：对每个提取出的 keyword 单独计算向量并检索，然后融合结果。
        - 避免了 v1 版本中将多个 keyword 拼接成一个字符串导致向量语义"被稀释"的问题。
        - 提供了更精确的来源追踪 (Source Trace)，例如 "via 'Python'"。
        
        Args:
            (Same as v1)
            
        Returns:
            (Same as v1) List of structured result dicts.
        """
        print(f"\n[Retrieve v3 (Split&Merge)] Query: {query[:60]}...")
        
        # Step A: Analyze Intent
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')

        # 确保 keywords 是列表
        if not isinstance(keywords, list):
            keywords = [keywords]

        # 过滤空词
        keywords = [k for k in keywords if k and k.strip()]
        self.logger.info(f"[v5] Topic: {topic_desc}, Keywords: {keywords}")


        keyword_vecs = {}      # Path 3: Split
        combined_vec = None    # Path 2: Combined
        topic_vec = None       # Path 1: Topic
        
        # 1. Topic Embedding
        vec_topic = self.get_embedding(topic_desc)

        # 2. Combined Keywords (String)
        if keywords:
            combined_str = " ".join(keywords)
            combined_vec = self.get_embedding(combined_str)
        
        # 3. Split Keywords (List)
        for kw in keywords:
            keyword_vecs[kw] = self.get_embedding(kw)
        
        # 确定 K 值 (Time-aware expansion)
        factor = 2 if time_sorting != 'none' else 1
        k_topics_actual = k_topics * factor
        # 总预算分给 Combined 和 Split 两路
        k_keywords_combined = k_keywords * factor  # Combined 路给足预算
        k_keywords_split = k_keywords * factor     # Split 路也给足预
        
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}
        
        # ==================================================================================
        # Path 1: Topic-based Recall (Macro)
        # ==================================================================================
        self.logger.info(f"Path 1: Topic-based Recall (top-{k_topics_actual}, version={self.topic_retrieval_version})...")
        topic_notes = self._retrieve_via_topic_vector_dispatch(topic_vec, keywords, k_topics_actual)
        for note_id, score, source_info in topic_notes:
            # self._init_score_entry(notes_scores, note_id)
            if note_id not in notes_scores:
                    note_obj = self._get_note(note_id)
                    # 如果从图存储没拿到对象，暂时存 None，后续过滤
                    notes_scores[note_id] = {
                        'note': note_obj,
                        'topic_score': 0.0,
                        'combined_score': 0.0,
                        'split_score': 0.0,
                        'sources': []
                    }
            notes_scores[note_id]['topic_score'] += score
            notes_scores[note_id]['sources'].append(f"[Topic] {source_info}")
        
        # ==================================================================================
        # Path 2: Combined Keyword Recall (The "Semantic Center")
        # ==================================================================================
        # 这就是你最开始的方案，它能召回 "Red Bull" 这种复合概念
        if combined_vec:
            self.logger.info(f"Path 2: Combined Keyword Recall ('{' '.join(keywords)}')...")
            combined_results = self._retrieve_via_keyword_vector(combined_vec, k_keywords_combined)
            
            for note_id, score, source_desc in combined_results:
                # self._init_score_entry(notes_scores, note_id)
                if note_id not in notes_scores:
                    note_obj = self._get_note(note_id)
                    # 如果从图存储没拿到对象，暂时存 None，后续过滤
                    notes_scores[note_id] = {
                        'note': note_obj,
                        'topic_score': 0.0,
                        'combined_score': 0.0,
                        'split_score': 0.0,
                        'sources': []
                    }
                # 记录 Combined 分数
                notes_scores[note_id]['combined_score'] = max(notes_scores[note_id]['combined_score'], score)
                notes_scores[note_id]['sources'].append(f"[Combined] {source_desc} (via combined intent)")
        
        # ==================================================================================
        # Path 3: Split Keyword Recall (The "Precision Coverage")
        # ==================================================================================
        # 这是 v4 方案，用于查漏补缺
        if keyword_vecs:
            self.logger.info(f"Path 3: Split Keyword Recall ({len(keyword_vecs)} keywords)...")
            
            # 这里的 K 值可以稍微激进一点，因为有 Combined 路保底
            per_keyword_k = k_keywords_split 
            
            kw_temp_results = {} # note_id -> {max_score, matched_count, sources}
            
            for kw_text, kw_vec in keyword_vecs.items():
                sub_results = self._retrieve_via_keyword_vector(kw_vec, per_keyword_k)
                
                for note_id, score, source_desc in sub_results:
                    specific_source = f"{source_desc} (via '{kw_text}')"
                    
                    if note_id not in kw_temp_results:
                        kw_temp_results[note_id] = {
                            'max_score': score,
                            'matched_queries': {kw_text},
                            'sources': [specific_source]
                        }
                    else:
                        kw_temp_results[note_id]['max_score'] = max(kw_temp_results[note_id]['max_score'], score)
                        kw_temp_results[note_id]['matched_queries'].add(kw_text)
                        kw_temp_results[note_id]['sources'].append(specific_source)
            
            # 注入主表 (带覆盖度奖励)
            COVERAGE_ALPHA = 0.15
            for note_id, data in kw_temp_results.items():
                # self._init_score_entry(notes_scores, note_id)
                if note_id not in notes_scores:
                    note_obj = self._get_note(note_id)
                    # 如果从图存储没拿到对象，暂时存 None，后续过滤
                    notes_scores[note_id] = {
                        'note': note_obj,
                        'topic_score': 0.0,
                        'combined_score': 0.0,
                        'split_score': 0.0,
                        'sources': []
                    }
                
                max_s = data['max_score']
                match_count = len(data['matched_queries'])
                boost = 1.0 + COVERAGE_ALPHA * (match_count - 1)
                final_s = max_s * boost
                
                notes_scores[note_id]['split_score'] = max(notes_scores[note_id]['split_score'], final_s)
                for src in data['sources']:
                    notes_scores[note_id]['sources'].append(f"[Split] {src}")
        
        # ==================================================================================
        # Step C: Fusion & Ranking
        # ==================================================================================
        # 过滤无效笔记
        notes_scores = {nid: data for nid, data in notes_scores.items() if data['note'] is not None}
        
        for note_id, data in notes_scores.items():
            # 融合公式：
            # Topic (0.3) + Combined (0.4) + Split (0.3)
            # Combined 权重稍高，因为它通常最准；Split 用于辅助覆盖
            t_score = data['topic_score']
            c_score = data['combined_score']
            s_score = data['split_score']
            
            # 如果 Combined 和 Split 都有分，取最大的那个作为 Keyword Score，或者加权融合
            # 这里建议：取 Combined 和 Split 中的最大值作为 "Keyword Representation"，然后再和 Topic 融合
            # 这样避免了双重计算
            best_keyword_score = max(c_score, s_score)
            
            data['final_score'] = t_score * 0.4 + best_keyword_score * 0.6
        
        # Step C.5: Time weighting (Optional, keeping consistent with v1 comment out)
        # if time_sorting != 'none':
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
            
        # Select Top-K Anchors
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        self.logger.info(f"[v2] Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical Expansion (Same as v1)
        expanded_notes = {}
        
        # Add anchors
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
            
        if enable_logical_expansion:
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5, # 邻居降权
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
        
        # Step E: Format Structured Results (Same as v1)
        structured_results = []
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if not note: continue
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': data.get('score', 0.0),
                'timestamp': getattr(note, 'timestamp', ''),
                'parsed_time': self._parse_timestamp(getattr(note, 'timestamp', '')),
                'source_label': 'Retrieval' if data.get('source_type') == 'anchor' else 'Neighbor',
                'sources': data.get('sources', []),
                'source_id': getattr(note, 'source_id', None),
                'keywords': note.keywords
            })
            
        # Sorting
        if sort_by == 'score':
            structured_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'time':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            
        return structured_results
    
    def retrieve_topic_aware_structured_v6(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc') -> List[Dict]:
        """
        [v2 Experimental] Intent-driven graph navigation retrieval with "Split & Merge" Keyword Strategy.
        
        改进点：
        - 采用 "Split & Merge" 策略：对每个提取出的 keyword 单独计算向量并检索，然后融合结果。
        - 避免了 v1 版本中将多个 keyword 拼接成一个字符串导致向量语义"被稀释"的问题。
        - 提供了更精确的来源追踪 (Source Trace)，例如 "via 'Python'"。
        
        Args:
            (Same as v1)
            
        Returns:
            (Same as v1) List of structured result dicts.
        """
        print(f"\n[Retrieve v3 (Split&Merge)] Query: {query[:60]}...")
        
        # Step A: Analyze Intent
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')

        # 确保 keywords 是列表
        if not isinstance(keywords, list):
            keywords = [keywords]

        # 过滤空词
        keywords = [k for k in keywords if k and k.strip()]
        self.logger.info(f"[v5] Topic: {topic_desc}, Keywords: {keywords}")


        topic_vec = self.get_embedding(topic_desc)
        combined_vec = self.get_embedding(" ".join(keywords)) if keywords else None
        keyword_vecs = {kw: self.get_embedding(kw) for kw in keywords}
        
        # 确定 K 值 (Time-aware expansion)
        factor = 2 if time_sorting != 'none' else 1
        k_topics_actual = k_topics * factor
        # Combined 路非常重要，给足预算，确保 v1 的效果
        k_keywords_combined = max(k_keywords * factor, 10) 
        k_keywords_split = k_keywords * factor
        
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}
        
        # ==================================================================================
        # Path 1: Topic-based Recall (Macro)
        # ==================================================================================
        self.logger.info(f"Path 1: Topic-based Recall (top-{k_topics_actual}, version={self.topic_retrieval_version})...")
        topic_notes = self._retrieve_via_topic_vector_dispatch(topic_vec, keywords, k_topics_actual)
        for nid, score, src in topic_notes:
            self._init_score_entry(notes_scores, nid)
            notes_scores[nid]['topic_score'] = score
            notes_scores[nid]['sources'].append(f"[Topic] {src}")
        
        # Path 2: Combined Keyword (v1 逻辑 - 核心路径)
        if combined_vec:
            combined_results = self._retrieve_via_keyword_vector(combined_vec, k_keywords_combined)
            for nid, score, src in combined_results:
                self._init_score_entry(notes_scores, nid)
                notes_scores[nid]['combined_score'] = score
                notes_scores[nid]['sources'].append(f"[Combined] {src}")
        
        # Path 3: Split Keyword (v4 逻辑 - 查漏补缺)
        if keyword_vecs:
            # 这里 K 值不要太大，避免引入太多噪声
            per_keyword_k = max(3, int(k_keywords_split / len(keyword_vecs)) + 2)
            
            for kw_text, kw_vec in keyword_vecs.items():
                sub_results = self._retrieve_via_keyword_vector(kw_vec, per_keyword_k)
                for nid, score, src in sub_results:
                    self._init_score_entry(notes_scores, nid)
                    # 【关键修正 1】: 移除 Coverage Bonus，只记录 MAX 分数
                    # 避免 "命中3个垃圾词" 的分数 超过 "命中1个核心词"
                    notes_scores[nid]['split_score'] = max(notes_scores[nid]['split_score'], score)
                    notes_scores[nid]['sources'].append(f"[Split] {src} (via '{kw_text}')")
        
        # Step C: Safe Fusion (关键修正)
        filtered_scores = {nid: data for nid, data in notes_scores.items() if data['note'] is not None}
        
        for nid, data in filtered_scores.items():
            t_score = data['topic_score']
            c_score = data['combined_score']
            s_score = data['split_score']
            
            # 【关键修正 2】: 优先信任 Combined Score
            # 只有当 Split Score 真的非常高 (说明细节匹配极其完美) 时才采用
            # 否则，我们给 Combined Score 一个 1.1 倍的置信度加成
            
            effective_keyword_score = max(c_score * 1.1, s_score)
            
            # 融合公式
            data['final_score'] = t_score * 0.4 + effective_keyword_score * 0.6
            
            # 【关键修正 3】: 如果 Combined Path 召回了它，且分数尚可，给予保底加分
            # 这相当于一种 "Hard Bias"，确保 v1 能找到的东西尽量不丢
            if c_score > 0.75:
                data['final_score'] += 0.05
        
        # Step C.5: Time weighting (Optional, keeping consistent with v1 comment out)
        # if time_sorting != 'none':
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
            
        # Select Top-K Anchors
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        self.logger.info(f"[v2] Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical Expansion (Same as v1)
        expanded_notes = {}
        
        # Add anchors
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
            
        if enable_logical_expansion:
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5, # 邻居降权
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
        
        # Step E: Format Structured Results (Same as v1)
        structured_results = []
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if not note: continue
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': data.get('score', 0.0),
                'timestamp': getattr(note, 'timestamp', ''),
                'parsed_time': self._parse_timestamp(getattr(note, 'timestamp', '')),
                'source_label': 'Retrieval' if data.get('source_type') == 'anchor' else 'Neighbor',
                'sources': data.get('sources', []),
                'source_id': getattr(note, 'source_id', None),
                'keywords': note.keywords
            })
            
        # Sorting
        if sort_by == 'score':
            structured_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'time':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            
        return structured_results

    def retrieve_topic_aware_structured_v7(self,
                                   query: str,
                                   k_final: int = 10,
                                   k_topics: int = 3,
                                   k_keywords: int = 5,
                                   enable_logical_expansion: bool = True,
                                   sort_by: str = 'time_asc',
                                   enable_topic_path: bool = True,
                                   enable_keyword_path: bool = True) -> List[Dict]:
        """
        [v7 Symbolic Fusion] Triangle Recall + Text-based Re-ranking.
        
        改进策略：
        1. **Triangle Recall**: 沿用 v5 策略，确保召回率最大化。
        2. **Symbolic Bonus (核心)**: 不再单纯依赖向量相似度。在排序前，检查 Note 内容是否【逐字包含】Query Keywords。
           - 包含一个关键词：加分 (e.g., +0.05)
           - 解决向量"语义漂移"导致的 Ranking 不准问题。
        """
        print(f"\n[Retrieve v7 (Symbolic Fusion)] Query: {query[:60]}...")
        
        # 1. Intent Analysis (无需修改 Prompt)
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords'] # List[str]
        time_sorting = intent.get('time_sorting', 'none')
        
        if not isinstance(keywords, list): keywords = [keywords]
        # 预处理关键词用于文本匹配：转小写，去空
        target_keywords = [k.lower().strip() for k in keywords if k and k.strip()]
        
        # 2. 执行向量召回 (复用 v5 的三角逻辑)
        # 这里为了代码简洁，逻辑是：计算 Embeddings -> 图存储检索 -> 合并分数
        
        # --- (A) Embeddings ---
        topic_vec = self.get_embedding(topic_desc)
        combined_vec = self.get_embedding(" ".join(keywords)) if keywords else None
        keyword_vecs = {kw: self.get_embedding(kw) for kw in keywords}
        
        # --- (B) Retrieval (Triangle Strategy) ---
        factor = 2 if time_sorting != 'none' else 1
        notes_scores = {} # nid -> {scores..., note object}
        
        # Log ablation state
        self.logger.info(f"[Ablation] enable_topic_path={enable_topic_path}, enable_keyword_path={enable_keyword_path}")

        # Path 1: Topic (skip if enable_topic_path=False)
        if enable_topic_path:
            self.logger.info(f"Path 1: Topic Recall (top-{k_topics * factor}, version={self.topic_retrieval_version})...")
            t_notes = self._retrieve_via_topic_vector_dispatch(topic_vec, keywords, k_topics * factor)
            for nid, sc, src in t_notes:
                self._init_score_entry(notes_scores, nid)
                notes_scores[nid]['topic_score'] = sc
                notes_scores[nid]['sources'].append(f"[Topic] {src}")
        else:
            self.logger.info(f"Path 1: Topic Recall [SKIPPED - Ablation: no_topic]")

        # Path 2: Combined (v1 Logic - High Precision) - skip if enable_keyword_path=False
        if enable_keyword_path and combined_vec:
            c_notes = self._retrieve_via_keyword_vector(combined_vec, k_keywords * factor)
            for nid, sc, src in c_notes:
                self._init_score_entry(notes_scores, nid)
                notes_scores[nid]['combined_score'] = sc
                notes_scores[nid]['sources'].append(f"[Combined] {src}")
        elif not enable_keyword_path:
            self.logger.info(f"Path 2: Combined Keyword Recall [SKIPPED - Ablation: no_kws]")
        
        # Path 3: Split (v4 Logic - High Coverage) - skip if enable_keyword_path=False
        if enable_keyword_path and keyword_vecs:
            # 适当放宽 K 值，让更多候选进入 Re-ranking 阶段
            per_kw_k = k_keywords * factor 
            for kw_text, kw_vec in keyword_vecs.items():
                s_notes = self._retrieve_via_keyword_vector(kw_vec, per_kw_k)
                for nid, sc, src in s_notes:
                    self._init_score_entry(notes_scores, nid)
                    notes_scores[nid]['split_score'] = max(notes_scores[nid]['split_score'], sc)
                    notes_scores[nid]['sources'].append(f"[Split] {src}")
        elif not enable_keyword_path:
            self.logger.info(f"Path 3: Split Keyword Recall [SKIPPED - Ablation: no_kws]")

        # ==========================================================
        # Step C: Symbolic Re-ranking (v7 核心)
        # ==========================================================
        
        # 过滤无效笔记
        candidates = {nid: data for nid, data in notes_scores.items() if data['note'] is not None}
        
        # 参数设定
        BONUS_PER_MATCH = 0.05  # 每个精确匹配的词奖励 0.05 分
        MAX_BONUS = 0.25        # 最大奖励上限 (防止过度干预)
        
        for nid, data in candidates.items():
            # 1. 基础向量分 (Vector Score)
            # 采用 v5_robust 的融合逻辑：优先信赖 Combined
            t_score = data['topic_score']
            c_score = data['combined_score']
            s_score = data['split_score']
            
            # 向量融合: 如果 Combined 分数高，优先用它；否则看 Split
            vector_score = max(c_score, s_score)
            if c_score > s_score: 
                vector_score *= 1.05 # 给 Combined 微小加权
            
            base_score = t_score * 0.4 + vector_score * 0.6
            
            # 2. 文本匹配分 (Symbolic Bonus)
            # 这是一个"硬"指标：向量可能看走眼，但如果词确实在里面，那就错不了
            content_lower = data['note'].content.lower()
            match_count = 0
            matched_terms = []
            
            for kw in target_keywords:
                # 简单子串匹配 (比正则快且够用)
                if kw in content_lower:
                    match_count += 1
                    matched_terms.append(kw)
            
            # 计算奖励
            symbolic_bonus = min(match_count * BONUS_PER_MATCH, MAX_BONUS)
            
            # 3. 最终得分
            data['final_score'] = base_score + symbolic_bonus
            
            if symbolic_bonus > 0:
                data['sources'].append(f"[Symbolic] Bonus +{symbolic_bonus:.2f} (Matched: {matched_terms})")

        # Select Top-K Anchors
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        self.logger.info(f"[v2] Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical Expansion (Same as v1)
        expanded_notes = {}
        
        # Add anchors
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
            
        if enable_logical_expansion:
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)
                for neighbor_info in neighbors:
                    neighbor_id = neighbor_info["id"] if isinstance(neighbor_info, dict) else neighbor_info
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': data['final_score'] * 0.5, # 邻居降权
                                'sources': [f"Logical neighbor of {note_id[:8]}"]
                            }
        
        # Step E: Format Structured Results (Same as v1)
        structured_results = []
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if not note: continue
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': data.get('score', 0.0),
                'timestamp': getattr(note, 'timestamp', ''),
                'parsed_time': self._parse_timestamp(getattr(note, 'timestamp', '')),
                'source_label': 'Retrieval' if data.get('source_type') == 'anchor' else 'Neighbor',
                'sources': data.get('sources', []),
                'source_id': getattr(note, 'source_id', None),
                'keywords': note.keywords
            })
            
        # Sorting
        if sort_by == 'score':
            structured_results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == 'time':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            
        return structured_results

    def retrieve_topic_aware_structured_v8_gating(self,
                                       query: str,
                                       k_final: int = 10,
                                       k_topics: int = 3,
                                       k_keywords: int = 5,
                                       enable_logical_expansion: bool = True,
                                       sort_by: str = 'time_asc',
                                       enable_topic_path: bool = True,
                                       enable_keyword_path: bool = True) -> List[Dict]:
        """
        Intent-driven graph navigation retrieval with time-aware scoring.
        返回结构化的检索结果列表，而不是格式化字符串。
        
        与 retrieve_topic_aware 相同的检索逻辑，但返回结构化数据以支持去重和后续处理。
        
        Retrieval Flow:
        Step A: Analyze query intent with LLM and compute embeddings
        Step B: Dual-path graph recall via the typed graph store (with expanded k for time-aware queries)
            - Path 1 (Topic): Search topic_vector_index → find Notes
            - Path 2 (Keyword): Search keyword_vector_index → find Notes
        Step C: Fusion and ranking
        Step C.5: Apply time-aware weighting (if time_sorting != 'none')
        Step D: Logical expansion via RELATED_TO relationships
        Step E: Sort results by score or time
        
        Args:
            query: User query string
            k_final: Final number of anchor notes to return (default 10)
            k_topics: Number of topics to retrieve (default 3)
            k_keywords: Number of keywords to retrieve (default 5)
            enable_logical_expansion: Whether to expand via RELATED_TO (default True)
            sort_by: Sort order for final results (default 'score')
                - 'score': Sort by score descending (highest first)
                - 'time': Sort by timestamp descending (newest first)
                - 'time_asc': Sort by timestamp ascending (oldest first)
            
        Returns:
            List of dictionaries, each containing:
            - id: note ID
            - content: note content
            - score: final retrieval score
            - timestamp: timestamp string
            - parsed_time: parsed datetime object (for sorting)
            - source_label: source type ('Retrieval' or 'Neighbor')
            - sources: list of source descriptions
            - source_id: external source ID for evaluation
        """
        print(f"\n[Retrieve Structured v8] Query: {query[:60]}...")
        
        # Step A: Analyze & Vectorize
        intent = self._analyze_query_intent(query)
        topic_desc = intent['topic_desc']
        keywords = intent['keywords']
        time_sorting = intent.get('time_sorting', 'none')
        self.logger.info(f"Topic: {topic_desc}, Keywords: {keywords}, Time Sorting: {time_sorting}")
        
        print(f"  [2] Computing embeddings...")
        try:
            vec_topic = self.get_embedding(topic_desc)
            keywords_str = ' '.join(keywords) if isinstance(keywords, list) else keywords
            vec_keyword = self.get_embedding(keywords_str)
            # print(f"      ✓ Topic embedding: dim={len(vec_topic)}")
            # print(f"      ✓ Keyword embedding: dim={len(vec_keyword)}")
        except Exception as e:
            # print(f"      ✗ Error computing embeddings: {e}")
            return []
        
        # Step B: Dual-path graph recall
        k_topics_actual = k_topics * 2 if time_sorting != 'none' else k_topics
        k_keywords_actual = k_keywords * 2 if time_sorting != 'none' else k_keywords
        
        if time_sorting != 'none':
            # print(f"      (Time-aware mode: expanding retrieval k from {k_topics}/{k_keywords} to {k_topics_actual}/{k_keywords_actual})")
            self.logger.info(f"Time-aware mode: expanding retrieval k from {k_topics}/{k_keywords} to {k_topics_actual}/{k_keywords_actual}")
        notes_scores = {}  # note_id -> {note, topic_score, keyword_score, sources}
        
        # Log ablation state
        self.logger.info(f"[Ablation] enable_topic_path={enable_topic_path}, enable_keyword_path={enable_keyword_path}")
        
        # Path 1: Topic-based recall (skip if enable_topic_path=False)
        if enable_topic_path:
            # print(f"  [3] Path 1: Topic-based recall (top-{k_topics_actual} topics)...")
            self.logger.info(f"Path 1: Topic-based recall (top-{k_topics_actual} topics, version={self.topic_retrieval_version})...")
            topic_notes = self._retrieve_via_topic_vector_dispatch(vec_topic, keywords, k_topics_actual)
            # print(f"      ✓ Found {len(topic_notes)} notes via topics")
            self.logger.info(f"Found {len(topic_notes)} notes via topics")
            
            for note_id, score, source_info in topic_notes:
                if note_id not in notes_scores:
                    notes_scores[note_id] = {
                        'note': self._get_note(note_id),
                        'topic_score': 0.0,
                        'keyword_score': 0.0,
                        'sources': []
                    }
                notes_scores[note_id]['topic_score'] += score
                notes_scores[note_id]['sources'].append(f"[Topic] {source_info}")
        else:
            self.logger.info(f"Path 1: Topic-based recall [SKIPPED - Ablation: no_topic]")
        
        # Path 2: Keyword-based recall (skip if enable_keyword_path=False)
        if enable_keyword_path:
            # print(f"  [4] Path 2: Keyword-based recall (top-{k_keywords_actual} keywords)...")
            self.logger.info(f"Path 2: Keyword-based recall (top-{k_keywords_actual} keywords)...")
            keyword_notes = self._retrieve_via_keyword_vector(vec_keyword, k_keywords_actual)
            # print(f"      ✓ Found {len(keyword_notes)} notes via keywords")
            self.logger.info(f"Found {len(keyword_notes)} notes via keywords")
            
            for note_id, score, source_info in keyword_notes:
                if note_id not in notes_scores:
                    notes_scores[note_id] = {
                        'note': self._get_note(note_id),
                        'topic_score': 0.0,
                        'keyword_score': 0.0,
                        'sources': []
                    }
                notes_scores[note_id]['keyword_score'] += score
                notes_scores[note_id]['sources'].append(f"[Keyword] {source_info}")
        else:
            self.logger.info(f"Path 2: Keyword-based recall [SKIPPED - Ablation: no_kws]")
        
        # Filter invalid notes
        notes_scores = {
            nid: data for nid, data in notes_scores.items() 
            if data['note'] is not None
        }
        
        # print(f"  [5] Fusion: {len(notes_scores)} unique notes")
        self.logger.info(f"Fusion: {len(notes_scores)} unique notes")
        
        # Step C: Fusion ranking
        for note_id, data in notes_scores.items():
            data['final_score'] = data['topic_score'] * 0.4 + data['keyword_score'] * 0.6
        
        # Step C.5: Apply time-aware weighting (if time_sorting != 'none')
        # if time_sorting != 'none':
        #     print(f"  [5.5] Applying time-aware weighting (mode: {time_sorting})...")
        #     notes_scores = self._apply_time_weight(notes_scores, time_sorting)
        
        # Sort and select top-k_final as anchor notes
        sorted_notes = sorted(
            notes_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )[:k_final]
        
        # print(f" ✓ Selected top-{len(sorted_notes)} anchor notes")
        self.logger.info(f"Selected top-{len(sorted_notes)} anchor notes")
        
        # Step D: Logical expansion via RELATED_TO
        expanded_notes = {}
        
        # Add anchor notes
        for note_id, data in sorted_notes:
            expanded_notes[note_id] = {
                'note': data['note'],
                'source_type': 'anchor',
                'score': data['final_score'],
                'sources': data['sources']
            }
        
        if enable_logical_expansion:
            # print(f"  [6] Logical expansion via RELATED_TO relationships...")
            self.logger.info(f"Logical expansion via RELATED_TO relationships...")
            
            for note_id, data in sorted_notes:
                neighbors = self._get_related_neighbors(note_id)

                for nb in neighbors:
                    neighbor_id = nb['id']
                    edge_weight = nb['weight']
                    rel_type = nb['type']
                    if neighbor_id not in expanded_notes:
                        neighbor_note = self._get_note(neighbor_id)
                        if neighbor_note:
                            # === 核心修改：基于梯度的分数传播 ===
                            # 基础传播：邻居分数 = 锚点分数 * 边权重
                            # 这符合 "Semantic Message Passing" 的物理定义
                            propagated_score = data['final_score'] * edge_weight
                            
                            # === 核心修改：关系感知 ===
                            if rel_type == 'CONFLICTS':
                                # 冲突的节点也是相关的（为了回答"有什么不同"），但标记清楚
                                source_label = f"Neighbor (Contrast) of {note_id[:8]}"
                                # 可选：给冲突节点一点额外降权，防止它抢了正例的风头，或者保持原样因为它很重要
                                propagated_score *= 0.9 
                            elif rel_type == 'SUPPORTS':
                                source_label = f"Neighbor (Support) of {note_id[:8]}"
                            else:
                                source_label = f"Neighbor of {note_id[:8]}"
                            
                            expanded_notes[neighbor_id] = {
                                'note': neighbor_note,
                                'source_type': 'logical_neighbor',
                                'score': propagated_score, 
                                'sources': [source_label]
                            }
            
            # print(f" ✓ Expanded to {len(expanded_notes)} total notes")
            self.logger.info(f"Expanded to {len(expanded_notes)} total notes")
        else:
            self.logger.info(f"Logical expansion [SKIPPED - Ablation: no_neighbor]")
        
        # Step E: Convert to structured list
        structured_results = []
        
        for note_id, data in expanded_notes.items():
            note = data.get('note')
            if note is None:
                continue
            
            source_type = data.get('source_type', 'unknown')
            score = data.get('score', 0.0)
            sources = data.get('sources', [])
            
            timestamp = getattr(note, 'timestamp', '')
            parsed_time = self._parse_timestamp(timestamp)
            source_id = getattr(note, 'source_id', None)
            
            source_label = 'Retrieval' if source_type == 'anchor' else 'Neighbor'
            
            structured_results.append({
                'id': note_id,
                'content': note.content,
                'context': note.context,
                'score': score,
                'timestamp': timestamp,
                'parsed_time': parsed_time,
                'source_label': source_label,
                'sources': sources,
                'source_id': source_id,
                'keywords': note.keywords
            })
        
        # Step F: Sort results based on sort_by parameter
        if sort_by == 'score':
            # Sort by score descending (highest first)
            structured_results.sort(key=lambda x: x['score'], reverse=True)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by score DESC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by score DESC)")
        elif sort_by == 'time':
            # Sort by time descending (newest first)
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=True)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by time DESC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by time DESC)")
        elif sort_by == 'time_asc':
            # Sort by time ascending (oldest first)
            structured_results.sort(key=lambda x: x['parsed_time'], reverse=False)
            # print(f"  [7] Structured results: {len(structured_results)} items (sorted by time ASC)")
            self.logger.info(f"Structured results: {len(structured_results)} items (sorted by time ASC)")
        else:
            # print(f"  [7] Structured results: {len(structured_results)} items (unsorted)")
            self.logger.info(f"Structured results: {len(structured_results)} items (unsorted)")
        
        return structured_results

    def _init_score_entry(self, notes_scores, note_id):
        """Helper to init dict entry"""
        if note_id not in notes_scores:
            note_obj = self._get_note(note_id)
            # 如果从图存储没拿到对象，暂时存 None，后续过滤
            notes_scores[note_id] = {
                'note': note_obj,
                'topic_score': 0.0,
                'combined_score': 0.0,
                'split_score': 0.0,
                'sources': []
            }
    
    def _topic_note_scores(self, ranked_topics, *, hybrid=False):
        totals = defaultdict(float)
        descriptions = defaultdict(list)
        for topic_id, topic_score in ranked_topics:
            topic = self.store.get_node(topic_id)
            if topic is None:
                continue
            shared = defaultdict(set)
            for belongs in self.store.edges(topic_id, types='BELONGS_TO', direction='in'):
                keyword_id = belongs['source']
                for has_keyword in self.store.edges(keyword_id, types='HAS_KEYWORD', direction='in'):
                    shared[has_keyword['source']].add(keyword_id)
            for note_id, keywords in shared.items():
                totals[note_id] += topic_score * len(keywords)
                description = topic['properties'].get('description') or 'Unknown Topic'
                if description not in descriptions[note_id]:
                    descriptions[note_id].append(description)
        results = []
        for note_id, score in totals.items():
            label = '; '.join(descriptions[note_id][:2])
            if hybrid:
                label = 'Topic(Hybrid): ' + label
            results.append((note_id, score, label))
        return sorted(results, key=lambda row: (-row[1], row[0]))

    def _retrieve_via_topic_vector(self, topic_embedding: List[float],
                                   k: int) -> List[Tuple[str, float, str]]:
        """Top-k topics, weighted by the number of linked keywords per note."""
        if k <= 0:
            return []
        return self._topic_note_scores(self.store.search('Topic', topic_embedding, k))

    def _retrieve_via_topic_vector_v2(self, topic_embedding: List[float],
                                      query_keywords: List[str], k: int
                                      ) -> List[Tuple[str, float, str]]:
        """Triangular topic ranking: vector relevance plus symbolic overlap."""
        if k <= 0:
            return []
        ranked = []
        keywords = set(query_keywords or [])
        for topic_id, score in self.store.search('Topic', topic_embedding, max(k * 3, 10)):
            topic_words = {
                self.store.get_node(edge['source'])['properties'].get('name')
                for edge in self.store.edges(topic_id, types='BELONGS_TO', direction='in')}
            overlap = len(topic_words & keywords)
            hybrid_score = score * (1.0 + overlap / len(keywords)) if keywords else score
            ranked.append((topic_id, hybrid_score))
        ranked.sort(key=lambda row: (-row[1], row[0]))
        return self._topic_note_scores(ranked[:k], hybrid=True)

    def _retrieve_via_topic_vector_dispatch(self, 
                                            topic_embedding: List[float],
                                            query_keywords: List[str],
                                            k: int) -> List[Tuple[str, float, str]]:
        """
        Topic 向量检索的 dispatch 方法，根据配置选择 v1 或 v2 版本。
        
        Args:
            topic_embedding: Topic 描述的 embedding 向量
            query_keywords: 查询中提取的关键词列表（v2 版本需要）
            k: 检索的 Topic 数量
            
        Returns:
            List[(note_id, score, source_info)]
        """
        if self.topic_retrieval_version == 'v2':
            return self._retrieve_via_topic_vector_v2(
                topic_embedding=topic_embedding,
                query_keywords=query_keywords,
                k=k
            )
        else:
            # 默认使用 v1
            return self._retrieve_via_topic_vector(
                topic_embedding=topic_embedding,
                k=k
            )

    def _retrieve_via_keyword_vector(self, keyword_embedding: List[float],
                                     k: int) -> List[Tuple[str, float, str]]:
        """Top-k keywords expanded through incoming HAS_KEYWORD edges."""
        if k <= 0:
            return []
        totals = defaultdict(float)
        names = defaultdict(list)
        for keyword_id, score in self.store.search('Keyword', keyword_embedding, k):
            keyword = self.store.get_node(keyword_id)['properties'].get('name', '')
            for edge in self.store.edges(keyword_id, types='HAS_KEYWORD', direction='in'):
                note_id = edge['source']
                totals[note_id] += score
                names[note_id].append(keyword)
        return [(note_id, score, ', '.join(names[note_id][:3]))
                for note_id, score in sorted(totals.items(), key=lambda row: (-row[1], row[0]))]

    def _get_note(self, note_id: str) -> Optional[MemoryNote]:
        note = self.memories.get(note_id)
        if note is not None:
            return note
        node = self.store.get_node(note_id)
        if node is None or node['label'] != 'MemoryNote':
            return None
        props = node['properties']
        note = MemoryNote(content=props.get('content', ''), id=note_id,
                          keywords=props.get('keywords', []),
                          context=props.get('context', ''),
                          timestamp=props.get('timestamp', ''),
                          importance_score=props.get('importance_score', 1.0),
                          retrieval_count=props.get('retrieval_count', 0))
        note.source_id = props.get('source_id')
        self.memories[note_id] = note
        return note

    def _get_related_neighbors(self, note_id: str) -> List[Dict]:
        """Top ten outgoing semantic neighbors, weighted as in the old query."""
        neighbors = []
        for edge in self.store.edges(note_id, types={'RELATED_TO', 'SUPPORTS', 'CONFLICTS'}):
            target = self.store.get_node(edge['target'])
            if target is not None and target['label'] == 'MemoryNote':
                neighbors.append({'id': edge['target'], 'type': edge['type'],
                                  'weight': edge['properties'].get('weight', 0.5)})
        return sorted(neighbors, key=lambda row: (-row['weight'], row['id']))[:10]

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        解析时间戳，支持多种格式。
        
        Args:
            timestamp_str: 时间戳字符串
            
        Returns:
            datetime 对象，如果解析失败返回 datetime.min
        """
        if not timestamp_str:
            return datetime.min
        
        # 定义支持的时间格式
        formats = [
            "%I:%M %p on %d %B, %Y",  # 系统默认格式: "10:30 AM on 15 December, 2025"
            "%d %B, %Y",               # LoComo 格式: "3 July, 2023"
            "%Y-%m-%d %H:%M:%S",       # 标准格式: "2025-12-19 10:30:00"
            "%Y-%m-%d",                # 简单日期: "2025-12-19"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # 所有格式都失败，返回最小时间
        return datetime.min
    
    def _format_intent_driven_context(self, notes_data: Dict[str, Dict]) -> str:
        """
        Format intent-driven retrieval results with clear source attribution.
        按时间排序，显式标注分数和来源。
        
        Args:
            notes_data: {note_id: {note, source_type, score, sources}}
            
        Returns:
            Formatted context string
        """
        from datetime import datetime
        
        # 收集所有笔记条目
        all_entries = []
        
        for note_id, data in notes_data.items():
            note = data.get('note')
            if note is None:
                continue
            
            source_type = data.get('source_type', 'unknown')
            score = data.get('score', 0.0)
            sources = data.get('sources', [])
            
            # 解析时间戳
            timestamp = getattr(note, 'timestamp', '')
            parsed_time = self._parse_timestamp(timestamp)
            
            # 确定来源标签
            if source_type == 'anchor':
                source_label = 'Retrieval'
            elif source_type == 'logical_neighbor':
                source_label = 'Neighbor'
            else:
                source_label = 'Unknown'
            
            all_entries.append({
                'note': note,
                'score': score,
                'source_label': source_label,
                'parsed_time': parsed_time,
                'timestamp': timestamp,
                'sources': sources
            })
        
        # 检查是否所有时间都解析失败
        valid_times = [e for e in all_entries if e['parsed_time'] != datetime.min]
        
        if len(valid_times) == 0 and len(all_entries) > 0:
            # 所有时间都无法解析，回退到按分数排序
            all_entries.sort(key=lambda x: x['score'], reverse=True)
            self.logger.warning("[Format] All timestamps failed to parse, falling back to score-based sorting")
        else:
            # 按时间排序（从旧到新）
            all_entries.sort(key=lambda x: x['parsed_time'])
        
        # 格式化输出
        lines = []
        for entry in all_entries:
            note = entry['note']
            score = entry['score']
            source_label = entry['source_label']
            timestamp = entry['timestamp']
            
            # 格式: (Score: {score:.2f}) [{source_label}] [Time: {timestamp}] {content}
            # line = f"(Score: {score:.2f}) [{source_label}] [Time: {timestamp}] {note.content}"
            line = f"(Score: {score:.2f}) [Time: {timestamp}] [Content]: {note.content} [Context]: {note.context} [Keywords]: {note.keywords}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _get_semantic_neighbors(self, note_id: str) -> List[str]:
        """Five notes linked by RELATED_TO in either direction."""
        neighbors = set()
        for edge in self.store.edges(note_id, types='RELATED_TO', direction='both'):
            other = edge['target'] if edge['source'] == note_id else edge['source']
            node = self.store.get_node(other)
            if node is not None and node['label'] == 'MemoryNote':
                neighbors.add(other)
        return sorted(neighbors)[:5]

    def set_maintenance_threshold(self, threshold: int):
        """
        设置 Topic 层自动维护阈值。
        
        Args:
            threshold: 新的阈值（0 表示禁用自动维护）
        """
        self.maintenance_threshold = threshold
        if threshold == 0:
            print(f"[Maintenance] Auto-maintenance disabled")
        else:
            print(f"[Maintenance] Threshold updated to {threshold}")
    
    def reset_maintenance_counter(self):
        """重置维护计数器（通常在手动调用 build_topic_layer 后使用）"""
        self.maintenance_counter = 0
    
    def force_topic_refresh(self, min_keywords_per_topic: int = 3, max_keywords_for_summary: int = 10):
        """
        强制刷新 Topic 层（手动触发，不依赖计数器）。
        
        这是 build_topic_layer 的别名，提供更明确的语义。
        通常在批量导入完成后手动调用。
        
        Args:
            min_keywords_per_topic: 每个 Topic 至少包含的 Keyword 数量
            max_keywords_for_summary: 生成 Topic 描述时使用的最大 Keyword 数量
        """
        print("\n[Force Refresh] Manually triggering topic layer rebuild...")
        self.build_topic_layer(min_keywords_per_topic, max_keywords_for_summary)
        self.reset_maintenance_counter()
        print("[Force Refresh] ✓ Complete")
    
    def build_topic_layer(self, min_keywords_per_topic: int = 3, max_keywords_for_summary: int = 10):
        """
        构建 Topic 层：从 Keyword 聚类生成 Topic 节点。
        
        实现步骤：
        1. 从存储获取 Keyword 图（节点和 CO_OCCURS 关系）
        2. 使用 Leiden 算法进行社群检测
        3. 为每个社群生成 Topic 描述和 embedding
        4. 写回内存图（Topic 节点和 BELONGS_TO 关系）
        
        Args:
            min_keywords_per_topic: 每个 Topic 至少包含的 Keyword 数量；
                确定性容量后处理会重分配较小社群与未分配关键词
            max_keywords_for_summary: 生成 Topic 描述时使用的最大 Keyword 数量
            
        Note:
            此方法可以被自动触发（通过计数器）或手动调用。
            对于批量导入场景，建议使用 force_topic_refresh() 更明确。
        """
        print("\n" + "=" * 70)
        print("Building Topic Layer from Keyword Clustering")
        print("=" * 70)
        
        # Step 1: 获取 Keyword 图数据
        keyword_graph = self._fetch_keyword_graph()
        
        if not keyword_graph.nodes():
            print("No keywords found. Skipping topic layer building.")
            return
        
        print(f"\n[1] Fetched keyword graph:")
        print(f"    - Nodes: {len(keyword_graph.nodes())}")
        print(f"    - Edges: {len(keyword_graph.edges())}")
        
        # Step 2: 执行聚类
        clusters = self._cluster_keywords(keyword_graph, min_size=min_keywords_per_topic)
        
        print(f"\n[2] Clustering completed:")
        print(f"    - Total clusters: {len(clusters)}")
        print(f"    - Valid clusters (>= {min_keywords_per_topic} keywords): {len(clusters)}")
        
        if not clusters:
            print("No valid clusters found. Skipping topic generation.")
            return
        
        # Step 3: 生成 Topic
        topics = self._generate_topics(clusters, keyword_graph, max_keywords_for_summary)
        
        print(f"\n[3] Generated {len(topics)} topics")
        
        # Step 4: 写回图存储
        self._write_topics_to_store(topics)
        
        print(f"\n[4] Topics written to graph store successfully")
        print("=" * 70)
    
    def _fetch_keyword_graph(self) -> nx.Graph:
        """Build the transient weighted NetworkX input required by Leiden."""
        graph = nx.Graph()
        for keyword in self.store.nodes('Keyword'):
            graph.add_node(keyword['properties']['name'],
                           embedding=keyword['properties'].get('embedding'))
        for edge in self.store.edges(types='CO_OCCURS'):
            source = self.store.get_node(edge['source'])
            target = self.store.get_node(edge['target'])
            if source is not None and target is not None:
                graph.add_edge(source['properties']['name'], target['properties']['name'],
                               weight=edge['properties'].get('count', 1))
        return graph

    def _cluster_keywords(self, graph: nx.Graph, min_size: int = 3) -> Dict[int, List[str]]:
        """
        使用社群检测算法对 Keyword 图进行聚类。
        
        Args:
            graph: Keyword 图（NetworkX）
            
        Returns:
            clusters: {cluster_id: [keyword1, keyword2, ...]}
        """
        clusters = {}
        

        try:
            # graspologic raises EmptyNetworkError when no edges exist. The
            # bounded postprocessor can partition these isolated keywords.
            if graph.number_of_edges() == 0:
                community_results = []
                leiden_skipped_reason = 'no_keyword_cooccurrence_edges'
            else:
                print("    Using Leiden algorithm (graspologic)...")
                community_results = hierarchical_leiden(
                    graph, max_cluster_size=15, random_seed=self.graph_seed)
                leiden_skipped_reason = None
            
            # HierarchicalClusters includes transitional parent assignments.
            # Only terminal assignments form the Topic partition.
            assigned = set()
            for item in community_results:
                if not item.is_final_cluster:
                    continue
                cluster_id = item.cluster
                node = item.node
                if node in assigned:
                    raise ValueError(f'keyword has multiple final Leiden clusters: {node}')
                assigned.add(node)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node)
            final_sizes = {str(cluster_id): len(nodes) for cluster_id, nodes in clusters.items()}
            clusters, partition_audit = bounded_keyword_partition(
                graph, clusters, min_size=min_size, max_size=15)
            for field in ('input_cluster_sizes', 'output_cluster_sizes'):
                partition_audit[field] = {str(cluster_id): size
                                          for cluster_id, size in partition_audit[field].items()}
            self.last_topic_clustering_trace = {
                'max_cluster_size': 15,
                'final_cluster_sizes': final_sizes,
                'unassigned_keywords': sorted(set(graph.nodes()) - assigned),
                'leiden_skipped_reason': leiden_skipped_reason,
                'partition_audit': partition_audit,
            }
                
        except Exception as e:
            raise RuntimeError('Leiden topic clustering failed') from e
        
        return clusters
    
    
    def _generate_topics(self, 
                        clusters: Dict[int, List[str]], 
                        keyword_graph: nx.Graph,
                        max_keywords: int = 10) -> List[Dict]:
        """
        为每个 Keyword 社群生成 Topic。
        
        Args:
            clusters: Keyword 聚类结果
            keyword_graph: Keyword 图（用于获取权重等信息）
            max_keywords: 生成描述时使用的最大 Keyword 数量
            
        Returns:
            topics: List[{
                'id': topic_id,
                'cluster_id': cluster_id,
                'keywords': [keyword1, ...],
                'description': topic_description,
                'embedding': [...]
            }]
        """
        topics = []
        
        for cluster_id, members in sorted(clusters.items(), key=lambda item: item[0]):
            keywords = sorted(set(members))
            print(f"\n    Generating Topic for Cluster {cluster_id} ({len(keywords)} keywords)...")
            
            try:
                # 1. 选择重要的 keywords（基于度中心性）
                important_keywords = self._select_important_keywords(
                    keywords, keyword_graph, max_keywords
                )
                
                # 2. 生成 Topic 描述
                description = self._generate_topic_description(important_keywords)
                
                # Paper Eq. 7: arithmetic centroid of every keyword vector.
                vectors = []
                for keyword in keywords:
                    node = self.store.get_node(self._keyword_id(keyword))
                    if node is None or node['label'] != 'Keyword':
                        raise ValueError(f'Missing keyword vector for topic: {keyword}')
                    vectors.append(node['properties']['embedding'])
                embedding = [sum(values) / len(vectors) for values in zip(*vectors)]
                
                # 4. 创建 Topic 对象
                topic = {
                    'id': str(cluster_id),
                    'cluster_id': cluster_id,
                    'keywords': keywords,
                    'important_keywords': important_keywords,
                    'description': description,
                    'embedding': embedding
                }
                
                topics.append(topic)
                
                print(f"      ✓ Topic generated:")
                print(f"        Description: {description[:80]}...")
                print(f"        Keywords: {', '.join(important_keywords[:5])}")
                print(f"        Topic: {description}")
                
            except Exception as e:
                raise RuntimeError(f'Topic generation failed for cluster {cluster_id}') from e
        
        return topics
    
    def _select_important_keywords(self, 
                                   keywords: List[str], 
                                   graph: nx.Graph, 
                                   max_count: int) -> List[str]:
        """
        从社群中选择重要的 Keywords（基于度中心性）。
        
        Args:
            keywords: 社群中的所有 keywords
            graph: Keyword 图
            max_count: 最多选择的数量
            
        Returns:
            重要的 keywords 列表（按重要性排序）
        """
        # 计算每个 keyword 在子图中的度中心性
        subgraph = graph.subgraph(keywords)
        
        if len(subgraph.nodes()) == 0:
            return keywords[:max_count]
        
        # 计算度中心性（考虑边的权重）
        keyword_scores = {}
        for kw in keywords:
            if kw in subgraph:
                # 使用加权度作为重要性分数
                degree = subgraph.degree(kw, weight='weight')
                keyword_scores[kw] = degree
            else:
                keyword_scores[kw] = 0
        
        # 按分数排序
        sorted_keywords = sorted(
            keyword_scores.items(), 
            key=lambda x: (-x[1], x[0])
        )
        
        # 返回前 max_count 个
        return [kw for kw, score in sorted_keywords[:max_count]]
    
    def _generate_topic_description(self, keywords: List[str]) -> str:
        """
        使用 LLM 为 Keyword 列表生成 Topic 描述。
        
        Args:
            keywords: Keyword 列表
            
        Returns:
            Topic 描述文本
        """
        keywords_str = ", ".join(keywords)
        
        prompt = topic_description_v0(keywords_str)
        
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, 
                response_format=None,
                temperature=0.3
            )
            
            # 清理响应
            description = response.strip()
            
            # Empty descriptions are generation failures, not valid topics.
            if len(description) < 3:
                raise ValueError('Topic description is empty or too short')
            
            return description
            
        except Exception as e:
            raise RuntimeError('Topic description generation failed') from e
    
    def _write_topics_to_store(self, topics: List[Dict]):
        """Replace the topic layer while preserving notes and keyword relations."""
        self.store.remove_nodes('Topic')
        for topic in topics:
            topic_id = 'topic:' + topic['id']
            self.store.upsert_node(topic_id, 'Topic', {
                'cluster_id': topic['cluster_id'],
                'description': topic['description'],
                'embedding': topic['embedding'],
                'keyword_count': len(topic['keywords'])})
            for keyword in topic['keywords']:
                keyword_id = self._keyword_id(keyword)
                if self.store.get_node(keyword_id) is not None:
                    self.store.upsert_edge(keyword_id, topic_id, 'BELONGS_TO')

    def retrieve_paper_structured(self, query: str, k_keywords: int = 10,
                                  k_topics: int = 3, k_final: Optional[int] = 60,
                                  sort_by: str = 'time_asc',
                                  enable_logical_expansion: bool = True,
                                  enable_topic_path: bool = True,
                                  enable_keyword_path: bool = True,
                                  return_all_candidates: bool = False,
                                  rrf_topic_weight: float = 1.0,
                                  rrf_keyword_weight: float = 1.0,
                                  neighbor_score_scale: float = 1.0,
                                  keyword_rrf_tie_policy: str = 'ordinal') -> List[Dict]:
        """Paper retrieval: two complete note channels, RRF, then one-hop expansion.

        Each intent keyword independently searches ten keyword nodes. The
        historical implementation does not specify how to combine those note
        rankings; this implementation sums their scores within the keyword
        channel. RRF uses k0=60. Neighbor scores follow the
        paper's parent RRF score times the typed edge weight (default 0.5),
        using the maximum when a neighbor is reached more than once.
        The final pool is capped after
        expansion, not before it.
        """
        if (type(k_keywords) is not int or k_keywords <= 0
                or type(k_topics) is not int or k_topics <= 0
                or (k_final is not None and (type(k_final) is not int or k_final <= 0))):
            raise ValueError('paper retrieval k values must be positive or k_final=None')
        for name, value in (('rrf_topic_weight', rrf_topic_weight),
                            ('rrf_keyword_weight', rrf_keyword_weight)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if (type(neighbor_score_scale) not in (int, float)
                or not math.isfinite(neighbor_score_scale)
                or not 0 <= neighbor_score_scale <= 1):
            raise ValueError('neighbor_score_scale must be finite in [0, 1]')
        if keyword_rrf_tie_policy not in ('ordinal', 'mean_reciprocal'):
            raise ValueError('Unsupported keyword RRF tie policy')
        intent = self._analyze_query_intent(query, allow_empty_keywords=True)
        keywords = intent.get('keywords') or []
        if isinstance(keywords, str):
            keywords = [keywords]
        tie_key = lambda note_id: (
            str(getattr(self._get_note(note_id), 'source_id', None) or note_id), note_id)
        topic_notes = []
        if enable_topic_path:
            vector = self.get_embedding(intent['topic_desc'])
            topic_notes = self._retrieve_via_topic_vector(vector, k_topics)
        keyword_scores = defaultdict(float)
        keyword_trace = []
        if enable_keyword_path:
            for keyword in keywords:
                vector = self.get_embedding(keyword)
                vector_hits = self.store.search('Keyword', vector, k_keywords)
                note_hits = self._retrieve_via_keyword_vector(vector, k_keywords)
                keyword_trace.append({
                    'keyword': keyword,
                    'keyword_nodes': [{'id': node_id, 'score': score}
                                      for node_id, score in vector_hits],
                    'notes': [{'id': note_id, 'rank': rank, 'score': score}
                              for rank, (note_id, score, _) in enumerate(note_hits, 1)]})
                for note_id, score, _ in note_hits:
                    keyword_scores[note_id] += score
        topic_ranking = [note_id for note_id, _, _ in
                         sorted(topic_notes, key=lambda row: (-row[1], tie_key(row[0])))]
        keyword_ranking = [note_id for note_id in
                           sorted(keyword_scores, key=lambda note_id:
                                  (-keyword_scores[note_id], tie_key(note_id)))]
        rrf_scores = defaultdict(float)
        for rank, note_id in enumerate(topic_ranking, 1):
            rrf_scores[note_id] += rrf_topic_weight / (60 + rank)
        keyword_tie_groups = []
        rank = 0
        while rank < len(keyword_ranking):
            end = rank + 1
            score = keyword_scores[keyword_ranking[rank]]
            while end < len(keyword_ranking) and keyword_scores[keyword_ranking[end]] == score:
                end += 1
            reciprocal = (sum(1 / (60 + position) for position in range(rank + 1, end + 1))
                          / (end - rank)) if end - rank > 1 else None
            for position in range(rank, end):
                note_id = keyword_ranking[position]
                if keyword_rrf_tie_policy == 'mean_reciprocal' and end - rank > 1:
                    rrf_scores[note_id] += rrf_keyword_weight * reciprocal
                else:
                    rrf_scores[note_id] += rrf_keyword_weight / (60 + position + 1)
            if keyword_rrf_tie_policy == 'mean_reciprocal' and end - rank > 1:
                keyword_tie_groups.append({'ids': keyword_ranking[rank:end],
                                           'ranks': [rank + 1, end],
                                           'combined_score': score,
                                           'reciprocal_contribution': reciprocal})
            rank = end
        anchors = sorted(rrf_scores, key=lambda note_id:
                         (-rrf_scores[note_id], tie_key(note_id)))
        pooled = {note_id: {'score': rrf_scores[note_id], 'source_label': 'Retrieval',
                            'sources': [channel for channel, ranking in
                                        (('Topic', topic_ranking), ('Keyword', keyword_ranking))
                                        if note_id in ranking]}
                  for note_id in anchors}
        expansion = []
        if enable_logical_expansion:
            semantic_types = {'RELATED_TO', 'SUPPORTS', 'CONFLICTS'}
            for parent_id in anchors:
                for edge in self.store.edges(parent_id, types=semantic_types, direction='out'):
                    target_id = edge['target']
                    target = self.store.get_node(target_id)
                    if target is None or target['label'] != 'MemoryNote':
                        continue
                    weight = edge['properties'].get('weight', 0.5)
                    if (type(weight) not in (int, float) or not math.isfinite(weight)
                            or not 0 <= weight <= 1):
                        raise ValueError('semantic edge weight must be finite in [0, 1]')
                    score = rrf_scores[parent_id] * weight * neighbor_score_scale
                    expansion.append({'source': parent_id, 'target': target_id,
                                      'type': edge['type'], 'weight': weight,
                                      'score': score})
                    if target_id not in pooled:
                        pooled[target_id] = {'score': score, 'source_label': 'Neighbor',
                                             'sources': [f'Logical neighbor of {parent_id[:8]}']}
                    elif score > pooled[target_id]['score']:
                        pooled[target_id]['score'] = score
                        if target_id not in rrf_scores:
                            pooled[target_id]['sources'] = [f'Logical neighbor of {parent_id[:8]}']
        ranked = sorted(pooled, key=lambda note_id: (-pooled[note_id]['score'], tie_key(note_id)))
        selected = ranked if k_final is None else ranked[:k_final]
        seen_source_ids = []
        for note_id in ranked:
            source_id = getattr(self._get_note(note_id), 'source_id', None)
            if source_id is not None:
                seen_source_ids.append(source_id)
        self.last_retrieval_trace = {
            'mode': 'paper', 'query': query, 'k_keywords_per_entity': k_keywords,
            'query_intent_attempts': deepcopy(self.last_query_intent_trace),
            'k_topics': k_topics, 'k_final': k_final, 'rrf_k0': 60,
            'keyword_queries': keyword_trace,
            'topic_note_ranking': topic_ranking,
            'keyword_note_ranking': keyword_ranking,
            'rrf_anchors': [{'id': note_id, 'score': rrf_scores[note_id]}
                            for note_id in anchors],
            'expanded_edges': expansion, 'seen_note_ids': ranked,
            'seen_source_ids': seen_source_ids, 'selected_ids': selected,
        }
        if (rrf_topic_weight != 1.0 or rrf_keyword_weight != 1.0
                or neighbor_score_scale != 1.0):
            self.last_retrieval_trace['retrieval_weights'] = {
                'rrf_topic_weight': rrf_topic_weight,
                'rrf_keyword_weight': rrf_keyword_weight,
                'neighbor_score_scale': neighbor_score_scale,
            }
        if keyword_rrf_tie_policy != 'ordinal':
            self.last_retrieval_trace['keyword_rrf_tie_policy'] = keyword_rrf_tie_policy
            self.last_retrieval_trace['keyword_rrf_tie_groups'] = keyword_tie_groups
        if return_all_candidates:
            self.last_retrieval_trace['returned_all_candidates'] = True
        results = []
        for note_id in (ranked if return_all_candidates else selected):
            note = self._get_note(note_id)
            if note is None:
                continue
            timestamp = getattr(note, 'timestamp', '')
            results.append({'id': note_id, 'content': note.content,
                            'context': note.context, 'score': pooled[note_id]['score'],
                            'timestamp': timestamp,
                            'parsed_time': self._parse_timestamp(timestamp),
                            'source_label': pooled[note_id]['source_label'],
                            'sources': pooled[note_id]['sources'],
                            'source_id': getattr(note, 'source_id', None),
                            'keywords': note.keywords})
        if sort_by == 'time':
            results.sort(key=lambda row: (str(row['source_id'] or row['id']), row['id']))
            results.sort(key=lambda row: row['parsed_time'], reverse=True)
        elif sort_by == 'time_asc':
            results.sort(key=lambda row: (row['parsed_time'],
                                          str(row['source_id'] or row['id']), row['id']))
        elif sort_by == 'score':
            pass  # Already sorted by score with stable source_id/id ties.
        else:
            raise ValueError('sort_by must be score, time, or time_asc')
        return results

    def snapshot_state(self) -> Dict:
        """Return independent, plain-data graph and memory state for a rollout."""
        fields = ('content', 'id', 'keywords', 'links', 'importance_score',
                  'retrieval_count', 'timestamp', 'last_accessed', 'context',
                  'evolution_history', 'category', 'tags', 'source_id')
        return {
            'schema_version': 1,
            'memory_revision': self.MEMORY_REVISION,
            'graph': self.store.snapshot(),
            'memories': [{field: deepcopy(getattr(note, field, None)) for field in fields}
                         for _, note in sorted(self.memories.items())],
            'maintenance_counter': self.maintenance_counter,
            'maintenance_threshold': self.maintenance_threshold,
            'graph_seed': self.graph_seed,
            'topic_clustering_trace': deepcopy(self.last_topic_clustering_trace),
        }

    def save_graph(self, path) -> None:
        """Atomically save the typed graph and all mutable memory state as JSON."""
        snapshot = self.snapshot_state()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent,
                                             prefix=f'.{destination.name}.', suffix='.tmp',
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_graph(self, path) -> None:
        """Read a graph snapshot without contacting a model or database."""
        with Path(path).open(encoding='utf-8') as stream:
            snapshot = json.load(stream)
        self.restore_state(snapshot)

    def restore_state(self, snapshot: Dict) -> None:
        """Validate a snapshot fully, then replace only this instance's data state."""
        if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1:
            raise ValueError('unsupported memory graph snapshot')
        if snapshot.get('memory_revision') != self.MEMORY_REVISION:
            raise ValueError('memory graph revision differs from current construction semantics')
        store = InMemoryGraphStore.from_snapshot(snapshot.get('graph'))
        raw_memories = snapshot.get('memories')
        counter = snapshot.get('maintenance_counter')
        threshold = snapshot.get('maintenance_threshold')
        graph_seed = snapshot.get('graph_seed')
        topic_trace = snapshot.get('topic_clustering_trace')
        if (not isinstance(raw_memories, list) or type(counter) is not int or counter < 0
                or type(threshold) is not int or threshold < 0
                or (graph_seed is not None and (type(graph_seed) is not int or graph_seed < 0))
                or (topic_trace is not None and not isinstance(topic_trace, dict))):
            raise ValueError('invalid memory graph snapshot metadata')
        if self.graph_seed is not None and graph_seed != self.graph_seed:
            raise ValueError('saved graph_seed differs from configured seed')
        if topic_trace is not None:
            audit = topic_trace.get('partition_audit')
            strategy = 'deterministic_affinity_cardinality_postprocess_v1'
            if (not isinstance(audit, dict) or audit.get('strategy') != strategy
                    or audit.get('pure_leiden_output') is not False
                    or type(audit.get('total_keywords')) is not int
                    or type(audit.get('min_size')) is not int
                    or audit.get('max_size') != 15
                    or type(audit.get('feasible')) is not bool
                    or not isinstance(audit.get('input_cluster_sizes'), dict)
                    or not isinstance(audit.get('output_cluster_sizes'), dict)):
                raise ValueError('invalid saved topic partition audit')
            output_sizes = audit['output_cluster_sizes']
            if (audit['total_keywords'] < 0 or audit['min_size'] < 1
                    or any(not isinstance(key, str) or not key.lstrip('-').isdigit()
                    or type(size) is not int for key, size in output_sizes.items())
                    or audit['total_keywords'] > len(store.nodes('Keyword'))):
                raise ValueError('invalid saved topic partition audit')
            if audit['feasible']:
                if (audit.get('covered_keywords') != audit['total_keywords']
                        or sum(output_sizes.values()) != audit['total_keywords']
                        or any(not audit['min_size'] <= size <= 15
                               for size in output_sizes.values())):
                    raise ValueError('invalid saved topic partition audit')
                topics = {str(node['properties'].get('cluster_id')):
                          node['properties'].get('keyword_count')
                          for node in store.nodes('Topic')}
                if topics != output_sizes:
                    raise ValueError('saved topic partition differs from graph topics')
                belonging = store.edges(types='BELONGS_TO')
                if (len(belonging) != audit['total_keywords']
                        or len({edge['source'] for edge in belonging}) != len(belonging)):
                    raise ValueError('saved topic partition differs from graph edges')
            elif output_sizes or audit.get('covered_keywords') != 0:
                raise ValueError('invalid saved topic partition audit')
        memories = {}
        for data in raw_memories:
            if not isinstance(data, dict) or not isinstance(data.get('id'), str):
                raise ValueError('invalid saved memory')
            data = deepcopy(data)
            note_id = data['id']
            if note_id in memories:
                raise ValueError('duplicate saved memory')
            note = MemoryNote(**{key: data.get(key) for key in (
                'content', 'id', 'keywords', 'links', 'importance_score',
                'retrieval_count', 'timestamp', 'last_accessed', 'context',
                'evolution_history', 'category', 'tags')})
            # The constructor substitutes defaults for false-y values.
            for field in ('keywords', 'links', 'importance_score', 'retrieval_count',
                          'timestamp', 'last_accessed', 'context', 'evolution_history',
                          'category', 'tags'):
                setattr(note, field, data.get(field))
            note.source_id = data.get('source_id')
            node = store.get_node(note_id)
            if node is None or node['label'] != 'MemoryNote':
                raise ValueError('saved memory has no graph node')
            shared_fields = ('content', 'context', 'timestamp', 'keywords',
                             'importance_score', 'retrieval_count', 'source_id')
            if any(node['properties'].get(field) != getattr(note, field)
                   for field in shared_fields):
                raise ValueError('saved memory differs from graph node')
            memories[note_id] = note
        graph_ids = {node['id'] for node in store.nodes('MemoryNote')}
        if set(memories) != graph_ids:
            raise ValueError('graph MemoryNote set differs from saved memories')
        self.store = store
        self.memories = memories
        self.maintenance_counter = counter
        self.maintenance_threshold = threshold
        self.graph_seed = graph_seed
        self.last_topic_clustering_trace = deepcopy(topic_trace)

    def close(self):
        """In-process graph storage needs no external connection cleanup."""
        return None
