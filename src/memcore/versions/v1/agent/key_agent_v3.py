"""
Keyword-based ToG Agent: MACER (Multi-Agent Collaborative Evolution Reasoning)

这个模块实现了基于 Keyword Graph 的 MACER 算法，专注于通过关键词话题桥梁进行多跳推理：
- Responser: 基于当前上下文生成答案
- Reflector: 评估答案充分性并识别缺失信息
- Query Evolution: 生成子查询以检索缺失信息
- Graph Evolution (Topic Walk): 通过 Keyword Graph 检索并整合新证据

与 ToG3 Agent 的区别：
- 不使用三元组提取
- 专注于 MemoryNote 和 Keyword 节点
- 使用 retrieve_topic_aware 进行话题感知检索
"""

import json
import logging
import re
import random
import copy
import math
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from memcore.versions.v1.agent.key_graph_memory_v3 import KeyGraphMemoryV3
from memcore.versions.v1.agent.memory_layer import LLMController

from memcore.versions.v1.prompts.prompt_loader import PromptLoader


# ==================== Key ToG Agent V2 ====================

class KeyToGAgentV3:
    """
    基于 Key Graph 的 ToG Agent，实现 MACER 推理循环。
    
    这个 Agent 通过 Key 话题桥梁进行多跳推理：
    
    1. Responser: 仅从提供的上下文生成答案（无幻觉）
       - 只使用检索到的信息，不使用先验知识
       - 如果上下文不足则返回 "Unknown"
    
    2. Reflector: 评估答案充分性并识别缺失信息
       - 返回: sufficient (bool), missing_info (str), confidence (float)
    
    3. Query Evolution: 生成聚焦的子查询以检索缺失信息
       - 基于 reflector 的 missing_info 和先前的推理历史
    
    4. Topic Walk: 通过 Key Graph 检索新证据并整合
       - 使用 retrieve_topic_aware 方法
       - 通过 Key 共现和话题桥梁找到相关记忆
       - 合并新证据到现有上下文
    
    MACER 循环持续迭代直到答案充分或达到最大步数。
    """
    
    def __init__(self,
                 memory_system: KeyGraphMemoryV3,
                 log_level: int = logging.INFO,
                 logger: logging.Logger = None,
                 prompt_config_path: str = None,
                 use_format_instruction: bool = False,
                 retrieval_version: str = 'paper',
                 evidence_pool_version: str = 'v2',
            sub_query_version: str = 'v2',
            responder_llm_config: Dict = None,
            render_top_fraction: float = 1.0,
            final_top_k: Optional[int] = 60,
            context_char_limit: int = 90000,
            evidence_selection_version: str = 'legacy_rrf',
            rrf_topic_weight: float = 1.0,
            rrf_keyword_weight: float = 1.0,
            neighbor_score_scale: float = 1.0,
            keyword_rrf_tie_policy: str = 'ordinal',
            round_top_k: Optional[int] = None):
        """
        初始化 Key ToG Agent V3。
        
        Args:
            memory_system: KeyGraphMemoryV3 实例
            log_level: 日志级别
            logger: 外部传入的 logger 实例，如果为 None 则创建内部 logger
            prompt_config_path: prompt配置文件路径，用于选择不同版本的prompts
            use_format_instruction: 是否在 context 前添加格式说明指引
            retrieval_version: 检索方法版本 ('v1' 或 'v2')
                - 'v1': retrieve_topic_aware_structured (拼接 keyword 向量)
                - 'v2': retrieve_topic_aware_structured_v2 (Split & Merge 策略)
            evidence_pool_version: 证据池更新版本 ('v1' 或 'v2')
                - 'v1': _update_evidence_pool (带阈值过滤)
                - 'v2': _update_evidence_pool_v2 (不过滤，保留所有 Top-K)
            sub_query_version: 子查询版本 ('v1' 或 'v2')
                - 'v1': _answer_sub_query (基于 context 生成子查询)
                - 'v2': _answer_sub_query_v2 (基于 context 和 format_instruction 生成子查询)
            responder_llm_config: Responder LLM 配置字典，用于初始化独立的 LLM 控制器
                - 如果提供，将创建独立的 LLMController 用于 Responder/Reflector/Query Evolution
                - 如果为 None，则复用 memory_system.llm_controller (legacy behavior)
            render_top_fraction: 渲染时只保留前 N% 分数的证据 (0.0-1.0)
                - 1.0: 保留全部（默认）
                - 0.33: 只保留前 1/3 分数的证据
                - 会同时筛选 evidence_pool，影响后续 recall/hitrate 计算
            round_top_k: paper 模式每轮在邻居扩展后的返回上限；None 沿用 final_top_k
            final_top_k: 跨轮累积证据池的上限；None 不限制条数
        """
        self.memory = memory_system
        
        # 初始化 Responder LLM（独立或复用）
        if responder_llm_config is not None:
            # 创建独立的 LLMController 用于 Agent 的前端任务
            self.llm = LLMController(
                backend=responder_llm_config.get('backend', 'openai'),
                model=responder_llm_config.get('model', 'gpt-4o-mini'),
                api_key=responder_llm_config.get('api_key'),
                base_url=responder_llm_config.get('base_url'),
                extra_body=responder_llm_config.get('extra_body')
            )
        else:
            # Legacy behavior: 复用 memory_system 的 LLM
            self.llm = memory_system.llm_controller
        
        # 初始化证据池（用于去重和累积检索结果）
        self.evidence_pool = {}  # Dict[str, Dict]: {note_id: {content, score, timestamp, parsed_time, source_label, sources}}
        if final_top_k is not None and (type(final_top_k) is not int or final_top_k < 1):
            raise ValueError("final_top_k must be a positive integer or None")
        self.final_top_k = final_top_k
        if round_top_k is not None:
            if type(round_top_k) is not int or round_top_k <= 0:
                raise ValueError('round_top_k must be a positive integer or None')
            if retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf':
                raise ValueError('round_top_k requires paper retrieval and legacy_rrf evidence selection')
        self.round_top_k = round_top_k
        if type(context_char_limit) is not int or context_char_limit <= 0:
            raise ValueError("context_char_limit must be a positive integer")
        self.context_char_limit = context_char_limit
        if evidence_selection_version not in ('legacy_rrf', 'query_note_cosine_v1'):
            raise ValueError('Unsupported evidence selection version')
        if evidence_selection_version == 'query_note_cosine_v1' and (
                retrieval_version != 'paper' or evidence_pool_version != 'v2'
                or final_top_k is None or render_top_fraction != 1.0):
            raise ValueError('Cosine selection requires paper retrieval, v2 pool, bounded K and full rendering')
        for name, value in (('rrf_topic_weight', rrf_topic_weight),
                            ('rrf_keyword_weight', rrf_keyword_weight)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if (type(neighbor_score_scale) not in (int, float)
                or not math.isfinite(neighbor_score_scale)
                or not 0 <= neighbor_score_scale <= 1):
            raise ValueError('neighbor_score_scale must be finite in [0, 1]')
        if (rrf_topic_weight != 1.0 or rrf_keyword_weight != 1.0
                or neighbor_score_scale != 1.0) and (
                    retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf'):
            raise ValueError('Retrieval weights require paper retrieval and legacy_rrf evidence selection')
        if keyword_rrf_tie_policy not in ('ordinal', 'mean_reciprocal'):
            raise ValueError('Unsupported keyword RRF tie policy')
        if keyword_rrf_tie_policy != 'ordinal' and (
                retrieval_version != 'paper' or evidence_selection_version != 'legacy_rrf'):
            raise ValueError('Keyword RRF tie policy requires paper retrieval and legacy_rrf evidence selection')
        self.rrf_topic_weight = rrf_topic_weight
        self.rrf_keyword_weight = rrf_keyword_weight
        self.neighbor_score_scale = neighbor_score_scale
        self.keyword_rrf_tie_policy = keyword_rrf_tie_policy
        self.evidence_selection_version = evidence_selection_version
        self._selection_archive = {}
        self._selection_scores = {}
        self._selection_query = None
        self._selection_query_record = None
        self.evidence_trace = []
        self.presented_source_ids = []
        self.partial_source_ids = []
        self._rendered_evidence = []
        
        # 初始化Prompt加载器
        self.prompt_loader = PromptLoader(prompt_config_path)
        
        # 设置日志：优先使用外部 logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("KeyToGAgentV3")
            self.logger.setLevel(log_level)
            # self.logger.propagate = False  # 防止日志传播到父 logger，避免重复输出
            
            # 创建控制台处理器（如果不存在）
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(log_level)
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        # TODO
        # self.score_threshold = 0.42
        # self.score_threshold = 0.4
        self.score_threshold = 0.2
        
        # Format instruction 配置
        self.use_format_instruction = use_format_instruction
        
        # 检索版本配置
        self.retrieval_version = retrieval_version
        
        # 证据池更新版本配置
        self.evidence_pool_version = evidence_pool_version

        self.sub_query_version = sub_query_version
        
        # 渲染时分数筛选配置
        self.render_top_fraction = render_top_fraction
        
        self.logger.info("Key ToG Agent V3 初始化成功")
        self.logger.info(f"[Dual-Model] Memory LLM: {memory_system.llm_controller.llm.model} | Responder LLM: {self.llm.llm.model}")
        self.logger.info(f"score_threshold: {self.score_threshold}")
        self.logger.info(f"use_format_instruction: {self.use_format_instruction}")
        self.logger.info(f"retrieval_version: {self.retrieval_version}")
        self.logger.info(f"evidence_pool_version: {self.evidence_pool_version}")
        self.logger.info(f"Prompt配置: {self.prompt_loader.get_config_summary()}")
        self.user_prompt = "empty"
        self.logger.info(f"sub_query_version: {self.sub_query_version}")
        self.logger.info(f"render_top_fraction: {self.render_top_fraction}")
    
    def macer_loop(self,
                   question: str,
                   max_steps: int = 3,
                   initial_k: int = 5,
                   evolution_k: int = 3,
                   max_topic_neighbors: int = 3,
                   category: int = 1,
                   ground_truth_answer: str = "",
                   temperature_c5: float = 0.0,
                   sort_by: str = 'time_asc',
                   score_threshold: float = 0.5,
                   ablation_flags: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        MACER: Multi-Agent Collaborative Evolution Reasoning Loop.
        
        基于 Key Graph 的推理循环
        
        Args:
            question: 要回答的问题
            max_steps: 最大推理迭代次数
            initial_k: 仅 legacy 检索的初轮锚点数量；paper 使用 round_top_k
            evolution_k: 仅 legacy 检索的后续轮锚点数量；paper 使用 round_top_k
            max_topic_neighbors: 仅 legacy 的 Topic 数量；paper 固定为 3
            category: 问题分类 (1-5)
                1,4: 默认短语回答
                2: 需要日期的回答
                3: 简短短语回答
                5: 对抗性问题（选择题）
            ground_truth_answer: 用于 category 5 的真实答案
            temperature_c5: category 5 使用的温度参数
            sort_by: 检索结果的排序方式 (default 'score')
                - 'score': 按分数降序排列 (最高分在前)
                - 'time': 按时间降序排列 (最新在前)
                - 'time_asc': 按时间升序排列 (最早在前)
            score_threshold: 分数阈值，过滤掉分数低于此值的检索结果 (default 0.0 表示不过滤)
            ablation_flags: 消融实验标志 (default None，表示所有组件启用)
                支持的键:
                - 'no_topic': 禁用 Topic 检索路径
                - 'no_neighbor': 禁用邻居扩展 (Logical Expansion)
                - 'no_loop': 禁用 MACER 循环，只执行单步检索-生成
                - 'no_kws': 禁用 Keyword 检索路径
            
        Returns:
            字典，包含:
            - answer: 最终答案
            - reasoning_steps: 推理步骤列表
            - total_steps: 实际步骤数
            - sufficient: 答案是否充分
            - final_context: 最终累积的上下文
            - retrieved_source_ids: 检索到的所有 source_id 列表
        """
        self.logger.info("=" * 80)
        self.logger.info(f"开始 MACER 循环，问题: {question}")
        self.logger.info("=" * 80)
        
        # ==================== Ablation Flags 处理 ====================
        # 默认值：所有组件都启用
        if ablation_flags is None:
            ablation_flags = {}
        
        # 解析消融标志
        no_topic = ablation_flags.get('no_topic', False)
        no_neighbor = ablation_flags.get('no_neighbor', False)
        no_loop = ablation_flags.get('no_loop', False)
        no_kws = ablation_flags.get('no_kws', False)
        
        # 转换为检索方法需要的布尔值
        enable_topic_path = not no_topic
        enable_keyword_path = not no_kws
        enable_logical_expansion = not no_neighbor
        
        self.logger.info(f"[Ablation] no_topic={no_topic}, no_neighbor={no_neighbor}, no_loop={no_loop}, no_kws={no_kws}")
        self.logger.info(f"[Ablation] enable_topic_path={enable_topic_path}, enable_keyword_path={enable_keyword_path}, enable_logical_expansion={enable_logical_expansion}")
        
        # ==================== Step 0: 初始化证据池和检索 ====================
        self.evidence_pool = {}
        self.retrieved_source_ids = set()  # all candidates seen during this query
        self.evidence_trace = []
        self.presented_source_ids = []
        self.partial_source_ids = []
        self._rendered_evidence = []
        self.current_sort_by = sort_by  # 保存当前排序方式
        if self.evidence_selection_version == 'query_note_cosine_v1':
            self._start_cosine_question(question)
        # self.score_threshold = self.score_threshold  # 保存分数阈值
        
        # 使用结构化检索获取初始证据（根据配置选择 v1 或 v2）
        self.logger.info(f"[Step 0] 初始检索 (Topic-Aware Structured {self.retrieval_version}, sort_by={sort_by})...")
        initial_nodes = self._retrieve_structured(
            query=question,
            **self._retrieval_limits(initial_k, max_topic_neighbors),
            sort_by=sort_by,  # 排序方式
            enable_topic_path=enable_topic_path,
            enable_keyword_path=enable_keyword_path,
            enable_logical_expansion=enable_logical_expansion
        )
        
        self.logger.info(f"[Step 0] 初始检索到 {len(initial_nodes)} 个结构化节点")
        
        # 更新证据池（根据配置选择版本）
        if self.evidence_selection_version == 'query_note_cosine_v1':
            self._ingest_cosine_candidates(initial_nodes, sort_by)
        else:
            self._update_evidence_pool_dispatch(initial_nodes)
        
        # 渲染上下文（使用指定的排序方式）
        current_context = self._render_context(sort_by=sort_by)
        current_context = current_context.replace('says :', ':')
        current_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', current_context)
        
        # 初始化历史和跟踪
        history = []
        reasoning_steps = []
        step = 0
        
        # ==================== MACER 循环 ====================
        while step < max_steps:
            step += 1
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"[Step {step}] MACER 迭代 {step}/{max_steps}")
            self.logger.info(f"{'=' * 80}")
            
            step_info = {
                "step": step,
                "timestamp": datetime.now().isoformat()
            }
            
            # ========== A. Responser: 生成答案 ==========
            self.logger.info(f"[Step {step}] Responser: 生成答案 (Category {category})...")
            
            current_answer = self._generate_answer(
                question=question,
                context=current_context,
                category=category,
                ground_truth_answer=ground_truth_answer,
                temperature_c5=temperature_c5
            )
            self.logger.info(f"[Current Answer]: {current_answer}")
            # 提取 Final Answer 之后的内容
            if "Final Answer:" in current_answer:
                # 分割字符串，取最后一部分，并去除首尾空格
                answer = current_answer.split("Final Answer:")[1].strip()
            else:
                # 兜底策略：如果模型没按格式输出，尝试直接使用返回结果
                answer = current_answer.strip()
            current_answer = answer
            
            step_info["answer"] = current_answer
            self.logger.info(f"[Step {step}] Answer: {current_answer}")
            
            # ========== Ablation: no_loop 处理 ===========
            # 如果启用了 no_loop 消融，在第一步生成答案后立即退出
            if no_loop:
                self.logger.info(f"[Ablation] no_loop=True, 跳过 Reflector 和 Query Evolution，直接返回答案")
                step_info["status"] = "ablation_no_loop"
                reasoning_steps.append(step_info)
                
                return {
                    "answer": current_answer,
                    "reasoning_steps": reasoning_steps,
                    "total_steps": step,
                    "sufficient": True,  # 消融模式下标记为充分
                    "final_context": current_context,
                    "confidence": 1.0,  # 消融模式下默认置信度为 1.0
                    "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
                    "prompt": self.user_prompt,
                    **self._evidence_audit(),
                }
            
            # ========== B. Reflector: 评估充分性 ==========
            self.logger.info(f"[Step {step}] Reflector: 评估答案充分性...")
            
            reflection = self._reflect_on_answer(question, current_context, current_answer)
            self.logger.info(f"[Reflection]: {reflection}")
            
            sufficient = reflection.get("sufficient", False)
            missing_info = reflection.get("missing_info", "")
            confidence = reflection.get("confidence", 0.0)
            
            step_info["sufficient"] = sufficient
            step_info["missing_info"] = missing_info
            step_info["confidence"] = confidence
            
            self.logger.info(f"[Step {step}] Reflector: 充分={sufficient}, 置信度={confidence:.2f}")
            
            if not sufficient:
                self.logger.warning(f"[Step {step}] Reflector: 不充分！缺失: {missing_info}")
            
            # 检查是否充分
            if sufficient:
                self.logger.info(f"[Step {step}] ✓ 达到充分性！退出循环。")
                step_info["status"] = "sufficient"
                reasoning_steps.append(step_info)
                
                return {
                    "answer": current_answer,
                    "reasoning_steps": reasoning_steps,
                    "total_steps": step,
                    "sufficient": True,
                    "final_context": current_context,
                    "confidence": confidence,
                    "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
                    "prompt": self.user_prompt,
                    **self._evidence_audit(),
                }
            
            # 如果不充分且达到最大步数，仍然返回
            if step >= max_steps:
                self.logger.warning(f"[Step {step}] 达到最大步数但未充分")
                step_info["status"] = "max_steps_reached"
                reasoning_steps.append(step_info)
                break
            
            # ========== C. Query Evolution: 生成子查询 ==========
            self.logger.info(f"[Step {step}] Query Evolution: 生成子查询...")
            
            # 构建先前推理历史
            prev_reasoning = self._build_reasoning_history(history)
            self.logger.info(f"[Prev Reasoning]: {prev_reasoning}")
            
            # 生成子查询
            sub_query = self._evolve_query(
                question=question,
                missing_info=missing_info,
                prev_reasoning=prev_reasoning,
                current_context=current_context
            )
            if not sub_query:
                self.logger.warning(f"[Step {step}] Query Evolution: 无法生成子查询，退出循环")
                step_info["status"] = "no_subquery"
                reasoning_steps.append(step_info)
                break
            
            step_info["sub_query"] = sub_query
            self.logger.info(f"[Step {step}] Query Evolution: 子查询 = '{sub_query}'")
            
            # ========== D. Graph Evolution (Topic Walk): 检索新证据（结构化）==========
            self.logger.info(f"[Step {step}] Topic Walk: 通过 Key Graph 检索新证据...")
            self.logger.info(f"[Step {step}] → 使用子查询: '{sub_query}'")
            
            new_nodes = self._retrieve_structured(
                query=sub_query,
                **self._retrieval_limits(evolution_k, max_topic_neighbors),
                sort_by=sort_by,  # 保持排序一致性
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
            if self.evidence_selection_version == 'query_note_cosine_v1':
                subanswer_nodes = self._ingest_cosine_candidates(
                    new_nodes, sort_by, for_subanswer=True)
            else:
                subanswer_nodes = new_nodes
            
            self.logger.info(f"[Step {step}] Topic Walk: 检索到 {len(new_nodes)} 个新节点")
            # ========== E. Sub-QA: 回答子问题 ==========
            self.logger.info(f"[Step {step}] Sub-Answerer: 回答子问题...")
            
            # 临时渲染新节点为上下文字符串
            temp_context_lines = []
            for node in subanswer_nodes:
                # line = f"(Score: {node['score']:.2f}) [{node['source_label']}] [Time: {node['timestamp']}] {node['content']}"
                line = f"(Score: {node['score']:.2f}) [Time: {node['timestamp']}] [Content]: {node['content']} [Context]: {node['context']}"
                temp_context_lines.append(line)
            temp_context = "\n".join(temp_context_lines)
            temp_context = temp_context.replace('says :', ':')
            temp_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', temp_context)
            if self.evidence_selection_version == 'query_note_cosine_v1':
                self.evidence_trace[-1]['selection'].update(
                    subanswer_context=temp_context,
                    subanswer_context_note_count=len(subanswer_nodes),
                    subanswer_context_characters=len(temp_context))
            
            # 调用子问答器
            sub_answer = self._answer_sub_query_dispatch(sub_query, temp_context)
            if "Answer:" in sub_answer:
                sub_answer = sub_answer.replace("Answer:", "")
            self.logger.info(f"[Step {step}] Sub-answer: {sub_answer[:200]}...")
            
            step_info["sub_answer"] = sub_answer
            step_info["new_nodes_count"] = len(new_nodes)
            
            # ========== F. 更新证据池（自动去重）==========
            self.logger.info(f"[Step {step}] 更新证据池（去重）...")
            pool_size_before = len(self.evidence_pool)
            
            # 根据配置选择版本
            if self.evidence_selection_version != 'query_note_cosine_v1':
                self._update_evidence_pool_dispatch(new_nodes)
            
            pool_size_after = len(self.evidence_pool)
            self.logger.info(f"[Step {step}] 证据池: {pool_size_before} → {pool_size_after} 个唯一证据")
            
            # ========== G. 重新渲染全局上下文 ==========
            self.logger.info(f"[Step {step}] 渲染全局上下文 (sort_by={sort_by})...")
            
            old_length = len(current_context)
            current_context = self._render_context(sort_by=sort_by)
            current_context = current_context.replace('says :', ':')
            current_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', current_context)
            new_length = len(current_context)
            
            self.logger.info(f"[Step {step}] 上下文已更新: {old_length} → {new_length} 字符")
            
            # ========== H. 更新推理历史 ==========
            history.append({
                "step": step,
                "sub_query": sub_query,
                "missing_info": missing_info,
                "answer": sub_answer
            })
            
            step_info["status"] = "continued"
            reasoning_steps.append(step_info)
        
        # ==================== 最终答案 ====================
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("[Final] 使用累积上下文生成最终答案...")
        self.logger.info(f"{'=' * 80}")

        final_history_str = self._build_reasoning_history(history)
        combined_context = f"""
        {current_context}

        [Reasoning Process]{final_history_str}
        
        """
        
        final_answer = self._generate_answer(
            question=question,
            context=combined_context,
            category=category,
            ground_truth_answer=ground_truth_answer,
            temperature_c5=temperature_c5
        )
        self.logger.info(f"[Final]: {final_answer}")
        if "Final Answer:" in final_answer:
            # 分割字符串，取最后一部分，并去除首尾空格
            answer = final_answer.split("Final Answer:")[1].strip()
        else:
            # 兜底策略：如果模型没按格式输出，尝试直接使用返回结果
            answer = final_answer.strip()
        final_answer = answer
        
        self.logger.info(f"[Final] 最终答案已生成 (长度: {len(final_answer)})")
        
        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "total_steps": step,
            "sufficient": False,
            "final_context": current_context,
            "final_history_str": final_history_str,
            "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
            "prompt": self.user_prompt,
                    **self._evidence_audit(),
        }

    def macer_loop_v2(self,
                   question: str,
                   max_steps: int = 3,
                   initial_k: int = 5,
                   evolution_k: int = 3,
                   max_topic_neighbors: int = 3,
                   category: int = 1,
                   ground_truth_answer: str = "",
                   temperature_c5: float = 0.0,
                   sort_by: str = 'time_asc',
                   score_threshold: float = 0.5,
                   ablation_flags: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        MACER: Multi-Agent Collaborative Evolution Reasoning Loop.
        
        基于 Key Graph 的推理循环
        
        Args:
            question: 要回答的问题
            max_steps: 最大推理迭代次数
            initial_k: 仅 legacy 检索的初轮锚点数量；paper 使用 round_top_k
            evolution_k: 仅 legacy 检索的后续轮锚点数量；paper 使用 round_top_k
            max_topic_neighbors: 仅 legacy 的 Topic 数量；paper 固定为 3
            category: 问题分类 (1-5)
                1,4: 默认短语回答
                2: 需要日期的回答
                3: 简短短语回答
                5: 对抗性问题（选择题）
            ground_truth_answer: 用于 category 5 的真实答案
            temperature_c5: category 5 使用的温度参数
            sort_by: 检索结果的排序方式 (default 'score')
            ablation_flags: 消融实验标志 (default None，表示所有组件启用)
                支持的键:
                - 'no_topic': 禁用 Topic 检索路径
                - 'no_neighbor': 禁用邻居扩展
                - 'no_loop': 禁用循环推理（单步检索生成）
                - 'no_kws': 禁用关键词检索路径
                - 'score': 按分数降序排列 (最高分在前)
                - 'time': 按时间降序排列 (最新在前)
                - 'time_asc': 按时间升序排列 (最早在前)
            score_threshold: 分数阈值，过滤掉分数低于此值的检索结果 (default 0.0 表示不过滤)
            
        Returns:
            字典，包含:
            - answer: 最终答案
            - reasoning_steps: 推理步骤列表
            - total_steps: 实际步骤数
            - sufficient: 答案是否充分
            - final_context: 最终累积的上下文
            - retrieved_source_ids: 检索到的所有 source_id 列表
        """
        self.logger.info("=" * 80)
        self.logger.info(f"开始 MACER 循环，问题: {question}")
        self.logger.info("=" * 80)
        
        # ==================== Ablation Flags 处理 ====================
        # 默认值：所有组件都启用
        if ablation_flags is None:
            ablation_flags = {}
        
        # 解析消融标志
        no_topic = ablation_flags.get('no_topic', False)
        no_neighbor = ablation_flags.get('no_neighbor', False)
        no_loop = ablation_flags.get('no_loop', False)
        no_kws = ablation_flags.get('no_kws', False)
        
        # 转换为检索方法需要的布尔值
        enable_topic_path = not no_topic
        enable_keyword_path = not no_kws
        enable_logical_expansion = not no_neighbor
        
        # 日志记录消融设置
        if any([no_topic, no_neighbor, no_loop, no_kws]):
            self.logger.info(f"[Ablation] 消融实验模式:")
            self.logger.info(f"  - no_topic: {no_topic} (Topic检索: {'DISABLED' if no_topic else 'ENABLED'})")
            self.logger.info(f"  - no_neighbor: {no_neighbor} (邻居扩展: {'DISABLED' if no_neighbor else 'ENABLED'})")
            self.logger.info(f"  - no_loop: {no_loop} (循环推理: {'DISABLED' if no_loop else 'ENABLED'})")
            self.logger.info(f"  - no_kws: {no_kws} (关键词检索: {'DISABLED' if no_kws else 'ENABLED'})")
        
        # ==================== Step 0: 初始化证据池和检索 ====================
        self.evidence_pool = {}
        self.retrieved_source_ids = set()  # all candidates seen during this query
        self.evidence_trace = []
        self.presented_source_ids = []
        self.partial_source_ids = []
        self._rendered_evidence = []
        self.current_sort_by = sort_by  # 保存当前排序方式
        # self.score_threshold = self.score_threshold  # 保存分数阈值
        if self.evidence_selection_version == 'query_note_cosine_v1':
            self._start_cosine_question(question)
        
        # 使用结构化检索获取初始证据（根据配置选择 v1 或 v2）
        self.logger.info(f"[Step 0] 初始检索 (Topic-Aware Structured {self.retrieval_version}, sort_by={sort_by})...")
        initial_nodes = self._retrieve_structured(
            query=question,
            **self._retrieval_limits(initial_k, max_topic_neighbors),
            sort_by=sort_by,  # 排序方式
            enable_topic_path=enable_topic_path,
            enable_keyword_path=enable_keyword_path,
            enable_logical_expansion=enable_logical_expansion
        )
        
        self.logger.info(f"[Step 0] 初始检索到 {len(initial_nodes)} 个结构化节点")
        
        # 更新证据池（根据配置选择版本）
        if self.evidence_selection_version == 'query_note_cosine_v1':
            self._ingest_cosine_candidates(initial_nodes, sort_by)
        else:
            self._update_evidence_pool_dispatch(initial_nodes)
        
        # 渲染上下文（使用指定的排序方式）
        current_context = self._render_context(sort_by=sort_by)
        current_context = current_context.replace('says :', ':')
        current_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', current_context)
        
        # 初始化历史和跟踪
        history = []
        reasoning_steps = []
        step = 0
        
        # ==================== MACER 循环 ====================
        while step < max_steps:
            step += 1
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"[Step {step}] MACER 迭代 {step}/{max_steps}")
            self.logger.info(f"{'=' * 80}")
            
            step_info = {
                "step": step,
                "timestamp": datetime.now().isoformat()
            }
            
            # ========== A. Responser: 生成答案 ==========
            self.logger.info(f"[Step {step}] Responser: 生成答案 (Category {category})...")
            
            # 构建先前推理历史 (step=1时history为空)
            prev_reasoning = self._build_reasoning_history(history)
            
            # step > 1 时加入推理历史
            if history:
                answer_context = f"""
{current_context}

[Reasoning Process]
{prev_reasoning}
"""
                self.logger.info(f"[Step {step}] 加入先前推理历史")
            else:
                answer_context = current_context
            
            current_answer = self._generate_answer(
                question=question,
                context=answer_context,
                category=category,
                ground_truth_answer=ground_truth_answer,
                temperature_c5=temperature_c5
            )
            self.logger.info(f"[Current Answer]: {current_answer}")
            # 提取 Final Answer 之后的内容
            if "Final Answer:" in current_answer:
                # 分割字符串，取最后一部分，并去除首尾空格
                answer = current_answer.split("Final Answer:")[1].strip()
            else:
                # 兜底策略：如果模型没按格式输出，尝试直接使用返回结果
                answer = current_answer.strip()
            current_answer = answer
            
            step_info["answer"] = current_answer
            self.logger.info(f"[Step {step}] Answer: {current_answer}")
            
            # ========== Ablation: no_loop 检查 ==========
            # 如果启用了 no_loop 消融，在第一步生成答案后立即退出
            if no_loop and step == 1:
                self.logger.info(f"[Step {step}] [Ablation] no_loop=True, 单步检索生成模式，跳过反思和循环")
                step_info["status"] = "ablation_no_loop_exit"
                reasoning_steps.append(step_info)
                
                return {
                    "answer": current_answer,
                    "reasoning_steps": reasoning_steps,
                    "total_steps": step,
                    "sufficient": True,  # 视为充分（消融模式）
                    "final_context": current_context,
                    "confidence": 1.0,  # 消融模式不评估置信度
                    "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
                    "prompt": self.user_prompt,
                    **self._evidence_audit(),
                }
            
            # ========== B. Reflector: 评估充分性 ==========
            self.logger.info(f"[Step {step}] Reflector: 评估答案充分性...")
            
            reflection = self._reflect_on_answer(question, current_context, current_answer)
            self.logger.info(f"[Reflection]: {reflection}")
            
            sufficient = reflection.get("sufficient", False)
            missing_info = reflection.get("missing_info", "")
            confidence = reflection.get("confidence", 0.0)
            
            step_info["sufficient"] = sufficient
            step_info["missing_info"] = missing_info
            step_info["confidence"] = confidence
            
            self.logger.info(f"[Step {step}] Reflector: 充分={sufficient}, 置信度={confidence:.2f}")
            
            if not sufficient:
                self.logger.warning(f"[Step {step}] Reflector: 不充分！缺失: {missing_info}")
            
            # 检查是否充分
            if sufficient:
                self.logger.info(f"[Step {step}] ✓ 达到充分性！退出循环。")
                step_info["status"] = "sufficient"
                reasoning_steps.append(step_info)
                
                return {
                    "answer": current_answer,
                    "reasoning_steps": reasoning_steps,
                    "total_steps": step,
                    "sufficient": True,
                    "final_context": current_context,
                    "confidence": confidence,
                    "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
                    "prompt": self.user_prompt,
                    **self._evidence_audit(),
                }
            
            # 如果不充分且达到最大步数，仍然返回
            if step >= max_steps:
                self.logger.warning(f"[Step {step}] 达到最大步数但未充分")
                step_info["status"] = "max_steps_reached"
                reasoning_steps.append(step_info)
                break
            
            # ========== C. Query Evolution: 生成子查询 ==========
            self.logger.info(f"[Step {step}] Query Evolution: 生成子查询...")
            
            # 构建先前推理历史
            prev_reasoning = self._build_reasoning_history(history)
            self.logger.info(f"[Prev Reasoning]: {prev_reasoning}")
            
            # 生成子查询
            sub_query = self._evolve_query(
                question=question,
                missing_info=missing_info,
                prev_reasoning=prev_reasoning,
                current_context=current_context
            )
            if not sub_query:
                self.logger.warning(f"[Step {step}] Query Evolution: 无法生成子查询，退出循环")
                step_info["status"] = "no_subquery"
                reasoning_steps.append(step_info)
                break
            
            step_info["sub_query"] = sub_query
            self.logger.info(f"[Step {step}] Query Evolution: 子查询 = '{sub_query}'")
            
            # ========== D. Graph Evolution (Topic Walk): 检索新证据（结构化）==========
            self.logger.info(f"[Step {step}] Topic Walk: 通过 Key Graph 检索新证据...")
            self.logger.info(f"[Step {step}] → 使用子查询: '{sub_query}'")
            
            new_nodes = self._retrieve_structured(
                query=sub_query,
                **self._retrieval_limits(evolution_k, max_topic_neighbors),
                sort_by=sort_by,  # 保持排序一致性
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
            if self.evidence_selection_version == 'query_note_cosine_v1':
                subanswer_nodes = self._ingest_cosine_candidates(
                    new_nodes, sort_by, for_subanswer=True)
            else:
                subanswer_nodes = new_nodes
            
            self.logger.info(f"[Step {step}] Topic Walk: 检索到 {len(new_nodes)} 个新节点")
            # ========== E. Sub-QA: 回答子问题 ==========
            self.logger.info(f"[Step {step}] Sub-Answerer: 回答子问题...")
            
            # 临时渲染新节点为上下文字符串
            temp_context_lines = []
            for node in subanswer_nodes:
                # line = f"(Score: {node['score']:.2f}) [{node['source_label']}] [Time: {node['timestamp']}] {node['content']}"
                line = f"(Score: {node['score']:.2f}) [Time: {node['timestamp']}] [Content]: {node['content']} [Context]: {node['context']}"
                temp_context_lines.append(line)
            temp_context = "\n".join(temp_context_lines)
            temp_context = temp_context.replace('says :', ':')
            temp_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', temp_context)
            if self.evidence_selection_version == 'query_note_cosine_v1':
                self.evidence_trace[-1]['selection'].update(
                    subanswer_context=temp_context,
                    subanswer_context_note_count=len(subanswer_nodes),
                    subanswer_context_characters=len(temp_context))
            
            # 调用子问答器
            sub_answer = self._answer_sub_query_dispatch(sub_query, temp_context)
            if "Answer:" in sub_answer:
                sub_answer = sub_answer.replace("Answer:", "")
            self.logger.info(f"[Step {step}] Sub-answer: {sub_answer[:200]}...")
            
            step_info["sub_answer"] = sub_answer
            step_info["new_nodes_count"] = len(new_nodes)
            
            # ========== F. 更新证据池（自动去重）==========
            self.logger.info(f"[Step {step}] 更新证据池（去重）...")
            pool_size_before = len(self.evidence_pool)
            
            # 根据配置选择版本
            if self.evidence_selection_version != 'query_note_cosine_v1':
                self._update_evidence_pool_dispatch(new_nodes)
            
            pool_size_after = len(self.evidence_pool)
            self.logger.info(f"[Step {step}] 证据池: {pool_size_before} → {pool_size_after} 个唯一证据")
            
            # ========== G. 重新渲染全局上下文 ==========
            self.logger.info(f"[Step {step}] 渲染全局上下文 (sort_by={sort_by})...")
            
            old_length = len(current_context)
            current_context = self._render_context(sort_by=sort_by)
            current_context = current_context.replace('says :', ':')
            current_context = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', current_context)
            new_length = len(current_context)
            
            self.logger.info(f"[Step {step}] 上下文已更新: {old_length} → {new_length} 字符")
            
            # ========== H. 更新推理历史 ==========
            history.append({
                "step": step,
                "sub_query": sub_query,
                "missing_info": missing_info,
                "answer": sub_answer
            })
            
            step_info["status"] = "continued"
            reasoning_steps.append(step_info)
        
        # ==================== 最终答案 ====================
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("[Final] 使用累积上下文生成最终答案...")
        self.logger.info(f"{'=' * 80}")

        final_history_str = self._build_reasoning_history(history)
        combined_context = f"""
        {current_context}

        [Reasoning Process]{final_history_str}
        
        """
        
        final_answer = self._generate_answer(
            question=question,
            context=combined_context,
            category=category,
            ground_truth_answer=ground_truth_answer,
            temperature_c5=temperature_c5
        )
        self.logger.info(f"[Final]: {final_answer}")
        if "Final Answer:" in final_answer:
            # 分割字符串，取最后一部分，并去除首尾空格
            answer = final_answer.split("Final Answer:")[1].strip()
        else:
            # 兜底策略：如果模型没按格式输出，尝试直接使用返回结果
            answer = final_answer.strip()
        final_answer = answer
        
        self.logger.info(f"[Final] 最终答案已生成 (长度: {len(final_answer)})")
        
        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "total_steps": step,
            "sufficient": False,
            "final_context": current_context,
            "final_history_str": final_history_str,
            "retrieved_source_ids": self._get_sorted_source_ids(sort_by),
            "prompt": self.user_prompt,
                    **self._evidence_audit(),
        }
    
    def _retrieval_limits(self, legacy_anchor_k: int, legacy_topic_k: int) -> Dict:
        """Resolve the profile's round budget before calling the retriever.

        Paper retrieval caps notes after expansion; legacy variants cap anchors
        before expansion. Neither is the accumulated evidence-pool limit.
        """
        if self.retrieval_version == 'paper':
            round_k = self.round_top_k if self.round_top_k is not None else self.final_top_k
            return {'k_keywords': 10, 'k_topics': 3, 'k_final': round_k}
        return {'k_keywords': legacy_anchor_k * 2, 'k_topics': legacy_topic_k,
                'k_final': legacy_anchor_k}

    def _retrieve_structured(self, query: str, k_keywords: int, k_topics: int,
                             k_final: Optional[int], sort_by: str,
                             enable_topic_path: bool = True,
                             enable_keyword_path: bool = True,
                             enable_logical_expansion: bool = True) -> List[Dict]:
        """
        根据配置选择检索方法，接收已解析的本轮预算。
        
        Args:
            query: 查询字符串
            k_keywords: Keyword 召回数量
            k_topics: Topic 召回数量
            k_final: 本轮检索上限；paper 在邻居扩展后截断，legacy 限制扩展前锚点
            sort_by: 排序方式
            
        Returns:
            结构化检索结果列表
        """
        # 记录消融配置
        self.logger.info(f"[_retrieve_structured] enable_topic_path={enable_topic_path}, enable_keyword_path={enable_keyword_path}, enable_logical_expansion={enable_logical_expansion}")
        
        if self.retrieval_version == 'paper':
            return self.memory.retrieve_paper_structured(
                query=query, k_keywords=k_keywords, k_topics=k_topics,
                k_final=k_final, sort_by=sort_by,
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion,
                **({'rrf_topic_weight': self.rrf_topic_weight,
                    'rrf_keyword_weight': self.rrf_keyword_weight,
                    'neighbor_score_scale': self.neighbor_score_scale}
                   if (self.rrf_topic_weight != 1.0 or self.rrf_keyword_weight != 1.0
                       or self.neighbor_score_scale != 1.0) else {}),
                **({'return_all_candidates': True}
                   if getattr(self, 'evidence_selection_version', 'legacy_rrf') == 'query_note_cosine_v1'
                   else {}),
                **({'keyword_rrf_tie_policy': self.keyword_rrf_tie_policy}
                   if self.keyword_rrf_tie_policy != 'ordinal' else {}),
            )
        if self.retrieval_version == 'v8':
            return self.memory.retrieve_topic_aware_structured_v8_gating(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by,
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
        elif self.retrieval_version == 'v7':
            return self.memory.retrieve_topic_aware_structured_v7(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by,
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
        elif self.retrieval_version == 'v6':
            # v6 暂不支持消融，使用默认参数
            self.logger.warning(f"[Ablation] v6 does not support ablation flags, ignoring them")
            return self.memory.retrieve_topic_aware_structured_v6(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by
            )
        elif self.retrieval_version == 'v5':
            # v5 暂不支持消融，使用默认参数
            self.logger.warning(f"[Ablation] v5 does not support ablation flags, ignoring them")
            return self.memory.retrieve_topic_aware_structured_v5(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by
            )
        elif self.retrieval_version == 'v4':
            # v4 暂不支持消融，使用默认参数
            self.logger.warning(f"[Ablation] v4 does not support ablation flags, ignoring them")
            return self.memory.retrieve_topic_aware_structured_v4(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by
            )
        elif self.retrieval_version == 'v3':
            # v3 暂不支持消融，使用默认参数
            self.logger.warning(f"[Ablation] v3 does not support ablation flags, ignoring them")
            return self.memory.retrieve_topic_aware_structured_v3(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by
            )
        elif self.retrieval_version == 'v2':
            self.logger.warning(f"Retrieval version v2 is deprecated. Using v1 instead.")
            return self.memory.retrieve_topic_aware_structured(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by,
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
        else:
            # 默认使用 v1
            return self.memory.retrieve_topic_aware_structured(
                query=query,
                k_keywords=k_keywords,
                k_topics=k_topics,
                k_final=k_final,
                sort_by=sort_by,
                enable_topic_path=enable_topic_path,
                enable_keyword_path=enable_keyword_path,
                enable_logical_expansion=enable_logical_expansion
            )
    
    def _generate_answer(self, 
                        question: str, 
                        context: str,
                        category: int = 1,
                        ground_truth_answer: str = "",
                        temperature_c5: float = 0.0) -> str:
        """
        Responser Agent: 基于问题和上下文生成答案。
        
        根据问题分类使用不同的 prompt 策略（对齐 test_advanced.py 的实现）。
        
        Args:
            question: 问题
            context: 当前上下文
            category: 问题分类 (1-5)
                1,4: 默认短语回答
                2: 需要日期的回答
                3: 简短短语回答
                5: 对抗性问题（选择题）
            ground_truth_answer: 用于 category 5 的真实答案
            temperature_c5: category 5 使用的温度参数
            
        Returns:
            生成的答案
        """
        # 可选的格式说明指引
        if self.use_format_instruction:
            format_instruction = """
        [Context Guide]
        1. Chronological Order: The events below are strictly sorted by time (Oldest -> Newest).
        2. Metadata: Each entry starts with a relevance score `(Score: 0.xx)` and timestamp `[Time: ...]`.
        
        3. **Data Structure**:
           - **[Content]**: The original verbatim text or dialogue evidence. (PRIMARY SOURCE)
           - **[Context]**: An AI-generated interpretation that resolves pronouns (he/it) and vague references. (AUXILIARY HELP)
        
        4. **Reasoning Strategy**: 
           - **Trust [Content] First**: Use the [Content] field for direct quotes and factual verification.
           - **Use [Context] for Disambiguation**: Refer to [Context] only to understand who "he/she" refers to or what specific event is being discussed.
           - **Conflict Resolution**: If [Content] and [Context] conflict, always trust the [Content].
           - Ignore the metadata tags in your final short answer.
        """
            context = f"{format_instruction}\n{context}"
        # Record which complete evidence entries survive the actual character
        # budget, including the format instruction prepended above.
        max_context_length = self.context_char_limit
        self._record_presented_evidence(context, max_context_length)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n"
        
        # 根据 category 构建不同的 prompt
        assert category in [1, 2, 3, 4, 5], f"Invalid category: {category}"
        
        temperature = 0.7
        
        if category == 5:  # 对抗性问题 - 选择题
            # 随机排列选项顺序
            answer_tmp = []
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(ground_truth_answer)
            else:
                answer_tmp.append(ground_truth_answer)
                answer_tmp.append('Not mentioned in the conversation')

            c5_prompt_func = self.prompt_loader.get_c5_prompt()
            user_prompt = c5_prompt_func(context, question, answer_tmp)
            temperature = temperature_c5
        elif category == 2:
            c2_prompt_func = self.prompt_loader.get_c2_prompt()
            user_prompt = c2_prompt_func(context, question)
        
        elif category == 3:  # 简短短语回答
            c3_prompt_func = self.prompt_loader.get_c3_prompt()
            user_prompt = c3_prompt_func(context, question)
        
        else:  # category 1 or 4 - 默认短语回答
            c1c4_prompt_func = self.prompt_loader.get_c1c4_prompt()
            user_prompt = c1c4_prompt_func(context, question)

        self.user_prompt = user_prompt
        
        # 使用 JSON schema 返回结果（保持与 test_advanced.py 一致）
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        # self.logger.info(f"user_prompt: {user_prompt}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.llm.llm._current_caller = "responser"
                response = self.llm.llm.get_completion(
                    user_prompt,
                    response_format=response_format,
                    temperature=temperature
                )

                # 解析 JSON 响应
                result = self._parse_json(response)
                answer = result.get("answer", "Unknown")

                return answer

            except Exception as e:
                self.logger.error(f"生成答案时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Answer generation failed after three attempts") from e
    
    def _answer_sub_query(self, sub_query: str, context: str) -> str:
        """
        [新增] Sub-Answerer: 专门回答子问题，构建推理链。

        目的：解决"只检索不回答"的问题，让模型在每一步都能获得中间结论。

        Args:
            sub_query: 需要回答的子问题
            context: 检索到的上下文信息

        Returns:
            子问题的答案，如果无法回答则返回 "Unknown"
        """

        prompt = f"""Based on the provided context, answer the specific sub-question concisely.
Context:
{context}

Sub-question: {sub_query}

Answer:
"""

        try:
            self.llm.llm._current_caller = "sub_answerer"
            response = self.llm.llm.get_completion(
                prompt,
                temperature=0.5
            )
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Sub-answer generation failed: {e}")
            return "Sub-answer Failed"

    def _answer_sub_query_v2(self, sub_query: str, context: str) -> str:
        """
        [新增] Sub-Answerer: 专门回答子问题，构建推理链。
        
        目的：解决"只检索不回答"的问题，让模型在每一步都能获得中间结论。
        
        Args:
            sub_query: 需要回答的子问题
            context: 检索到的上下文信息
            
        Returns:
            子问题的答案，如果无法回答则返回 "Unknown"
        """
        format_instruction = """
        [Context Guide]
        1. Chronological Order: The events below are strictly sorted by time (Oldest -> Newest).
        2. Metadata: Each entry starts with a relevance score `(Score: 0.xx)` and timestamp `[Time: ...]`.
        
        3. **Data Structure**:
           - **[Content]**: The original verbatim text or dialogue evidence. (PRIMARY SOURCE)
           - **[Context]**: An AI-generated interpretation that resolves pronouns (he/it) and vague references. (AUXILIARY HELP)
        
        4. **Reasoning Strategy**: 
           - **Trust [Content] First**: Use the [Content] field for direct quotes and factual verification.
           - **Use [Context] for Disambiguation**: Refer to [Context] only to understand who "he/she" refers to or what specific event is being discussed.
           - **Conflict Resolution**: If [Content] and [Context] conflict, always trust the [Content].
           - Ignore the metadata tags in your final short answer.
        \n
        """


        prompt = f"""{format_instruction} Based on the provided context, answer the specific sub-question concisely.
Context:
{context}

Sub-question: {sub_query}

Answer:
"""
        
        try:
            self.llm.llm._current_caller = "sub_answerer_v2"
            response = self.llm.llm.get_completion(
                prompt,
                temperature=0.5
            )
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Sub-answer generation failed: {e}")
            return "Sub-answer Failed"

    def _answer_sub_query_dispatch(self, sub_query: str, context: str) -> str:
        if self.sub_query_version == 'v2':
            return self._answer_sub_query_v2(sub_query, context)
        else:
            return self._answer_sub_query(sub_query, context)

    def _start_cosine_question(self, question: str) -> None:
        """Embed the exact original question once; no instruction prefix."""
        self._selection_archive = {}
        self._selection_scores = {}
        vector = np.asarray(self.memory.get_embedding(question), dtype=np.float64)
        if (vector.shape != (self.memory.store.embedding_dim,)
                or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0):
            raise ValueError('Invalid original-question embedding')
        self._selection_query = vector
        usage = copy.deepcopy(self.memory.embedding_controller.last_embedding_usage)
        if (not isinstance(usage, dict) or type(usage.get('input_tokens')) is not int
                or type(usage.get('total_tokens')) is not int
                or usage['input_tokens'] < 0 or usage['total_tokens'] < usage['input_tokens']
                or usage.get('call_count') != 1):
            raise ValueError('Original-question embedding usage is missing')
        self._selection_query_record = {
            'input': question, 'model': self.memory.embedding_controller.model,
            'dimension': int(vector.size), 'vector': vector.tolist(),
            'usage': usage,
        }

    def _cosine_score(self, note_id: str) -> float:
        if note_id not in self._selection_scores:
            note = self.memory.store.get_node(note_id)
            if note is None or note.get('label') != 'MemoryNote':
                raise ValueError(f'Cosine candidate is not a graph MemoryNote: {note_id}')
            vector = np.asarray(note['properties'].get('embedding'), dtype=np.float64)
            if (vector.shape != self._selection_query.shape
                    or not np.all(np.isfinite(vector))):
                raise ValueError(f'Invalid frozen note vector: {note_id}')
            norm = np.linalg.norm(vector)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError(f'Zero frozen note vector: {note_id}')
            score = float(np.dot(vector, self._selection_query) /
                          (norm * np.linalg.norm(self._selection_query)))
            if not math.isfinite(score):
                raise ValueError(f'Invalid cosine score: {note_id}')
            self._selection_scores[note_id] = score
        return self._selection_scores[note_id]

    @staticmethod
    def _selection_order(item):
        return (-item['selection_score'], str(item.get('source_id') or item['id']),
                str(item['id']))

    def _ingest_cosine_candidates(self, new_nodes: List[Dict], sort_by: str,
                                  for_subanswer: bool = False) -> List[Dict]:
        """Keep all graph-seen candidates private; expose only selected K to LLMs."""
        if self._selection_query is None:
            raise ValueError('Cosine selection question was not initialized')
        before = set(self.evidence_pool)
        retrieval = copy.deepcopy(getattr(self.memory, 'last_retrieval_trace', {}))
        self.retrieved_source_ids.update(retrieval.get('seen_source_ids', []))
        candidate_scores = []
        for raw in new_nodes:
            node = copy.deepcopy(raw)
            note_id = node.get('id')
            if not note_id:
                raise ValueError('Graph-seen candidate lacks note ID')
            node['selection_score'] = self._cosine_score(note_id)
            candidate_scores.append({
                'id': note_id, 'source_id': node.get('source_id'),
                'selection_score': node['selection_score'],
                'retrieval_score': node.get('score'),
            })
            previous = self._selection_archive.get(note_id)
            if previous is None or node.get('score', 0.0) > previous.get('score', 0.0):
                self._selection_archive[note_id] = node
        ordered = sorted(self._selection_archive.values(), key=self._selection_order)
        selected = ordered[:self.final_top_k]
        self.evidence_pool = {node['id']: node for node in selected}
        if sort_by == 'time_asc':
            subanswer_nodes = sorted(selected, key=lambda item: (
                item['parsed_time'], str(item.get('source_id') or item['id']), item['id']))
        elif sort_by == 'time':
            subanswer_nodes = sorted(selected, key=lambda item: (
                str(item.get('source_id') or item['id']), item['id']))
            subanswer_nodes.sort(key=lambda item: item['parsed_time'], reverse=True)
        else:
            subanswer_nodes = list(selected)
        selection = {
            'version': self.evidence_selection_version,
            'candidate_scores': candidate_scores,
            'archive_ids': sorted(self._selection_archive),
            'selected_ids': [node['id'] for node in selected],
            'subanswer_ids': [node['id'] for node in subanswer_nodes] if for_subanswer else [],
            'subanswer_source_ids': [node['source_id'] for node in subanswer_nodes]
            if for_subanswer else [],
        }
        if not self.evidence_trace:
            selection['query'] = copy.deepcopy(self._selection_query_record)
        self.evidence_trace.append({
            'round': len(self.evidence_trace),
            'candidates': [{'id': node['id'], 'source_id': node.get('source_id'),
                            'score': node.get('score', 0.0)} for node in selected],
            'pool_ids': list(self.evidence_pool),
            'pool_source_ids': self._get_sorted_source_ids('score'),
            'evicted_ids': sorted((before | {node['id'] for node in new_nodes})
                                  - set(self.evidence_pool)),
            'final_top_k': self.final_top_k, 'retrieval': retrieval,
            'selection': selection,
        })
        return subanswer_nodes
    
    def _update_evidence_pool_dispatch(self, new_nodes: List[Dict]) -> None:
        """
        根据配置选择证据池更新方法。
        
        Args:
            new_nodes: 结构化节点列表
        """
        # Copy candidates so later pool updates do not mutate retrieval records.
        new_nodes = copy.deepcopy(new_nodes)
        retrieval_trace = getattr(self.memory, "last_retrieval_trace", {})
        self.retrieved_source_ids.update(retrieval_trace.get("seen_source_ids", []))
        before = set(self.evidence_pool)
        if self.evidence_pool_version == 'v2':
            self._update_evidence_pool_v2(new_nodes)
        else:
            # 默认使用 v1
            self._update_evidence_pool(new_nodes)
        self._bound_evidence_pool()
        self.evidence_trace.append({
            "round": len(self.evidence_trace),
            "candidates": [{"id": node.get("id"), "source_id": node.get("source_id"),
                            "score": node.get("score", 0.0)} for node in new_nodes],
            "pool_ids": list(self.evidence_pool),
            "pool_source_ids": self._get_sorted_source_ids("score"),
            "evicted_ids": sorted((before | {node.get("id") for node in new_nodes if node.get("id")})
                                  - set(self.evidence_pool)),
            "final_top_k": self.final_top_k,
            "retrieval": copy.deepcopy(getattr(self.memory, "last_retrieval_trace", {})),
        })

    @staticmethod
    def _evidence_order(item):
        return (-item.get("score", 0.0), str(item.get("source_id") or item.get("id", "")),
                str(item.get("id", "")))

    def _bound_evidence_pool(self):
        order = (self._selection_order if getattr(self, 'evidence_selection_version', 'legacy_rrf') == 'query_note_cosine_v1'
                 else self._evidence_order)
        ordered = sorted(self.evidence_pool.values(), key=order)
        if self.final_top_k is not None:
            ordered = ordered[:self.final_top_k]
        self.evidence_pool = {item["id"]: item for item in ordered}

    def _evidence_audit(self):
        return {"evidence_trace": copy.deepcopy(self.evidence_trace),
                "seen_source_ids": sorted(self.retrieved_source_ids),
                "presented_source_ids": list(self.presented_source_ids),
                "partial_source_ids": list(self.partial_source_ids)}

    def _record_presented_evidence(self, context, character_limit):
        normalized = context.replace('says :', ':')
        normalized = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', normalized)
        # The caller has already normalized retrieval text. The format guide
        # contains no transformed timestamps, so these offsets match truncation.
        full, partial, start = [], [], 0
        for source_id, line in self._rendered_evidence:
            line = line.replace('says :', ':')
            line = re.sub(r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)\s+on\s+', '', line)
            position = normalized.find(line, start)
            if position < 0:
                continue
            start = position + len(line)
            if source_id and position < character_limit:
                target = full if start <= character_limit else partial
                if source_id not in target:
                    target.append(source_id)
        self.presented_source_ids, self.partial_source_ids = full, partial
        if getattr(self, 'evidence_selection_version', 'legacy_rrf') == 'query_note_cosine_v1' and (
                set(full) != {node.get('source_id') for node in self.evidence_pool.values()}
                or partial):
            raise ValueError('Cosine policy answer context must present all selected notes completely')
        if self.evidence_trace:
            self.evidence_trace[-1].update(presented_source_ids=list(full),
                                          partial_source_ids=list(partial),
                                          answer_context_characters=min(len(context), character_limit),
                                          context_char_limit=character_limit)
    
    def _update_evidence_pool(self, new_nodes: List[Dict]) -> None:
        """
        [新增] 更新证据池：去重并保留最高分数的证据。
        
        目的：
        - 避免重复检索相同的记忆
        - 保留最相关的证据（按分数）
        - 支持跨步骤的证据累积
        - 收集所有检索到的 source_id 用于 Recall 评估
        - 过滤掉分数低于阈值的噪声
        
        Args:
            new_nodes: 结构化节点列表，每个节点包含:
                - id: 笔记 ID
                - content: 笔记内容
                - score: 检索分数
                - timestamp: 时间戳
                - parsed_time: 解析后的时间
                - source_label: 来源标签
                - sources: 来源列表
                - source_id: 外部来源 ID（用于评估）
        """
        filtered_count = 0
        threshold = getattr(self, 'score_threshold', 0.0)
        
        for node in new_nodes:
            note_id = node.get('id')
            if not note_id:
                continue
            
            new_score = node.get('score', 0.0)
            
            # 收集 source_id 用于 Recall 评估 (无论是否入池，只要检索到了就算 hit，或者也可以只算入池的)
            # 这里保持原逻辑，只要检索结果里有就记录
            source_id = node.get('source_id')
            if source_id:
                self.retrieved_source_ids.add(source_id)
            
            # --- 核心修改开始 ---
            
            if note_id in self.evidence_pool:
                # === 情况 A: 节点已存在 ===
                existing_node = self.evidence_pool[note_id]
                old_score = existing_node.get('score', 0.0)
                
                # 1. 合并 Sources (去重)
                old_sources = existing_node.get('sources', [])
                new_sources = node.get('sources', [])
                
                # 使用 set 去重，保持 source trace
                # 注意：确保 sources 里的元素是字符串
                combined_sources = sorted(list(set(old_sources + new_sources)))
                
                # 2. 更新分数和主体
                if new_score > old_score:
                    # 如果新分数更高，替换整个节点，但使用合并后的 sources
                    self.evidence_pool[note_id] = node
                    self.evidence_pool[note_id]['sources'] = combined_sources
                    self.logger.debug(f"  Updated evidence {note_id[:8]} (score: {old_score:.3f} -> {new_score:.3f}, sources merged)")
                else:
                    # 如果旧分数更高，保留旧节点，但更新 sources (证明这次也召回了它)
                    self.evidence_pool[note_id]['sources'] = combined_sources
                    self.logger.debug(f"  Merged evidence {note_id[:8]} (score kept: {old_score:.3f}, sources merged)")
                    
            else:
                # === 情况 B: 节点不存在 (新节点) ===
                
                # 仅对新节点应用阈值过滤
                if threshold > 0 and new_score < threshold:
                    filtered_count += 1
                    # self.logger.debug(f"  Filtered evidence {note_id[:8]} (score {new_score:.3f} < threshold)")
                    continue
                
                self.evidence_pool[note_id] = node
                self.logger.debug(f"  Added evidence {note_id[:8]} (score: {new_score:.3f})")

            # --- 核心修改结束 ---
        
        # 日志输出：包含过滤信息
        if filtered_count > 0:
            self.logger.info(f"  Evidence pool updated: {len(self.evidence_pool)} unique items, {len(self.retrieved_source_ids)} source_ids (filtered {filtered_count} below threshold {threshold})")
        else:
            self.logger.info(f"  Evidence pool updated: {len(self.evidence_pool)} unique items, {len(self.retrieved_source_ids)} source_ids")
    
    def _update_evidence_pool_v2(self, new_nodes: List[Dict]) -> None:
        """
        [新增] 更新证据池：去重并保留最高分数的证据。
        
        目的：
        - 避免重复检索相同的记忆
        - 保留最相关的证据（按分数）
        - 支持跨步骤的证据累积
        - 收集所有检索到的 source_id 用于 Recall 评估
        - 过滤掉分数低于阈值的噪声
        
        Args:
            new_nodes: 结构化节点列表，每个节点包含:
                - id: 笔记 ID
                - content: 笔记内容
                - score: 检索分数
                - timestamp: 时间戳
                - parsed_time: 解析后的时间
                - source_label: 来源标签
                - sources: 来源列表
                - source_id: 外部来源 ID（用于评估）
        """
        
        for node in new_nodes:
            note_id = node.get('id')
            if not note_id:
                continue
            
            new_score = node.get('score', 0.0)
            
            # 收集 source_id 用于评估
            source_id = node.get('source_id')
            if source_id:
                self.retrieved_source_ids.add(source_id)
            
            if note_id in self.evidence_pool:
                # === 情况 A: 节点已存在 (执行合并) ===
                existing_node = self.evidence_pool[note_id]
                old_score = existing_node.get('score', 0.0)
                
                # 1. 合并 Sources (去重)
                old_sources = existing_node.get('sources', [])
                new_sources = node.get('sources', [])
                # 确保 sources 元素是字符串且去重
                combined_sources = sorted(list(set(old_sources + new_sources)))
                
                # 2. 始终保留最高分
                # 即使这次检索分数低，也不要降低它在池子里的分数
                final_score = max(old_score, new_score)
                
                # 更新节点
                # 注意：如果新分数更高，我们可能想更新 content/context (虽然通常是一样的)
                # 这里主要更新 score 和 sources
                if new_score > old_score:
                    self.evidence_pool[note_id] = node
                
                self.evidence_pool[note_id]['score'] = final_score
                self.evidence_pool[note_id]['sources'] = combined_sources
                
                # self.logger.debug(f"  Merged evidence {note_id[:8]} (score: {final_score:.3f}, sources: {len(combined_sources)})")
                    
            else:
                # === 情况 B: 节点不存在 (直接入池) ===
                # 【关键修改】：不再检查 if new_score < threshold
                # 只要检索系统认为它相关（Top-K），我们就先留着
                
                self.evidence_pool[note_id] = node
                # self.logger.debug(f"  Added evidence {note_id[:8]} (score: {new_score:.3f})")

        self.logger.info(f"  Evidence pool updated: {len(self.evidence_pool)} unique items")


    def _get_sorted_source_ids(self, sort_by: str = 'score') -> List[str]:
        """
        根据证据池的排序方式返回有序的 source_id 列表。
        
        目的：让 retrieved_source_ids 的顺序与检索结果排序一致，
        便于评估时对比检索顺序。
        
        Args:
            sort_by: 排序方式
                - 'score': 按分数降序（从高到低）
                - 'time': 按时间降序（从新到旧）
                - 'time_asc': 按时间升序（从旧到新）
        
        Returns:
            排序后的 source_id 列表（去重，保持顺序）
        """
        if not self.evidence_pool:
            return []
        
        # 获取所有证据
        evidence_list = list(self.evidence_pool.values())
        
        # 排序
        if sort_by == 'score':
            if getattr(self, 'evidence_selection_version', 'legacy_rrf') == 'query_note_cosine_v1':
                evidence_list.sort(key=self._selection_order)
            else:
                evidence_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        elif sort_by == 'time':
            evidence_list.sort(key=lambda x: x.get('parsed_time', datetime.min), reverse=True)
        elif sort_by == 'time_asc':
            evidence_list.sort(key=lambda x: x.get('parsed_time', datetime.min), reverse=False)
        
        # 按顺序提取 source_id（去重，保持顺序）
        sorted_source_ids = []
        seen = set()
        for item in evidence_list:
            source_id = item.get('source_id')
            if source_id and source_id not in seen:
                sorted_source_ids.append(source_id)
                seen.add(source_id)
        
        return sorted_source_ids
    
    def _render_context(self, sort_by: str = 'time_asc') -> str:
        """
        [新增] 将证据池渲染为格式化的上下文字符串。
        
        Args:
            sort_by: 排序方式
                - 'score': 按分数降序（从高到低）- 与 retrieval 排序一致
                - 'time': 按时间降序（从新到旧）- 最新记忆在前
                - 'time_asc': 按时间升序（从旧到新）- 最早记忆在前
        
        Returns:
            格式化的上下文字符串
        """
        self._rendered_evidence = []
        self._bound_evidence_pool()
        if not self.evidence_pool:
            return ""
        
        # 获取所有证据
        evidence_list = list(self.evidence_pool.values())
        original_count = len(evidence_list)
        
        # === 分数筛选：只保留前 N% 分数的证据 ===
        if self.render_top_fraction < 1.0 and len(evidence_list) > 1:
            # 先按分数降序排序
            evidence_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            # 计算保留数量（至少保留 1 个）
            keep_count = max(1, int(len(evidence_list) * self.render_top_fraction))
            
            # 筛选出要保留的证据
            kept_evidence = evidence_list[:keep_count]
            removed_evidence = evidence_list[keep_count:]
            
            # 同步更新 evidence_pool（移除被过滤的证据）
            for item in removed_evidence:
                note_id = item.get('id')
                if note_id and note_id in self.evidence_pool:
                    del self.evidence_pool[note_id]
            
            evidence_list = kept_evidence
            self.logger.info(f"  Top-fraction filter: {original_count} → {len(evidence_list)} (kept top {self.render_top_fraction*100:.0f}%)")
        
        # 排序
        if sort_by == 'score':
            # 按分数排序（从高到低）
            if getattr(self, 'evidence_selection_version', 'legacy_rrf') == 'query_note_cosine_v1':
                evidence_list.sort(key=self._selection_order)
            else:
                evidence_list.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            self.logger.debug(f"  Rendering context: sorted by score (high → low)")
        elif sort_by == 'time':
            # 按时间排序（从新到旧）
            evidence_list.sort(key=lambda x: x.get('parsed_time', datetime.min), reverse=True)
            self.logger.debug(f"  Rendering context: sorted by time (newest → oldest)")
        elif sort_by == 'time_asc':
            # 按时间排序（从旧到新）
            evidence_list.sort(key=lambda x: x.get('parsed_time', datetime.min), reverse=False)
            self.logger.debug(f"  Rendering context: sorted by time_asc (oldest → newest)")
        
        # 格式化输出
        lines = []
        for item in evidence_list:
            # 格式: (Score: {score:.2f}) [{source_label}] [Time: {timestamp}] {content}
            # line = f"(Score: {item['score']:.2f}) [{item['source_label']}] [Time: {item['timestamp']}] {item['content']}"
            line = f"(Score: {item['score']:.2f}) [Time: {item['timestamp']}] [Content]: {item['content']} [Context]: {item['context']} [Keywords]: {item['keywords']}"
            lines.append(line)
            self._rendered_evidence.append((item.get('source_id'), line))
        
        context = "\n".join(lines)
        self.logger.info(f"  Rendered context: {len(lines)} items, {len(context)} chars")
        return context
    
    def _refine_subgraph(self, query: str, raw_context: str) -> str:
        """
        [可选] 对检索到的上下文进行清洗和优化。
        
        目的：
        - 移除噪声信息
        - 突出与查询相关的内容
        - 压缩上下文长度
        
        Args:
            query: 当前查询
            raw_context: 原始上下文
            
        Returns:
            清洗后的上下文
        """
        # 简单实现：移除过长的行或重复信息
        # 可以根据需要扩展为更复杂的逻辑
        
        lines = raw_context.split('\n')
        
        # 过滤：移除空行和过长的行
        max_line_length = 500
        filtered_lines = [
            line for line in lines 
            if line.strip() and len(line) <= max_line_length
        ]
        
        refined = '\n'.join(filtered_lines)
        
        if len(refined) < len(raw_context):
            self.logger.debug(f"  Context refined: {len(raw_context)} → {len(refined)} chars")
        
        return refined
    
    def _reflect_on_answer(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Reflector Agent: 评估答案是否充分。
        
        Args:
            question: 问题
            context: 当前上下文
            answer: 当前答案
            
        Returns:
            字典，包含: sufficient, missing_info, confidence
        """
        # 组合上下文和答案进行反思
        max_context_length = self.context_char_limit
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... (truncated)"
        
        full_context = f"Current Context:\n{context}\nCurrent Answer:\n{answer}"
        
        reflector_func = self.prompt_loader.get_reflector_prompt()
        prompt = reflector_func(question, full_context)
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "reflector_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sufficient": {"type": "boolean"},
                        "missing_info": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["sufficient", "missing_info", "confidence"],
                    "additionalProperties": False
                },
                "strict": False
            }
        }
        
        try:
            self.llm.llm._current_caller = "reflector"
            response = self.llm.llm.get_completion(
                prompt,
                response_format=response_format,
                temperature=0.3
            )

            result = self._parse_json(response)
            return result

        except Exception as e:
            self.logger.error(f"反思时出错: {e}")
            return {
                "sufficient": True,  # 失败保护：假设充分
                "missing_info": "",
                "confidence": 0.5
            }
    
    def _evolve_query(self, question: str, missing_info: str,
                     prev_reasoning: str, current_context: str) -> Optional[str]:
        """
        Query Evolution Agent: 基于缺失信息生成子查询。
        
        改进点：
        - 真正使用 missing_info 参数构建 Prompt
        - 保留更多上下文（首尾各 2000 字符）而不是硬截断前 5 行
        - 使用 subquery_prompt_v1 明确告知 LLM 缺失的信息
        
        Args:
            question: 原始问题
            missing_info: 缺失信息描述（来自 Reflector）
            prev_reasoning: 先前推理步骤
            current_context: 当前上下文
            
        Returns:
            子查询字符串，如果没有进一步的查询则返回 None
        """
        # 构建上下文摘要
        # context_lines = current_context.split('\n')[:5]
        # context_summary = '\n'.join(context_lines)
        
        # 获取subquery prompt函数
        subquery_func = self.prompt_loader.get_subquery_prompt()
        # 检查是v0还是v1版本，因为参数不同
        import inspect
        sig = inspect.signature(subquery_func)
        if 'missing_info' in sig.parameters:
            # v1版本
            prompt = subquery_func(
                query_str=question, 
                context_str=current_context, 
                prev_reasoning=prev_reasoning if prev_reasoning else "None", 
                missing_info=missing_info if missing_info else "Unknown - need more information"
            )
        else:
            # v0版本
            prompt = subquery_func(
                query_str=question, 
                context_str=current_context, 
                prev_reasoning=prev_reasoning if prev_reasoning else "None"
            )
        
        try:
            self.llm.llm._current_caller = "query_evolution"
            response = self.llm.llm.get_completion(
                prompt,
                response_format=None,
                temperature=0.5
            )

            sub_query = response.strip()
            # The prompt ends with "Sub-question:", which the model may repeat.
            # Repeated known labels also occur (for example, "Output: Question:").
            label_pattern = (r'^(?:sub\s*-\s*question|sub\s*-\s*query|subquery|'
                             r'next\s+question|output|question)\s*:\s*')
            while sub_query:
                stripped = re.sub(label_pattern, '', sub_query, count=1,
                                  flags=re.IGNORECASE).strip()
                if len(stripped) >= len(sub_query):
                    break
                sub_query = stripped
            if not sub_query or re.fullmatch(r'''none\.?|['"]none\.?['"]''', sub_query,
                                             flags=re.IGNORECASE):
                return None

            return sub_query
            
        except Exception as e:
            self.logger.error(f"查询演化时出错: {e}")
            return None
    
    def _build_reasoning_history(self, history: List[Dict]) -> str:
        if not history:
            return "None"
        
        reasoning_parts = []
        for item in history:
            step = item.get('step', '?')
            sub_query = item.get('sub_query', '')
            answer_preview = item.get('answer', '')
            
            reasoning_parts.append(f"- Step {step}: Sub Query: {sub_query}")
            reasoning_parts.append(f"Answer: {answer_preview}")
        
        return '\n'.join(reasoning_parts)
    
    def _parse_json(self, response_str: str) -> Dict[str, Any]:
        """
        健壮的 JSON 解析器，处理各种响应格式。
        
        处理：
        - 纯 JSON 字符串
        - Markdown 代码块中的 JSON (```json ... ```)
        - 带额外空格的 JSON
        
        Args:
            response_str: LLM 的响应字符串
            
        Returns:
            解析后的 JSON 字典
        """
        # 首先尝试直接解析
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            pass
        
        # 尝试从 markdown 代码块中提取 JSON
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_str, re.DOTALL)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试在字符串中找到 JSON 对象
        brace_start = response_str.find('{')
        brace_end = response_str.rfind('}')
        
        if brace_start != -1 and brace_end != -1:
            try:
                json_str = response_str[brace_start:brace_end + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 如果所有方法都失败，抛出错误
        raise ValueError(f"无法从响应中解析 JSON: {response_str[:100]}...")
    
    def _log_context_summary(self, context: str, prefix: str):
        """
        记录上下文摘要，展示 Keyword 信息。
        
        Args:
            context: 上下文字符串
            prefix: 日志前缀
        """
        # 提取关键信息
        lines = context.split('\n')
        
        # 统计不同类型的信息
        anchors = 0
        semantic_neighbors = 0
        topic_neighbors = 0
        keywords_found = set()
        
        for line in lines:
            if 'Direct Matches' in line:
                anchors += 1
            elif 'Semantic Neighbors' in line:
                semantic_neighbors += 1
            elif 'Topic Neighbors' in line:
                topic_neighbors += 1
            elif 'Keywords:' in line or 'keywords:' in line.lower():
                # 提取关键词
                kw_part = line.split('Keywords:')[-1].split('keywords:')[-1]
                kws = [k.strip() for k in kw_part.split(',')]
                keywords_found.update(kws[:5])  # 限制数量
        
        self.logger.info(f"{prefix} 上下文包含:")
        if anchors:
            self.logger.info(f"{prefix}   - 主要相关记忆: ~{anchors} 个")
        if semantic_neighbors:
            self.logger.info(f"{prefix}   - 语义关联记忆: ~{semantic_neighbors} 个")
        if topic_neighbors:
            self.logger.info(f"{prefix}   - 话题相关记忆: ~{topic_neighbors} 个")
        if keywords_found:
            kw_preview = list(keywords_found)[:8]
            self.logger.info(f"{prefix}   - 发现的关键词: {', '.join(kw_preview)}")


# ==================== 便捷函数 ====================

def create_key_agent(memory_system: KeyGraphMemoryV3,
                        log_level: int = logging.INFO) -> KeyToGAgentV3:
    """
    工厂函数，创建 KeyToGAgentV3。
    
    Args:
        memory_system: KeywordGraphSystem 实例
        log_level: 日志级别
        
    Returns:
        初始化的 KeyToGAgentV3
    """
    return KeyToGAgentV3(memory_system=memory_system, log_level=log_level)


# ==================== 示例用法 ====================
