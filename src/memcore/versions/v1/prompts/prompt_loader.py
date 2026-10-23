"""
Prompt版本加载器
根据配置文件动态加载不同版本的prompts
"""

import os
import yaml
from typing import Dict, Callable, Any


# 导入所有prompt版本
from memcore.versions.v1.prompts.locomo_prompts.c1c4 import (
    answer_c1_c4_v0, answer_c1_c4_v1, answer_c1_c4_v2,
    answer_c1_c4_v3, answer_c1_c4_v4, answer_c1_c4_v5, 
    answer_c1_c4_v6, answer_c1_c4_v7, answer_c1_c4_v8, 
    answer_c1_c4_v9, answer_c1_c4_v10, answer_c1_c4_v11
)
from memcore.versions.v1.prompts.locomo_prompts.c2 import (
    answer_c2_v0, answer_c2_v1, answer_c2_v2,
    answer_c2_v3, answer_c2_v4, answer_c2_v5, answer_c2_v6, answer_c2_v7, answer_c2_v8
)
from memcore.versions.v1.prompts.locomo_prompts.c3 import answer_c3_v0, answer_c3_v1, answer_c3_v2
from memcore.versions.v1.prompts.locomo_prompts.c5 import answer_c5_v0, answer_c5_v1, answer_c5_v2, answer_c5_v3, answer_c5_v4, answer_c5_v5
from memcore.versions.v1.prompts.subquery_prompt import subquery_prompt_v0, subquery_prompt_v1
from memcore.versions.v1.prompts.reflector_prompts import reflector_prompt_v0
from memcore.versions.v1.prompts.query_intent_prompts import query_intent_v0, query_intent_v1, query_intent_v2, query_intent_v3, query_intent_prompt_gpt_v0
from memcore.versions.v1.prompts.analyze_content_prompt import analyze_content_prompt_v0, analyze_content_prompt_v1, analyze_content_prompt_v2, analyze_content_prompt_v3, analyze_content_prompt_gpt_v0, analyze_content_prompt_factdense
from memcore.versions.v1.prompts.judgment_prompts import (
    judgment_prompts_v0, judgment_prompts_v1, judgment_prompts_v2, judgment_prompts_v3_gating,
    judgment_prompts_v4_gating, judgment_prompts_v3_gating_corrected)

# Prompt版本映射表
PROMPT_REGISTRY = {
    "c1c4": {
        "v0": answer_c1_c4_v0,
        "v1": answer_c1_c4_v1,
        "v2": answer_c1_c4_v2,
        "v3": answer_c1_c4_v3,
        "v4": answer_c1_c4_v4,
        "v5": answer_c1_c4_v5,
        "v6": answer_c1_c4_v6,
        "v7": answer_c1_c4_v7,
        "v8": answer_c1_c4_v8,
        "v9": answer_c1_c4_v9,
        "v10": answer_c1_c4_v10,
        "v11": answer_c1_c4_v11,
    },
    "c2": {
        "v0": answer_c2_v0,
        "v1": answer_c2_v1,
        "v2": answer_c2_v2,
        "v3": answer_c2_v3,
        "v4": answer_c2_v4,
        "v5": answer_c2_v5,
        "v6": answer_c2_v6,
        "v7": answer_c2_v7,
        "v8": answer_c2_v8,
    },
    "c3": {
        "v0": answer_c3_v0,
        "v1": answer_c3_v1,
        "v2": answer_c3_v2,
    },
    "c5": {
        "v0": answer_c5_v0,
        "v1": answer_c5_v1,
        "v2": answer_c5_v2,
        "v3": answer_c5_v3,
        "v4": answer_c5_v4,
        "v5": answer_c5_v5,
    },
    "subquery": {
        "v0": subquery_prompt_v0,
        "v1": subquery_prompt_v1,
    },
    "reflector": {
        "v0": reflector_prompt_v0,
    },
    "query_intent": {
        "v0": query_intent_v0,
        "v1": query_intent_v1,
        "v2": query_intent_v2,
        "v3": query_intent_v3,
        "gpt_v0": query_intent_prompt_gpt_v0
    },
    "analyze_content": {
        "v0": analyze_content_prompt_v0,
        "v1": analyze_content_prompt_v1,
        "v2": analyze_content_prompt_v2,
        "v3": analyze_content_prompt_v3,
        "gpt_v0": analyze_content_prompt_gpt_v0,
        "factdense": analyze_content_prompt_factdense,
    },
    "judgment": {
        "v0": judgment_prompts_v0,
        "v1": judgment_prompts_v1,
        "v2": judgment_prompts_v2,
        "v3_gating": judgment_prompts_v3_gating,
        "v4_gating": judgment_prompts_v4_gating,
        "v3_gating_corrected": judgment_prompts_v3_gating_corrected,
    },
}

# 默认配置
DEFAULT_CONFIG = {
    "c1c4": "v11",
    "c2": "v8",
    "c3": "v2",
    "c5": "v5",
    "subquery": "v1",
    "reflector": "v0",
    "query_intent": "v0",
    "analyze_content": "v3",
    "judgment": "v2",
}


class PromptLoader:
    """Prompt版本加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化PromptLoader
        
        Args:
            config_path: prompt配置文件路径，如果为None则使用默认配置
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """从yaml文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                prompts_config = data.get("prompts", {})

                # 更新配置
                for key in self.config:
                    if key in prompts_config:
                        self.config[key] = prompts_config[key]
        except Exception as e:
            print(f"Warning: Failed to load prompt config from {config_path}: {e}")
    
    def get_prompt_func(self, category: str) -> Callable:
        """
        获取指定category的prompt函数
        
        Args:
            category: prompt类别 (c1c4, c2, c3, c5, subquery, reflector)
            
        Returns:
            对应版本的prompt函数
        """
        version = self.config.get(category, DEFAULT_CONFIG.get(category))
        
        if category not in PROMPT_REGISTRY:
            raise ValueError(f"Unknown prompt category: {category}")
        
        if version not in PROMPT_REGISTRY[category]:
            raise ValueError(f"Unknown version {version} for category {category}")
        
        return PROMPT_REGISTRY[category][version]
    
    def get_c1c4_prompt(self) -> Callable:
        """获取Category 1/4的prompt函数"""
        return self.get_prompt_func("c1c4")
    
    def get_c2_prompt(self) -> Callable:
        """获取Category 2的prompt函数"""
        return self.get_prompt_func("c2")
    
    def get_c3_prompt(self) -> Callable:
        """获取Category 3的prompt函数"""
        return self.get_prompt_func("c3")
    
    def get_c5_prompt(self) -> Callable:
        """获取Category 5的prompt函数"""
        return self.get_prompt_func("c5")
    
    def get_subquery_prompt(self) -> Callable:
        """获取子查询prompt函数"""
        return self.get_prompt_func("subquery")
    
    def get_reflector_prompt(self) -> Callable:
        """获取反思器prompt函数"""
        return self.get_prompt_func("reflector")
    
    def get_query_intent_prompt(self) -> Callable:
        """获取查询意图prompt函数"""
        return self.get_prompt_func("query_intent")
    
    def get_analyze_content_prompt(self) -> Callable:
        """获取分析内容prompt函数"""
        return self.get_prompt_func("analyze_content")
    
    def get_judgment_prompt(self, version: str = None) -> Callable:
        """
        获取判断prompt函数
        
        Args:
            version: 可选，指定版本（如 'v2_baseline', 'v3_gating'）
                    如果不指定，使用配置文件中的默认版本
        """
        if version is not None:
            if version not in PROMPT_REGISTRY["judgment"]:
                raise ValueError(f"Unknown version {version} for category judgment")
            return PROMPT_REGISTRY["judgment"][version]
        return self.get_prompt_func("judgment")
    
    def get_config_summary(self) -> Dict[str, str]:
        """返回当前配置摘要"""
        return self.config.copy()


def load_prompt_config(config_path: str = None) -> PromptLoader:
    """
    便捷函数：创建PromptLoader实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PromptLoader实例
    """
    return PromptLoader(config_path)

