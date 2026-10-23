"""Exercise real round and pool budget methods without loading model dependencies."""

import ast
import copy
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src/memcore/versions/v1/agent/key_agent_v3.py"


def agent_type():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "KeyToGAgentV3")
    names = {"_retrieval_limits", "_retrieve_structured", "_update_evidence_pool_dispatch",
             "_update_evidence_pool_v2", "_bound_evidence_pool", "_evidence_order",
             "_get_sorted_source_ids"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in names]
    namespace = dict(copy=copy, datetime=datetime, Dict=Dict, List=List, Optional=Optional)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["KeyToGAgentV3"]


def note(number, score):
    identifier = f"note:{number}"
    return {"id": identifier, "source_id": f"D1:{number}", "score": score,
            "content": f"observation {number}", "context": f"context {number}",
            "timestamp": "2024-01-01", "parsed_time": datetime(2024, 1, 1),
            "keywords": [str(number)], "sources": ["Keyword"]}


class FakeMemory:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []
        self.last_retrieval_trace = {}

    def retrieve_paper_structured(self, **kwargs):
        self.calls.append(kwargs)
        ranked = self.rounds[len(self.calls) - 1]
        selected = ranked if kwargs["k_final"] is None else ranked[:kwargs["k_final"]]
        self.last_retrieval_trace = {
            "mode": "paper", "k_final": kwargs["k_final"],
            "seen_source_ids": [item["source_id"] for item in ranked],
            "selected_ids": [item["id"] for item in selected],
        }
        return copy.deepcopy(selected)


class RoundBudgetTests(unittest.TestCase):
    def make_agent(self, final_top_k, round_top_k=None, rounds=None, version="paper"):
        agent = agent_type()()
        agent.retrieval_version = version
        agent.final_top_k = final_top_k
        agent.round_top_k = round_top_k
        agent.evidence_selection_version = "legacy_rrf"
        agent.evidence_pool_version = "v2"
        agent.rrf_topic_weight = 1.0
        agent.rrf_keyword_weight = 1.0
        agent.neighbor_score_scale = 1.0
        agent.keyword_rrf_tie_policy = "ordinal"
        agent.memory = FakeMemory(rounds or [])
        agent.logger = logging.getLogger("round-budget-test")
        agent.evidence_pool = {}
        agent.retrieved_source_ids = set()
        agent.evidence_trace = []
        return agent

    def retrieve_round(self, agent, question="where?"):
        return agent._retrieve_structured(
            query=question, sort_by="score", **agent._retrieval_limits(10, 7))

    def test_default_round_budget_inherits_final_for_20_and_60(self):
        for final_k in (20, 60):
            with self.subTest(final_top_k=final_k):
                agent = self.make_agent(final_k, rounds=[[note(i, 1 - i / 100)
                                                          for i in range(75)]])
                selected = self.retrieve_round(agent)
                self.assertEqual(len(selected), final_k)
                self.assertEqual(agent.memory.calls[0]["k_final"], final_k)
                self.assertEqual(agent.memory.calls[0]["k_topics"], 3)
                self.assertEqual(agent.memory.calls[0]["k_keywords"], 10)
                agent._update_evidence_pool_dispatch(selected)
                self.assertEqual(len(agent.evidence_pool), final_k)
                self.assertEqual(agent.evidence_trace[0]["retrieval"]["k_final"], final_k)
                self.assertEqual(agent.evidence_trace[0]["final_top_k"], final_k)

    def test_round_10_final_60_accumulates_distinct_rounds(self):
        agent = self.make_agent(60, 10, rounds=[
            [note(i, 1 - i / 100) for i in range(30)],
            [note(i, .9 - (i - 30) / 100) for i in range(30, 60)],
        ])
        for question in ("first", "second"):
            selected = self.retrieve_round(agent, question)
            self.assertEqual(len(selected), 10)
            agent._update_evidence_pool_dispatch(selected)
        self.assertEqual(len(agent.evidence_pool), 20)
        self.assertEqual(len(set(agent.evidence_pool)), 20)
        self.assertEqual([call["k_final"] for call in agent.memory.calls], [10, 10])
        self.assertEqual([event["final_top_k"] for event in agent.evidence_trace], [60, 60])
        self.assertEqual([event["retrieval"]["k_final"] for event in agent.evidence_trace],
                         [10, 10])

    def test_round_60_final_20_bounds_only_pool(self):
        agent = self.make_agent(20, 60, rounds=[[note(i, 1 - i / 100)
                                                 for i in range(75)]])
        selected = self.retrieve_round(agent)
        self.assertEqual(len(selected), 60)
        agent._update_evidence_pool_dispatch(selected)
        self.assertEqual(len(agent.evidence_pool), 20)
        self.assertEqual(agent.memory.calls[0]["k_final"], 60)
        self.assertEqual(agent.evidence_trace[0]["retrieval"]["k_final"], 60)
        self.assertEqual(agent.evidence_trace[0]["final_top_k"], 20)
        self.assertEqual(len(agent.evidence_trace[0]["candidates"]), 60)

    def test_direct_retrieve_structured_passes_explicit_parameters(self):
        agent = self.make_agent(20, 10, rounds=[[note(1, .9)]])
        agent._retrieve_structured(query="direct", k_keywords=6, k_topics=2,
                                   k_final=7, sort_by="time", enable_topic_path=False,
                                   enable_keyword_path=True, enable_logical_expansion=False)
        self.assertEqual(agent.memory.calls[0], {
            "query": "direct", "k_keywords": 6, "k_topics": 2, "k_final": 7,
            "sort_by": "time", "enable_topic_path": False,
            "enable_keyword_path": True, "enable_logical_expansion": False,
        })

    def test_legacy_limits_preserve_initial_and_evolution_parameters(self):
        agent = self.make_agent(60, version="v1")
        self.assertEqual(agent._retrieval_limits(10, 7),
                         {"k_keywords": 20, "k_topics": 7, "k_final": 10})
        self.assertEqual(agent._retrieval_limits(5, 7),
                         {"k_keywords": 10, "k_topics": 7, "k_final": 5})


if __name__ == "__main__":
    unittest.main()
