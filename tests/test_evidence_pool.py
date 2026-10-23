"""Check bounded iterative evidence and actual answer context without model imports."""
import ast
import copy
from datetime import datetime
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/memcore/versions/v1/agent/key_agent_v3.py"


def agent_type():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KeyToGAgentV3")
    names = {"_update_evidence_pool_dispatch", "_update_evidence_pool", "_update_evidence_pool_v2",
             "_bound_evidence_pool", "_evidence_order", "_get_sorted_source_ids", "_render_context",
             "_record_presented_evidence", "_evidence_audit", "_generate_answer",
             "_reflect_on_answer"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(copy=copy, datetime=datetime, re=re, Any=Any, Dict=Dict, List=List, Optional=Optional)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["KeyToGAgentV3"]


def note(identifier, score, content=None):
    return {"id": identifier, "source_id": "D1:" + identifier, "score": score,
            "content": content or "observation " + identifier, "context": "interpretation",
            "timestamp": "2024-01-01", "parsed_time": datetime(2024, 1, 1),
            "keywords": [identifier], "sources": ["topic"]}


class EvidencePoolTests(unittest.TestCase):
    def setUp(self):
        self.agent = agent_type()()
        self.agent.evidence_pool = {}
        self.agent.retrieved_source_ids = set()
        self.agent.evidence_trace = []
        self.agent.presented_source_ids = []
        self.agent.partial_source_ids = []
        self.agent.memory = SimpleNamespace(last_retrieval_trace={"query": "where?"})
        self.agent.evidence_pool_version = "v2"
        self.agent.final_top_k = 2
        self.agent.context_char_limit = 30000
        self.agent.render_top_fraction = 1.0
        self.agent.logger = logging.getLogger("evidence-test")

    def test_rounds_bound_pool_merge_max_and_preserve_seen(self):
        first = [note("c", .2), note("b", .8), note("a", .8)]
        self.agent._update_evidence_pool_dispatch(first)
        self.assertEqual(list(self.agent.evidence_pool), ["a", "b"])
        self.agent._update_evidence_pool_dispatch([note("a", .1), note("c", .9)])
        self.assertEqual(list(self.agent.evidence_pool), ["c", "a"])
        self.assertEqual(self.agent.evidence_pool["a"]["score"], .8)
        self.assertEqual(self.agent.retrieved_source_ids, {"D1:a", "D1:b", "D1:c"})
        self.assertEqual(self.agent.evidence_trace[-1]["evicted_ids"], ["b"])
        self.assertEqual(first[2]["score"], .8)
        self.agent.evidence_pool["a"]["sources"].append("changed")
        self.assertEqual(first[2]["sources"], ["topic"])

    def test_presented_ids_respect_real_character_budget(self):
        self.agent._update_evidence_pool_dispatch([note("a", .9), note("b", .8, "B" * 600)])
        rendered = self.agent._render_context("score")
        first_line = rendered.splitlines()[0]
        guide = "guide\n"
        self.agent._record_presented_evidence(guide + rendered, len(guide + first_line) + 30)
        audit = self.agent._evidence_audit()
        self.assertEqual(audit["presented_source_ids"], ["D1:a"])
        self.assertEqual(audit["partial_source_ids"], ["D1:b"])
        self.assertEqual(audit["seen_source_ids"], ["D1:a", "D1:b"])
        self.agent._record_presented_evidence(guide + rendered, len(guide + first_line))
        self.assertEqual(self.agent.partial_source_ids, [])

    def test_unbounded_is_explicit_and_fraction_keeps_seen(self):
        self.agent.final_top_k = None
        self.agent._update_evidence_pool_dispatch([note(str(i), i/100) for i in range(30)])
        self.assertEqual(len(self.agent.evidence_pool), 30)
        self.agent.render_top_fraction = .5
        self.agent._render_context("score")
        self.assertEqual(len(self.agent.evidence_pool), 15)
        self.assertEqual(len(self.agent.retrieved_source_ids), 30)

    def test_empty_render_clears_previous_spans(self):
        self.agent._update_evidence_pool_dispatch([note("a", .9)])
        self.agent._render_context()
        self.agent.evidence_pool.clear()
        self.assertEqual(self.agent._render_context(), "")
        self.agent._record_presented_evidence("", 30000)
        self.assertEqual(self.agent.presented_source_ids, [])

    def test_larger_limit_preserves_note_and_answer_reflector_context(self):
        self.agent._update_evidence_pool_dispatch([note("a", .9), note("b", .8, "B" * 600)])
        context = self.agent._render_context("score")
        first_line = context.splitlines()[0]
        prompts = []
        class FakeCompletion:
            def get_completion(self, prompt, **kwargs):
                prompts.append(prompt)
                return '{}'
        self.agent.llm = SimpleNamespace(llm=FakeCompletion())
        self.agent.prompt_loader = SimpleNamespace(
            get_c1c4_prompt=lambda: lambda ctx, question: ctx,
            get_reflector_prompt=lambda: lambda question, ctx: ctx)
        self.agent._parse_json = lambda response: {"answer": "ok", "sufficient": True}
        self.agent.use_format_instruction = False
        self.agent.context_char_limit = len(first_line) + 30
        self.agent._generate_answer("where?", context)
        self.assertEqual(self.agent.presented_source_ids, ["D1:a"])
        self.assertEqual(self.agent.partial_source_ids, ["D1:b"])
        self.assertEqual(self.agent.evidence_trace[-1]["context_char_limit"], len(first_line) + 30)
        self.assertNotIn("B" * 100, prompts[-1])
        self.agent._reflect_on_answer("where?", context, "ok")
        self.assertNotIn("B" * 100, prompts[-1])

        self.agent.context_char_limit = len(context) + 10
        self.agent._generate_answer("where?", context)
        self.assertEqual(self.agent.presented_source_ids, ["D1:a", "D1:b"])
        self.assertEqual(self.agent.partial_source_ids, [])
        self.assertIn("B" * 100, prompts[-1])
        self.agent._reflect_on_answer("where?", context, "ok")
        self.assertIn("B" * 100, prompts[-1])


if __name__ == "__main__":
    unittest.main()
