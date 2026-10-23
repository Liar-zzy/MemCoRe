"""Exercise the actual evaluator orchestration without importing its ML clients.

Only function definitions are compiled from evaluation.py. The real data loader,
prompt registry and token trackers run; all model/database boundaries are fakes.
"""

import ast
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import tempfile
import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from memcore.config import PROJECT_ROOT
from memcore.settings import load_yaml
from memcore.versions.v1.data.load_dataset import load_locomo_dataset
from memcore.versions.v1.prompts.prompt_loader import PromptLoader
from memcore.versions.v1.protocol import EXECUTION_REVISION
from memcore.versions.v1.utils.token_tracker import (
    enable_tracking, get_construction_tracker, get_retrieval_tracker,
)


def aggregate_stub(metrics, categories):
    """Deterministic arithmetic; no metric model or downloader is needed."""
    groups = {"overall": metrics}
    for category in set(categories):
        groups[f"category_{category}"] = [
            item for item, label in zip(metrics, categories) if label == category
        ]
    return {
        group: {
            key: {"mean": sum(item[key] for item in items) / len(items)}
            for key in (items[0] if items else {})
        }
        for group, items in groups.items()
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = []
        self.agents = []
        self.answer_calls = []
        self.fail_question_once = None
        self.failure_query_trace = None
        self.success_query_trace = None
        self.save_success = True
        self.track_costs = False
        self.capture_random = False
        self.random_draws = []
        self.name = "offline"
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(json.dumps([{
            "qa": [{"question": "Where?", "answer": "Paris", "category": 1,
                    "evidence": ["D1:1"]}],
            "conversation": {
                "speaker_a": "Alice", "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01",
                "session_1": [{"speaker": "Alice", "dia_id": "D1:1",
                               "text": "I live in Paris."}],
            },
            "event_summary": {}, "observation": {},
        }]), encoding="utf-8")
        self.llm = self.root / "llm.yaml"
        self.llm.write_text("llm:\n  model: offline-model\n  backend: openai\n", encoding="utf-8")
        self.cache = self.root / "cached" / "cached_mem_key_graph_v3" / self.name
        self.cache.mkdir(parents=True)
        self.results = self.root / "results" / self.name
        self.core_metrics = Mock(return_value={"exact_match": 1, "f1": 1.0, "bleu1": 1.0})
        owner = self

        class FakeAgent:
            def __init__(self, **kwargs):
                owner.events.append("agent")
                owner.agent_kwargs = kwargs
                owner.agents.append(self)
                self.last_evidence_trace = [{"stage": "initial"}]
                self.last_seen_source_ids = ["D1:1"]
                self.last_presented_source_ids = ["D1:1"]
                self.memory_system = SimpleNamespace(
                    memories={},
                    last_query_intent_trace=None,
                    llm_controller=SimpleNamespace(llm=SimpleNamespace()),
                    build_topic_layer=lambda **kw: owner.events.append("topics"),
                    save_graph=self.save_graph,
                    load_graph=self.load_graph,
                )

            def save_graph(self, path):
                owner.events.append("save")
                if not owner.save_success:
                    raise OSError("save failed")
                Path(path).write_text(json.dumps({"memories": self.memory_system.memories}),
                                      encoding="utf-8")

            def load_graph(self, path):
                owner.events.append("load")
                self.memory_system.memories = json.loads(Path(path).read_text())["memories"]

            def add_memory(self, content, **kwargs):
                owner.events.append("build")
                if owner.track_costs:
                    get_construction_tracker().record("build", 3, 2, 0.1)
                self.memory_system.memories[kwargs["source_id"]] = {
                    "content": content, "source_id": kwargs["source_id"],
                }

            def answer_question(self, *args):
                owner.events.append("answer")
                owner.answer_calls.append(args[0])
                if owner.capture_random:
                    owner.random_draws.append((args[0], random.random()))
                if owner.track_costs:
                    get_retrieval_tracker().record("answer", 5, 1, 0.2)
                if args[0] == owner.fail_question_once:
                    owner.fail_question_once = None
                    self.memory_system.last_query_intent_trace = owner.failure_query_trace
                    raise RuntimeError("injected QA failure")
                self.memory_system.last_query_intent_trace = owner.success_query_trace
                return ('{"answer": "Paris"}', None, "evidence", "history", ["D1:1"], "prompt")

        namespace = {
            "os": os, "json": json, "logging": logging, "datetime": datetime,
            "EXECUTION_REVISION": EXECUTION_REVISION,
            "random": random, "np": np, "math": math,
            "defaultdict": defaultdict, "Path": Path, "re": re,
            "load_yaml": load_yaml, "load_locomo_dataset": load_locomo_dataset,
            "PromptLoader": PromptLoader,
            "enable_tracking": enable_tracking,
            "get_construction_tracker": get_construction_tracker,
            "get_retrieval_tracker": get_retrieval_tracker,
        }
        path = PROJECT_ROOT / "src/memcore/versions/v1/evaluation.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        extracted = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
        exec(compile(extracted, str(path), "exec"), namespace)
        namespace.update({
            "advancedMemAgent": FakeAgent,
            "setup_logger": lambda *args: Mock(),
            "calculate_core_metrics": self.core_metrics,
            "aggregate_metrics": aggregate_stub,
            "AmemGraphSystem": type("UnusedGraphSystem", (), {}),
            "KEYWORD_AVAILABLE": False,
        })
        self.evaluate = namespace["evaluate_dataset"]

    def run_evaluation(self):
        with redirect_stdout(io.StringIO()):
            result = self.evaluate(
                dataset_path=str(self.dataset), output_path=str(self.root / "results"),
                llm_config_path=str(self.llm), name=self.name,
                cache_dir=str(self.root / "cached"), log_dir=str(self.root / "logs"),
            )
        self.assertEqual(result["total_questions"], 1)
        self.assertEqual(result["category_distribution"], {"1": 1})
        self.assertEqual(result["aggregate_metrics"]["overall"]["retrieval_recall"]["mean"], 1.0)
        return result

    def seed_graph(self):
        (self.cache / "graph_sample_0.json").write_text(json.dumps({
            "memories": {"cached-id": {"source_id": "D1:1", "content": "cached"}}
        }), encoding="utf-8")

    def add_questions(self):
        data = json.loads(self.dataset.read_text(encoding="utf-8"))
        data[0]["qa"].extend([
            {"question": "Where next?", "answer": "Paris", "category": 1,
             "evidence": ["D1:1"]},
            {"question": "Where finally?", "answer": "Paris", "category": 1,
             "evidence": ["D1:1"]},
        ])
        self.dataset.write_text(json.dumps(data), encoding="utf-8")

    def evaluate_raw(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.evaluate(
                dataset_path=str(self.dataset), output_path=str(self.root / "results"),
                llm_config_path=str(self.llm), name=self.name,
                cache_dir=str(self.root / "cached"), log_dir=str(self.root / "logs"),
                **kwargs,
            )

    def test_first_run_then_completed_resume_without_model_or_database(self):
        first = self.run_evaluation()
        self.assertEqual(self.agent_kwargs["context_char_limit"], 90000)
        self.assertEqual(first["context_char_limit"], 90000)
        self.assertEqual(first["evidence_budget"]["round_top_k"], 60)
        self.assertEqual(self.agent_kwargs["final_top_k"], 60)
        self.assertIn("build", self.events)
        self.assertIn("save", self.events)
        self.assertTrue((self.cache / "graph_sample_0.json").is_file())
        record = json.loads((self.results / f"sample_0_{self.name}_results.json").read_text())["qa_results"][0]
        self.assertEqual(record["seen_source_ids"], ["D1:1"])
        self.assertEqual(record["presented_source_ids"], ["D1:1"])
        self.assertEqual(record["evidence_trace"], [{"stage": "initial"}])
        self.assertTrue((self.results / f"sample_0_{self.name}_results.json").is_file())
        self.assertTrue((self.results / "metrics_summary.csv").is_file())
        self.assertEqual(json.loads((self.results / "final_results.json").read_text()), first)
        self.events.clear()
        self.core_metrics.reset_mock()
        resumed = self.run_evaluation()
        self.assertEqual(resumed, first)
        self.assertEqual(self.events, [])
        self.core_metrics.assert_not_called()

    def test_nondefault_context_limit_reaches_agent_and_final_results(self):
        result = self.evaluate_raw(context_char_limit=30000)
        self.assertEqual(self.agent_kwargs["context_char_limit"], 30000)
        self.assertEqual(result["context_char_limit"], 30000)
        self.assertEqual(json.loads((self.results / "final_results.json").read_text())["context_char_limit"], 30000)
        with self.assertRaisesRegex(ValueError, "context_char_limit"):
            self.evaluate_raw(context_char_limit=0)

    def test_round_budget_reaches_agent_and_checkpoint_and_guards_resume(self):
        result = self.evaluate_raw(round_top_k=10, final_top_k=60)
        self.assertEqual(self.agent_kwargs["round_top_k"], 10)
        self.assertEqual(self.agent_kwargs["final_top_k"], 60)
        self.assertEqual(result["round_top_k"], 10)
        path = self.results / f"sample_0_{self.name}_results.json"
        saved = path.read_text()
        budget = json.loads(saved)["evidence_budget"]
        self.assertEqual(budget["round_top_k"], 10)
        self.assertEqual(budget["final_top_k"], 60)
        self.events.clear()
        self.assertEqual(self.evaluate_raw(round_top_k=10, final_top_k=60), result)
        self.assertEqual(self.events, [])
        with self.assertRaisesRegex(ValueError, "use a new run name"):
            self.evaluate_raw(round_top_k=20, final_top_k=60)
        self.assertEqual(self.events, [])
        self.assertEqual(path.read_text(), saved)

    def test_old_parser_revision_cannot_be_resumed_as_new_results(self):
        self.run_evaluation()
        path = self.results / f"sample_0_{self.name}_results.json"
        saved = json.loads(path.read_text())
        saved["execution_revision"] = "qa-json-recovery-v0.2.3"
        path.write_text(json.dumps(saved))
        before = path.read_text()
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "use a new run name"):
            self.evaluate_raw()
        self.assertEqual(self.events, [])
        self.assertEqual(path.read_text(), before)

    def test_cosine_budget_records_unbounded_candidates_and_bounded_pool(self):
        result = self.evaluate_raw(evidence_selection_version="query_note_cosine_v1")
        self.assertIsNone(result["evidence_budget"]["round_top_k"])
        self.assertEqual(result["evidence_budget"]["final_top_k"], 60)

    def test_full_metrics_rejected_before_building_or_answering(self):
        with self.assertRaisesRegex(ValueError, "Unsupported metrics_mode"):
            self.evaluate_raw(metrics_mode="full")
        self.assertEqual(self.events, [])

    def test_invalid_or_ignored_budgets_fail_before_agent_creation(self):
        for kwargs in ({"round_top_k": 0}, {"round_top_k": True}, {"retrieve_k": 5},
                       {"round_top_k": 10, "retrieval_version": "v1"},
                       {"round_top_k": 10, "evidence_selection_version": "query_note_cosine_v1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.evaluate_raw(**kwargs)
        self.assertEqual(self.events, [])

    def test_resume_accepts_historical_sample_filename(self):
        first = self.run_evaluation()
        (self.results / f"sample_0_{self.name}_results.json").rename(
            self.results / "sample_0_results.json")
        self.events.clear()
        self.core_metrics.reset_mock()
        self.assertEqual(self.run_evaluation(), first)
        self.assertEqual(self.events, [])

    def test_build_then_save_before_answer(self):
        self.run_evaluation()
        self.assertLess(self.events.index("build"), self.events.index("save"))
        self.assertLess(self.events.index("save"), self.events.index("answer"))

    def test_graph_snapshot_restores_notes_without_build(self):
        self.seed_graph()
        self.run_evaluation()
        self.assertIn("load", self.events)
        self.assertNotIn("build", self.events)
        self.assertNotIn("topics", self.events)
        self.assertNotIn("save", self.events)
        self.assertEqual(self.agents[0].memory_system.memories,
                         {"cached-id": {"source_id": "D1:1", "content": "cached"}})

    def test_invalid_graph_snapshot_fails_without_rebuild(self):
        (self.cache / "graph_sample_0.json").write_text("invalid", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.evaluate_raw()
        self.assertIn("load", self.events)
        self.assertNotIn("build", self.events)

    def test_second_question_failure_resumes_only_missing_questions(self):
        self.add_questions()
        self.fail_question_once = "Where next?"
        partial = self.evaluate_raw()
        checkpoint_path = self.results / f"sample_0_{self.name}_results.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        self.assertEqual(partial["total_questions"], 1)
        self.assertEqual(partial["expected_questions"], 3)
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["execution_errors"], 1)
        self.assertEqual(checkpoint["total_qa"], 1)
        self.assertEqual(checkpoint["expected_qa"], 3)
        self.assertFalse(checkpoint["complete"])
        self.assertEqual(checkpoint["failed_qa_index"], 1)
        self.assertEqual(checkpoint["qa_results"][0]["qa_index"], 0)
        self.assertEqual(self.answer_calls, ["Where?", "Where next?"])

        # Simulate a legacy progress file that incorrectly marked a partial sample complete.
        progress = self.results / "progress_key_advanced.json"
        progress.write_text(json.dumps({"completed_samples": [0]}), encoding="utf-8")
        self.events.clear()
        complete = self.evaluate_raw()
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["total_questions"], 3)
        self.assertEqual(complete["execution_errors"], 0)
        self.assertEqual(complete["aggregate_metrics"]["overall"]["f1"]["mean"], 1.0)
        self.assertEqual(self.answer_calls,
                         ["Where?", "Where next?", "Where next?", "Where finally?"])
        checkpoint = json.loads(checkpoint_path.read_text())
        self.assertEqual([entry["qa_index"] for entry in checkpoint["qa_results"]], [0, 1, 2])
        self.assertEqual(checkpoint["errors"], [])
        self.assertEqual(len(checkpoint["error_history"]), 1)
        self.assertTrue(checkpoint["complete"])

        # A third invocation must skip all QAs and preserve both old costs and history.
        before = checkpoint
        previous_calls = list(self.answer_calls)
        third = self.evaluate_raw()
        self.assertTrue(third["complete"])
        self.assertEqual(third["execution_errors"], 0)
        self.assertEqual(self.answer_calls, previous_calls)
        self.assertEqual(json.loads(checkpoint_path.read_text()), before)

    def test_repeated_failure_keeps_one_active_error_and_all_attempts(self):
        self.add_questions()
        self.track_costs = True
        self.fail_question_once = "Where next?"
        first = self.evaluate_raw()
        self.assertFalse(first["complete"])
        self.assertEqual(first["execution_errors"], 1)
        path = self.results / f"sample_0_{self.name}_results.json"
        original = json.loads(path.read_text())
        self.assertEqual(len(original["errors"]), 1)
        self.fail_question_once = "Where next?"
        second = self.evaluate_raw()
        self.assertFalse(second["complete"])
        self.assertEqual(second["execution_errors"], 1)
        checkpoint = json.loads(path.read_text())
        self.assertEqual([error["qa_index"] for error in checkpoint["errors"]], [1])
        self.assertEqual([error["qa_index"] for error in checkpoint["error_history"]], [1, 1])
        self.assertEqual(len(checkpoint["cost_attempts"]), 2)
        self.assertEqual(checkpoint["retrieval_cost"]["call_count"], 3)
        third = self.evaluate_raw()
        self.assertTrue(third["complete"])
        self.assertEqual(third["execution_errors"], 0)
        final = json.loads(path.read_text())
        self.assertEqual(final["errors"], [])
        self.assertEqual(len(final["error_history"]), 2)
        self.assertEqual(final["retrieval_cost"]["call_count"], 5)

    def test_query_intent_failure_diagnostics_are_scoped_and_redacted(self):
        self.add_questions()
        self.success_query_trace = [{"attempt": 1, "retry_reason": None,
                                     "finish_reason": "stop", "output_characters": 40}]
        self.failure_query_trace = [
            {"attempt": index, "caller": "query_intent", "format": "json_schema",
             "output_characters": index * 12, "finish_reason": "length",
             "retry_reason": "json_decode_error", "raw_response": "secret"}
            for index in (1, 2, 3)]
        self.fail_question_once = "Where next?"
        self.evaluate_raw()
        path = self.results / f"sample_0_{self.name}_results.json"
        checkpoint = json.loads(path.read_text())
        attempts = checkpoint["errors"][0]["query_intent_attempts"]
        self.assertEqual([entry["attempt"] for entry in attempts], [1, 2, 3])
        self.assertEqual(attempts[-1]["finish_reason"], "length")
        self.assertEqual(attempts[-1]["output_characters"], 36)
        self.assertNotIn("raw_response", attempts[-1])
        self.assertEqual(checkpoint["error_history"][0]["query_intent_attempts"], attempts)

        # A new QA failure before intent analysis must not reuse the previous
        # successful query's trace, or the previous failed attempt's trace.
        self.failure_query_trace = None
        self.fail_question_once = "Where finally?"
        self.evaluate_raw()
        checkpoint = json.loads(path.read_text())
        self.assertEqual(checkpoint["errors"][0]["qa_index"], 2)
        self.assertNotIn("query_intent_attempts", checkpoint["errors"][0])
        self.assertEqual(len(checkpoint["error_history"]), 2)
        self.assertEqual(checkpoint["error_history"][0]["query_intent_attempts"], attempts)

    def test_legacy_error_list_migrates_without_losing_failure_history(self):
        self.add_questions()
        self.fail_question_once = "Where next?"
        self.evaluate_raw()
        path = self.results / f"sample_0_{self.name}_results.json"
        old = json.loads(path.read_text())
        old.pop("error_history")
        old["errors"].append(dict(old["errors"][0]))
        path.write_text(json.dumps(old), encoding="utf-8")
        recovered = self.evaluate_raw()
        self.assertTrue(recovered["complete"])
        checkpoint = json.loads(path.read_text())
        self.assertEqual(checkpoint["errors"], [])
        self.assertEqual(len(checkpoint["error_history"]), 2)
        self.assertEqual([row["qa_index"] for row in checkpoint["qa_results"]], [0, 1, 2])

    def test_parse_error_checkpoint_cannot_claim_completion(self):
        self.run_evaluation()
        path = self.results / f"sample_0_{self.name}_results.json"
        checkpoint = json.loads(path.read_text())
        checkpoint["qa_results"][0]["prediction_parse_error"] = True
        path.write_text(json.dumps(checkpoint), encoding="utf-8")
        self.answer_calls.clear()
        result = self.evaluate_raw()
        self.assertFalse(result["complete"])
        self.assertEqual(result["prediction_parse_errors"], 1)
        self.assertEqual(self.answer_calls, [])
        self.assertFalse(json.loads(path.read_text())["complete"])

    def test_legacy_complete_checkpoint_validates_question_identity(self):
        self.add_questions()
        self.evaluate_raw()
        path = self.results / f"sample_0_{self.name}_results.json"
        legacy = json.loads(path.read_text())
        legacy.pop("complete")
        legacy.pop("expected_qa")
        for entry in legacy["qa_results"]:
            entry.pop("qa_index")
        path.write_text(json.dumps(legacy), encoding="utf-8")
        self.answer_calls.clear()
        self.evaluate_raw()
        self.assertEqual(self.answer_calls, [])

        legacy["qa_results"][1]["reference"] = "different reference"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        self.evaluate_raw()
        self.assertEqual(self.answer_calls,
                         ["Where?", "Where next?", "Where finally?"])

    def test_legacy_partial_checkpoint_is_not_skipped_by_old_progress(self):
        self.add_questions()
        self.fail_question_once = "Where next?"
        self.evaluate_raw()
        path = self.results / f"sample_0_{self.name}_results.json"
        old = json.loads(path.read_text())
        old.pop("complete")
        old.pop("expected_qa")
        old["qa_results"][0].pop("qa_index")
        path.write_text(json.dumps(old), encoding="utf-8")
        (self.results / "progress_key_advanced.json").write_text(
            json.dumps({"completed_samples": [0]}), encoding="utf-8")
        self.answer_calls.clear()
        complete = self.evaluate_raw()
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["total_questions"], 3)
        self.assertEqual(self.answer_calls, ["Where next?", "Where finally?"])

    def test_core_metrics_mode_and_seed_resume(self):
        self.add_questions()
        self.capture_random = True
        self.fail_question_once = "Where next?"
        partial = self.evaluate_raw(metrics_mode="core_historical_v1", seed=17)
        self.assertFalse(partial["complete"])
        self.core_metrics.assert_called_once_with("Paris", "Paris", f1_version="historical_v1")
        self.assertEqual(self.core_metrics.call_count, 1)
        self.evaluate_raw(metrics_mode="core_historical_v1", seed=17)
        self.assertEqual(self.core_metrics.call_count, 3)
        second_draws = [value for question, value in self.random_draws
                        if question == "Where next?"]
        self.assertEqual(second_draws, [random.Random(18).random()] * 2)

    def test_snapshot_save_failure_stops_before_answer(self):
        self.save_success = False
        with self.assertRaisesRegex(OSError, "save failed"):
            self.evaluate_raw()
        self.assertIn("build", self.events)
        self.assertIn("save", self.events)
        self.assertNotIn("answer", self.events)
        self.assertFalse((self.results / "final_results.json").exists())

    def test_cost_attempts_preserve_build_and_accumulate_retrieval(self):
        self.add_questions()
        self.track_costs = True
        self.fail_question_once = "Where next?"
        self.evaluate_raw()
        path = self.results / f"sample_0_{self.name}_results.json"
        first = json.loads(path.read_text())
        self.assertEqual(first["construction_cost"]["call_count"], 1)
        self.assertEqual(first["retrieval_cost"]["call_count"], 2)
        self.evaluate_raw()
        resumed = json.loads(path.read_text())
        self.assertEqual(len(resumed["cost_attempts"]), 2)
        self.assertEqual([attempt["construction_cost"]["call_count"]
                          for attempt in resumed["cost_attempts"]], [1, 0])
        self.assertEqual([attempt["retrieval_cost"]["call_count"]
                          for attempt in resumed["cost_attempts"]], [2, 2])
        self.assertEqual(resumed["construction_cost"]["call_count"], 1)
        self.assertEqual(resumed["retrieval_cost"]["call_count"], 4)
        self.assertEqual(resumed["errors"], [])
        self.assertEqual(len(resumed["error_history"]), 1)
        calls_before = len(self.answer_calls)
        self.evaluate_raw()
        self.assertEqual(len(self.answer_calls), calls_before)
        self.assertEqual(json.loads(path.read_text())["cost_attempts"], resumed["cost_attempts"])

    def test_core_normalized_scores_empty_reference(self):
        data = json.loads(self.dataset.read_text())
        data[0]["qa"][0]["answer"] = None
        self.dataset.write_text(json.dumps(data), encoding="utf-8")
        result = self.evaluate_raw(metrics_mode="core_normalized_v1")
        self.assertTrue(result["complete"])
        self.core_metrics.assert_called_once_with("Paris", None, f1_version="normalized_v1")

    def test_sample_indices_keep_original_identity(self):
        samples = json.loads(self.dataset.read_text())
        samples.append(samples[0])
        self.dataset.write_text(json.dumps(samples), encoding="utf-8")
        result = self.evaluate_raw(sample_indices=[1])
        self.assertTrue(result["complete"])
        self.assertEqual(result["completed_samples"], [1])
        self.assertTrue((self.cache / "graph_sample_1.json").exists())
        self.assertFalse((self.cache / "graph_sample_0.json").exists())
        record = json.loads((self.results / f"sample_1_{self.name}_results.json").read_text())["qa_results"][0]
        self.assertEqual(record["sample_id"], 1)


if __name__ == "__main__":
    unittest.main()
