"""Offline construction, topic navigation, retrieval and graph persistence tests."""

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from memcore.versions.v1.agent import key_graph_memory_v3 as module
from memcore.versions.v1.agent.key_agent_v3 import KeyToGAgentV3


class FakeLLMController:
    def __init__(self, *args, **kwargs):
        self.llm = SimpleNamespace(_current_caller=None, model="fake")


class FakeEmbeddingController:
    def __init__(self, *args, **kwargs):
        self.model = "fake-3d"

    def get_embedding(self, text):
        return [1.0, 0.0, 0.0]


class GraphMemoryTests(unittest.TestCase):
    def make_system(self, seed=None):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "llm.yaml"
            config.write_text("llm: {backend: openai, model: fake}\n"
                              "embedding: {backend: openai, model: fake-3d}\n")
            with patch.object(module, "LLMController", FakeLLMController), patch.object(
                    module, "EmbeddingController", FakeEmbeddingController):
                return module.KeyGraphMemoryV3(
                    llm_config_path=str(config), topic_maintenance_threshold=100,
                    logger=logging.getLogger("test_graph_memory"), seed=seed)

    def test_mainline_import_never_attempts_legacy_graph_packages(self):
        script = """
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('neo4j', 'llama_index'):
            raise AssertionError('unexpected legacy import: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
from memcore.versions.v1.agent.key_graph_memory_v3 import KeyGraphMemoryV3
assert KeyGraphMemoryV3.MEMORY_REVISION
"""
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        result = subprocess.run([sys.executable, "-c", script],
                                env=environment, capture_output=True, text=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_construct_topic_retrieve_and_roundtrip(self):
        system = self.make_system(seed=42)
        metadata = [
            {"keywords": ["camping", "family", "river"], "context": "first context"},
            {"keywords": ["camping", "family", "travel"], "context": "second context"},
        ]
        with patch.object(system, "_generate_metadata_with_llm", side_effect=metadata), patch.object(
                system, "_llm_evolution_judgment", return_value=None):
            first = system.add_note("First memory", time="1:00 PM on 01 January, 2024", source_id="D1:1")
            second = system.add_note("Second memory", time="2:00 PM on 01 January, 2024", source_id="D1:2")
            with self.assertRaisesRegex(ValueError, "duplicate memory source_id"):
                system.add_note("Duplicate", source_id="D1:1")
        self.assertEqual((first, second), ("note:D1:1", "note:D1:2"))
        self.assertEqual(len(system.store.nodes("MemoryNote")), 2)
        self.assertEqual(len(system.store.nodes("Keyword")), 4)
        self.assertEqual(len(system.store.edges(types="HAS_KEYWORD")), 6)
        self.assertEqual(system.store.get_node(first)["properties"]["source_id"], "D1:1")
        self.assertTrue(system._fetch_keyword_graph().has_edge("camping", "family"))
        self.assertEqual(system._fetch_keyword_graph()["camping"]["family"]["weight"], 2)

        graph_nodes = list(system._fetch_keyword_graph().nodes())
        cluster = [SimpleNamespace(cluster=0, node=node, is_final_cluster=True)
                   for node in graph_nodes]
        with patch.object(module, "hierarchical_leiden", return_value=cluster) as leiden, patch.object(
                system, "_generate_topic_description", return_value="Family camping topic"):
            system.build_topic_layer()
        self.assertEqual(leiden.call_args.kwargs["random_seed"], 42)
        self.assertEqual(system.last_topic_clustering_trace["partition_audit"]["strategy"],
                         "deterministic_affinity_cardinality_postprocess_v1")
        self.assertEqual(len(system.store.nodes("Topic")), 1)
        self.assertEqual(system.store.nodes("Topic")[0]["id"], "topic:0")
        self.assertEqual(len(system.store.edges(types="BELONGS_TO")), 4)
        self.assertEqual({row[0] for row in system._retrieve_via_topic_vector([1, 0, 0], 3)},
                         {first, second})
        self.assertEqual({row[0] for row in system._retrieve_via_keyword_vector([1, 0, 0], 10)},
                         {first, second})
        with patch.object(system, "_analyze_query_intent", return_value={
                "topic_desc": "family camping", "keywords": ["camping", "family"],
                "time_sorting": "none"}):
            legacy = system.retrieve_topic_aware_structured(
                "Where did they camp?", k_final=2, sort_by="score",
                enable_logical_expansion=False)
            paper = system.retrieve_paper_structured(
                "Where did they camp?", k_final=2, sort_by="score",
                enable_logical_expansion=False)
        self.assertEqual({row["id"] for row in legacy}, {first, second})
        self.assertEqual({row["id"] for row in paper}, {first, second})
        self.assertEqual(len(system.last_retrieval_trace["keyword_queries"]), 2)
        self.assertEqual(len(system.last_retrieval_trace["rrf_anchors"]), 2)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            system.save_graph(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["graph"]["counts"]["nodes"], 7)
            restored = self.make_system()
            restored.load_graph(path)
            self.assertEqual(restored.store.snapshot(), system.store.snapshot())
            self.assertEqual(set(restored.memories), {first, second})
            self.assertEqual(restored.maintenance_counter, 2)
            self.assertEqual(restored.graph_seed, 42)
            self.assertEqual(restored.memories[first].source_id, "D1:1")
            self.assertEqual(restored.snapshot_state(), system.snapshot_state())

    def test_snapshot_restores_independent_rollout_state(self):
        original = self.make_system(seed=42)
        with patch.object(original, "_generate_metadata_with_llm", return_value={
                "keywords": ["camping"], "context": "original"}):
            first = original.add_note("first", source_id="D1:1")
        snapshot = original.snapshot_state()
        left = self.make_system(seed=42)
        right = self.make_system(seed=42)
        left.restore_state(snapshot)
        right.restore_state(snapshot)
        self.assertIsNot(left.llm_controller, right.llm_controller)
        self.assertIsNot(left.store, right.store)
        left._update_neighbors([{"id": first, "new_context": "left-only merge"}])
        left.memories[first].keywords.append("left-only keyword")
        with patch.object(left, "_generate_metadata_with_llm", return_value={
                "keywords": ["camping"], "context": "left-only"}), patch.object(
                left, "_llm_evolution_judgment", return_value=None):
            left.add_note("second", source_id="D1:2")
        self.assertEqual(right.memories[first].context, "original")
        self.assertEqual(right.store.get_node(first)["properties"]["context"], "original")
        self.assertEqual(set(right.memories), {first})
        self.assertEqual(snapshot["memories"][0]["context"], "original")
        self.assertEqual(snapshot["memories"][0]["keywords"], ["camping"])
        self.assertEqual(right.memories[first].keywords, ["camping"])
        self.assertEqual(snapshot["graph"]["counts"]["nodes"], 2)
        mismatch = self.make_system(seed=7)
        with self.assertRaisesRegex(ValueError, "graph_seed"):
            mismatch.restore_state(snapshot)
        self.assertEqual(mismatch.store.nodes(), [])
        damaged = deepcopy(snapshot)
        damaged["memories"][0]["content"] = "corrupt"
        before = right.snapshot_state()
        with self.assertRaisesRegex(ValueError, "differs from graph node"):
            right.restore_state(damaged)
        self.assertEqual(right.snapshot_state(), before)
        bad_audit = deepcopy(snapshot)
        bad_audit["topic_clustering_trace"] = {"partition_audit": {"strategy": "wrong"}}
        with self.assertRaisesRegex(ValueError, "topic partition audit"):
            right.restore_state(bad_audit)
        self.assertEqual(right.snapshot_state(), before)
        old_revision = deepcopy(snapshot)
        old_revision.pop("memory_revision")
        with self.assertRaisesRegex(ValueError, "revision"):
            right.restore_state(old_revision)
        old_revision["memory_revision"] = "raw-note-description-topic-v0"
        with self.assertRaisesRegex(ValueError, "revision"):
            right.restore_state(old_revision)
        self.assertEqual(right.snapshot_state(), before)

    def test_add_note_does_not_publish_memory_before_store_success(self):
        system = self.make_system()
        with patch.object(system, "_generate_metadata_with_llm", return_value={
                "keywords": [], "context": "context"}), patch.object(
                system, "_save_to_store", side_effect=RuntimeError("store failed")):
            with self.assertRaisesRegex(RuntimeError, "store failed"):
                system.add_note("raw", source_id="D1:1")
        self.assertEqual(system.memories, {})
        self.assertEqual(system.maintenance_counter, 0)

    def test_merge_updates_both_views_and_refreshes_embedding(self):
        system = self.make_system()
        note = module.MemoryNote(content="old", id="n1", keywords=[], context="old")
        system.memories[note.id] = note
        system.store.upsert_node(note.id, "MemoryNote", {
            "content": "old", "context": "old", "timestamp": note.timestamp,
            "keywords": [], "embedding": [1.0, 0.0, 0.0], "source_id": None})
        before = system.store.get_node(note.id)["properties"]["embedding"]
        system._update_neighbors([{"id": note.id, "new_context": "merged"}])
        self.assertEqual(system.memories[note.id].context, "merged")
        self.assertEqual(system.store.get_node(note.id)["properties"]["context"], "merged")
        self.assertEqual(system.store.get_node(note.id)["properties"]["embedding"], before)

    def test_context_embedding_and_merge_reindex_are_atomic_on_embedding_failure(self):
        system = self.make_system()
        vectors = {"first context": [1.0, 0.0, 0.0],
                   "second context": [0.0, 1.0, 0.0],
                   "merged context": [0.0, 1.0, 0.0]}
        metadata = [{"keywords": [], "context": "first context"},
                    {"keywords": [], "context": "second context"}]
        with patch.object(system, "get_embedding", side_effect=lambda text: vectors[text]) as embed, patch.object(
                system, "_generate_metadata_with_llm", side_effect=metadata):
            first = system.add_note("original first", source_id="D1:1")
            second = system.add_note("original second", source_id="D1:2")
            self.assertEqual([call.args[0] for call in embed.call_args_list],
                             ["first context", "second context"])
            self.assertEqual(system.store.search("MemoryNote", [0, 1, 0], 1)[0][0], second)
            system._update_neighbors([{"id": first, "new_context": "merged context"}])
            self.assertEqual(system.store.search("MemoryNote", [0, 1, 0], 1)[0][0], first)
            self.assertEqual(system.store.get_node(first)["properties"]["embedding"], [0, 1, 0])
            self.assertEqual(system.memories[first].content, "original first")
            self.assertEqual(system.store.get_node(first)["properties"]["content"], "original first")
            with self.assertRaises(KeyError):
                system._update_neighbors([
                    {"id": first, "new_context": "first context"},
                    {"id": second, "new_context": "unavailable"}])
        self.assertEqual(system.memories[first].context, "merged context")
        self.assertEqual(system.store.get_node(first)["properties"]["embedding"], [0, 1, 0])

    def test_topic_vector_is_all_keyword_centroid_without_description_embedding(self):
        system = self.make_system()
        names = ("alpha", "beta", "gamma")
        for index, name in enumerate(names):
            vector = [float(index == offset) for offset in range(3)]
            system.store.upsert_node(f"keyword:{name}", "Keyword",
                                     {"name": name, "embedding": vector})
        graph = module.nx.Graph()
        graph.add_nodes_from(names)
        with patch.object(system, "_generate_topic_description", return_value="A topic"), patch.object(
                system, "get_embedding", side_effect=AssertionError("description was embedded")):
            topics = system._generate_topics({0: list(names)}, graph)
        self.assertEqual(topics[0]["embedding"], [1 / 3] * 3)
        system._write_topics_to_store(topics)
        self.assertEqual(system.store.get_node("topic:0")["properties"]["embedding"],
                         [1 / 3] * 3)

    def test_leiden_leaves_are_repaired_to_full_bounded_partition(self):
        system = self.make_system(seed=42)
        graph = module.nx.Graph()
        words = [f"kw{i:02d}" for i in range(16)]
        graph.add_nodes_from(words)
        graph.add_edges_from(zip(words, words[1:]))
        parent = [SimpleNamespace(node=word, cluster=0, is_final_cluster=False)
                  for word in words]
        leaves = [SimpleNamespace(node=word, cluster=1 if index < 8 else 2,
                                  is_final_cluster=True)
                  for index, word in enumerate(words)]
        with patch.object(module, "hierarchical_leiden", return_value=parent + leaves) as leiden:
            clusters = system._cluster_keywords(graph)
        self.assertEqual({key: len(value) for key, value in clusters.items()}, {1: 8, 2: 8})
        self.assertEqual(set().union(*map(set, clusters.values())), set(words))
        self.assertEqual(leiden.call_args.kwargs["max_cluster_size"], 15)
        with patch.object(module, "hierarchical_leiden", return_value=[
                SimpleNamespace(node=word, cluster=0, is_final_cluster=True)
                for word in words]):
            split = system._cluster_keywords(graph)
        self.assertEqual(sorted(map(len, split.values())), [8, 8])
        self.assertEqual(set().union(*map(set, split.values())), set(words))
        self.assertEqual(system.last_topic_clustering_trace["partition_audit"]["split_count"], 1)
        graph.add_node("isolated")
        with patch.object(module, "hierarchical_leiden", return_value=leaves):
            repaired = system._cluster_keywords(graph)
        self.assertEqual(sum(map(len, repaired.values())), 17)
        self.assertEqual(sum("isolated" in group for group in repaired.values()), 1)
        self.assertEqual(system.last_topic_clustering_trace["partition_audit"][
            "filled_isolated_keywords"], 1)

    def test_no_edge_keyword_graph_skips_leiden_and_builds_bounded_topic(self):
        system = self.make_system(seed=42)
        for word in ("alpha", "beta", "gamma"):
            system.store.upsert_node(f"keyword:{word}", "Keyword", {
                "name": word, "embedding": [1.0, 0.0, 0.0]})
        with patch.object(module, "hierarchical_leiden", side_effect=AssertionError(
                "Leiden must not receive an edgeless graph")), patch.object(
                system, "_generate_topic_description", return_value="Isolated topic"):
            system.build_topic_layer()
        self.assertEqual(len(system.store.nodes("Topic")), 1)
        self.assertEqual(len(system.store.edges(types="BELONGS_TO")), 3)
        trace = system.last_topic_clustering_trace
        self.assertEqual(trace["leiden_skipped_reason"], "no_keyword_cooccurrence_edges")
        self.assertEqual(trace["partition_audit"]["filled_isolated_keywords"], 3)
        self.assertEqual(trace["partition_audit"]["covered_keywords"], 3)
        restored = self.make_system(seed=42)
        restored.restore_state(system.snapshot_state())
        self.assertEqual(restored.snapshot_state(), system.snapshot_state())
        empty = system._cluster_keywords(module.nx.Graph())
        self.assertEqual(empty, {})
        self.assertFalse(system.last_topic_clustering_trace["partition_audit"]["feasible"])
        for count in (1, 2):
            small = self.make_system(seed=42)
            for word in ("alpha", "beta")[:count]:
                small.store.upsert_node(f"keyword:{word}", "Keyword", {
                    "name": word, "embedding": [1.0, 0.0, 0.0]})
            with patch.object(module, "hierarchical_leiden", side_effect=AssertionError(
                    "Leiden must not receive an edgeless graph")):
                small.build_topic_layer()
            self.assertEqual(small.store.nodes("Topic"), [])
            self.assertFalse(small.last_topic_clustering_trace["partition_audit"]["feasible"])
            self.make_system(seed=42).restore_state(small.snapshot_state())
        no_keywords = self.make_system(seed=42)
        no_keywords.build_topic_layer()
        self.assertIsNone(no_keywords.last_topic_clustering_trace)

    def test_default_retrieval_and_agent_keep_sixty_notes(self):
        system = self.make_system()
        system.store.upsert_node("keyword:q", "Keyword", {"name": "q", "embedding": [1, 0, 0]})
        for i in range(75):
            note_id = f"note:{i:02d}"
            note = module.MemoryNote(content=note_id, id=note_id, context=note_id)
            note.source_id = note_id
            system.memories[note_id] = note
            system.store.upsert_node(note_id, "MemoryNote", {
                "content": note_id, "context": note_id, "timestamp": note.timestamp,
                "keywords": ["q"], "embedding": [1, 0, 0], "source_id": note_id})
            system.store.upsert_edge(note_id, "keyword:q", "HAS_KEYWORD")
        agent = KeyToGAgentV3(system, logger=logging.getLogger("test_default_budget"))
        agent.retrieved_source_ids = set()  # Normally initialized at the start of the QA loop.
        with patch.object(system, "_analyze_query_intent", return_value={
                "topic_desc": "q", "keywords": ["q"], "time_sorting": "none"}):
            direct = system.retrieve_paper_structured("q")
            selected = agent._retrieve_structured(
                query="q", sort_by="time_asc", **agent._retrieval_limits(10, 7))
        agent._update_evidence_pool_dispatch(selected)
        self.assertEqual(len(direct), 60)
        self.assertEqual([row["id"] for row in selected], [row["id"] for row in direct])
        self.assertEqual(len(agent.evidence_pool), 60)
        self.assertEqual(system.last_retrieval_trace["k_topics"], 3)
        self.assertEqual(system.last_retrieval_trace["k_keywords_per_entity"], 10)

    def test_weighted_semantic_propagation_changes_final_topk(self):
        system = self.make_system()
        for note_id in ("a", "b", "c"):
            note = module.MemoryNote(content=note_id, id=note_id, context=note_id)
            note.source_id = note_id
            system.memories[note_id] = note
            system.store.upsert_node(note_id, "MemoryNote", {
                "content": note_id, "context": note_id, "timestamp": note.timestamp,
                "keywords": [], "embedding": [1, 0, 0], "source_id": note_id})
        system.store.upsert_node("keyword:q", "Keyword", {"name": "q", "embedding": [1, 0, 0]})
        system.store.upsert_edge("a", "keyword:q", "HAS_KEYWORD")
        system.store.upsert_edge("a", "b", "SUPPORTS", {"weight": 0.9})
        system.store.upsert_edge("a", "c", "CONFLICTS", {"weight": 0.2})
        with patch.object(system, "_analyze_query_intent", return_value={
                "topic_desc": "q", "keywords": ["q"], "time_sorting": "none"}):
            results = system.retrieve_paper_structured("q", k_final=2, sort_by="score",
                                                       enable_topic_path=False)
        self.assertEqual([row["id"] for row in results], ["a", "b"])
        edges = {row["target"]: row for row in system.last_retrieval_trace["expanded_edges"]}
        self.assertEqual((edges["b"]["weight"], edges["c"]["weight"]), (0.9, 0.2))
        self.assertAlmostEqual(edges["b"]["score"], 0.9 / 61)
        self.assertAlmostEqual(edges["c"]["score"], 0.2 / 61)
        system.store.upsert_edge("a", "b", "SUPPORTS", {"weight": 1.1})
        with patch.object(system, "_analyze_query_intent", return_value={
                "topic_desc": "q", "keywords": ["q"], "time_sorting": "none"}):
            with self.assertRaisesRegex(ValueError, "semantic edge weight"):
                system.retrieve_paper_structured("q", k_final=2, sort_by="score",
                                                 enable_topic_path=False)

    def test_paper_channel_weights_and_neighbor_scale(self):
        system = self.make_system()
        for note_id in ("a", "b", "c"):
            note = module.MemoryNote(content=note_id, id=note_id, context=note_id)
            note.source_id = note_id
            system.memories[note_id] = note
            system.store.upsert_node(note_id, "MemoryNote", {
                "content": note_id, "context": note_id, "timestamp": note.timestamp,
                "keywords": [], "embedding": [1, 0, 0], "source_id": note_id})
        system.store.upsert_edge("a", "c", "SUPPORTS", {"weight": 0.8})
        intent = {"topic_desc": "topic", "keywords": ["term"], "time_sorting": "none"}
        with patch.object(system, "_analyze_query_intent", return_value=intent), patch.object(
                system, "_retrieve_via_topic_vector", return_value=[("a", 1.0, None)]), patch.object(
                system, "_retrieve_via_keyword_vector", return_value=[("b", 1.0, None)]), patch.object(
                system.store, "search", return_value=[]):
            baseline = system.retrieve_paper_structured("q", sort_by="score")
            baseline_trace = deepcopy(system.last_retrieval_trace)
            explicit_defaults = system.retrieve_paper_structured(
                "q", sort_by="score", rrf_topic_weight=1.0,
                rrf_keyword_weight=1.0, neighbor_score_scale=1.0)
            self.assertEqual(baseline, explicit_defaults)
            self.assertEqual(baseline_trace, system.last_retrieval_trace)
            self.assertNotIn("retrieval_weights", baseline_trace)
            keyword_favored = system.retrieve_paper_structured(
                "q", sort_by="score", rrf_keyword_weight=3.0)
            self.assertEqual(keyword_favored[0]["id"], "b")
            weighted = system.retrieve_paper_structured(
                "q", sort_by="score", rrf_topic_weight=2.0,
                rrf_keyword_weight=1.0, neighbor_score_scale=0.25)
        self.assertEqual([row["id"] for row in baseline], ["a", "b", "c"])
        self.assertEqual([row["id"] for row in weighted], ["a", "b", "c"])
        self.assertAlmostEqual(weighted[0]["score"], 2 / 61)
        self.assertAlmostEqual(weighted[2]["score"], (2 / 61) * 0.8 * 0.25)
        edge = system.last_retrieval_trace["expanded_edges"][0]
        self.assertEqual(edge["weight"], 0.8)
        self.assertAlmostEqual(edge["score"], weighted[2]["score"])
        self.assertEqual(system.last_retrieval_trace["retrieval_weights"], {
            "rrf_topic_weight": 2.0, "rrf_keyword_weight": 1.0,
            "neighbor_score_scale": 0.25})
        for key, value in (("rrf_topic_weight", 0), ("rrf_keyword_weight", float("inf")),
                           ("neighbor_score_scale", -0.1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                system.retrieve_paper_structured("q", **{key: value})

    def test_keyword_equal_score_mean_reciprocal_conserves_mass_and_changes_top20(self):
        system = self.make_system()
        ids = [f'n{i:02d}' for i in range(21)]
        for note_id in ids:
            note = module.MemoryNote(content=note_id, id=note_id, context=note_id)
            note.source_id = note_id
            system.memories[note_id] = note
            system.store.upsert_node(note_id, 'MemoryNote', {
                'content': note_id, 'context': note_id, 'timestamp': note.timestamp,
                'keywords': [], 'embedding': [1, 0, 0], 'source_id': note_id})
        topic_hits = [(note_id, float(rank), None) for rank, note_id in enumerate(ids, 1)]
        keyword_hits = [(note_id, 1.0, None) for note_id in ids]
        intent = {'topic_desc': 'topic', 'keywords': ['term'], 'time_sorting': 'none'}
        with patch.object(system, '_analyze_query_intent', return_value=intent), patch.object(
                system, '_retrieve_via_topic_vector', return_value=topic_hits), patch.object(
                system, '_retrieve_via_keyword_vector', return_value=keyword_hits), patch.object(
                system.store, 'search', return_value=[]):
            ordinal = system.retrieve_paper_structured('q', k_final=20, sort_by='score',
                                                        enable_logical_expansion=False)
            ordinal_trace = deepcopy(system.last_retrieval_trace)
            mean = system.retrieve_paper_structured('q', k_final=20, sort_by='score',
                enable_logical_expansion=False, keyword_rrf_tie_policy='mean_reciprocal')
            mean_trace = deepcopy(system.last_retrieval_trace)
            self.assertNotEqual([row['id'] for row in ordinal], [row['id'] for row in mean])
            self.assertNotIn('n10', [row['id'] for row in ordinal])
            self.assertIn('n10', [row['id'] for row in mean])
            self.assertNotIn('n00', [row['id'] for row in mean])
            self.assertNotIn('keyword_rrf_tie_policy', ordinal_trace)
            self.assertEqual(mean_trace['keyword_rrf_tie_policy'], 'mean_reciprocal')
            self.assertEqual(mean_trace['keyword_rrf_tie_groups'][0]['ranks'], [1, 21])
            ordinal_scores = {row['id']: row['score'] for row in ordinal_trace['rrf_anchors']}
            mean_scores = {row['id']: row['score'] for row in mean_trace['rrf_anchors']}
            for rank, note_id in enumerate(reversed(ids), 1):
                self.assertAlmostEqual(mean_scores[note_id] - 1 / (60 + rank),
                    sum(1 / (60 + r) for r in range(1, 22)) / 21)
            self.assertAlmostEqual(sum(mean_scores.values()), sum(ordinal_scores.values()))
            reversed_hits = list(reversed(keyword_hits))
            with patch.object(system, '_retrieve_via_keyword_vector', return_value=reversed_hits):
                system.retrieve_paper_structured('q', sort_by='score',
                    enable_logical_expansion=False, keyword_rrf_tie_policy='mean_reciprocal')
            self.assertEqual(mean_scores, {row['id']: row['score']
                                          for row in system.last_retrieval_trace['rrf_anchors']})

    def test_keyword_unequal_scores_keep_positional_rrf(self):
        system = self.make_system()
        for note_id in ('a', 'b'):
            note = module.MemoryNote(content=note_id, id=note_id, context=note_id)
            note.source_id = note_id
            system.memories[note_id] = note
            system.store.upsert_node(note_id, 'MemoryNote', {
                'content': note_id, 'context': note_id, 'timestamp': note.timestamp,
                'keywords': [], 'embedding': [1, 0, 0], 'source_id': note_id})
        with patch.object(system, '_analyze_query_intent', return_value={
                'topic_desc': 'x', 'keywords': ['x'], 'time_sorting': 'none'}), patch.object(
                system, '_retrieve_via_keyword_vector', return_value=[
                    ('a', 1.0, None), ('b', 0.999999999, None)]), patch.object(
                system.store, 'search', return_value=[]):
            baseline = system.retrieve_paper_structured('q', enable_topic_path=False,
                enable_logical_expansion=False, sort_by='score')
            mean = system.retrieve_paper_structured('q', enable_topic_path=False,
                enable_logical_expansion=False, sort_by='score',
                keyword_rrf_tie_policy='mean_reciprocal')
            weighted_ordinal = system.retrieve_paper_structured('q', enable_topic_path=False,
                enable_logical_expansion=False, sort_by='score', rrf_keyword_weight=2.3)
            weighted_mean = system.retrieve_paper_structured('q', enable_topic_path=False,
                enable_logical_expansion=False, sort_by='score', rrf_keyword_weight=2.3,
                keyword_rrf_tie_policy='mean_reciprocal')
        self.assertEqual(baseline, mean)
        self.assertEqual([row['score'] for row in mean], [1 / 61, 1 / 62])
        self.assertEqual(weighted_ordinal, weighted_mean)
        self.assertEqual([row['score'] for row in weighted_ordinal], [2.3 / 61, 2.3 / 62])

    def test_paper_expands_all_anchors_before_pool_limit(self):
        system = self.make_system()
        for note_id in ("a", "b", "c"):
            note = module.MemoryNote(content=note_id, id=note_id, keywords=[], context=note_id)
            note.source_id = f"D1:{note_id}"
            system.memories[note_id] = note
            system.store.upsert_node(note_id, "MemoryNote", {
                "content": note_id, "context": note_id, "timestamp": note.timestamp,
                "keywords": [], "embedding": [1.0, 0.0, 0.0], "source_id": note.source_id})
        system.store.upsert_node("keyword:one", "Keyword", {"name": "one",
                                                        "embedding": [1.0, 0.0, 0.0]})
        system.store.upsert_edge("a", "keyword:one", "HAS_KEYWORD")
        system.store.upsert_edge("b", "keyword:one", "HAS_KEYWORD")
        system.store.upsert_edge("b", "c", "SUPPORTS", {"weight": 0.9})
        with patch.object(system, "_analyze_query_intent", return_value={
                "topic_desc": "anything", "keywords": ["one"], "time_sorting": "none"}):
            result = system.retrieve_paper_structured("q", k_final=3, sort_by="score",
                                                      enable_topic_path=False)
            limited = system.retrieve_paper_structured("q", k_final=1, sort_by="score",
                                                       enable_topic_path=False)
            limited_trace = system.last_retrieval_trace
            uncapped = system.retrieve_paper_structured("q", k_final=None, sort_by="score",
                                                        enable_topic_path=False)
        self.assertEqual({row["id"] for row in result}, {"a", "b", "c"})
        self.assertEqual(result[-1]["source_label"], "Neighbor")
        self.assertEqual(system.last_retrieval_trace["expanded_edges"][0]["type"], "SUPPORTS")
        self.assertEqual(len(limited), 1)
        self.assertEqual({row["id"] for row in uncapped}, {"a", "b", "c"})
        self.assertTrue(set(limited_trace["selected_ids"]).issubset(
            limited_trace["seen_note_ids"]))
        self.assertEqual(set(limited_trace["seen_source_ids"]), {"D1:a", "D1:b", "D1:c"})

    def test_required_generation_errors_do_not_become_empty_results(self):
        system = self.make_system(seed=42)
        with self.assertRaisesRegex(RuntimeError, "Query intent analysis failed"):
            system._analyze_query_intent("question")
        with self.assertRaisesRegex(RuntimeError, "Topic description generation failed"):
            system._generate_topic_description(["camping"])
        graph = system._fetch_keyword_graph()
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edge("a", "b")
        with patch.object(module, "hierarchical_leiden", side_effect=RuntimeError("broken")):
            with self.assertRaisesRegex(RuntimeError, "Leiden topic clustering failed"):
                system._cluster_keywords(graph)


if __name__ == "__main__":
    unittest.main()
