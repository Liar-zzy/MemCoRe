"""Behavioral tests for the in-process graph, without service boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from memcore.versions.v1.storage.graph_store import InMemoryGraphStore


def example_graph():
    graph = InMemoryGraphStore(embedding_dim=2)
    graph.upsert_node("a", "MemoryNote", {"embedding": [1, 0], "tags": ["first"]})
    graph.upsert_node("b", "MemoryNote", {"embedding": [0, 1]})
    graph.upsert_node("c", "MemoryNote", {"embedding": [-1, 0]})
    graph.upsert_node("keyword", "Keyword", {"embedding": [1, 0]})
    graph.upsert_edge("a", "b", "SUPPORTS", {"strength": 0.7})
    graph.upsert_edge("a", "b", "RELATED_TO", {"strength": 0.5})
    graph.upsert_edge("b", "a", "CONFLICTS", {"strength": 0.8})
    graph.upsert_edge("a", "a", "RELATED_TO")
    return graph


class GraphStoreTests(unittest.TestCase):
    def test_exact_cosine_scores_exclusion_and_ties(self):
        graph = example_graph()
        self.assertEqual(graph.search("MemoryNote", [1, 0], 3),
                         [("a", 1.0), ("b", 0.5), ("c", 0.0)])
        self.assertEqual(graph.search("MemoryNote", [1, 0], 3, exclude_ids={"a"}),
                         [("b", 0.5), ("c", 0.0)])
        graph.upsert_node("aa", "MemoryNote", {"embedding": np.array([0, 2])})
        self.assertEqual([node_id for node_id, _ in graph.search("MemoryNote", [1, 0], 4)],
                         ["a", "aa", "b", "c"])
        self.assertEqual(graph.search("Keyword", [1, 0], 1), [("keyword", 1.0)])

    def test_index_refreshes_after_vector_or_label_update(self):
        graph = example_graph()
        graph.search("MemoryNote", [1, 0], 3)  # populate matrix
        cached_matrix = graph._label_matrices["MemoryNote"][1]
        graph.upsert_node("a", "MemoryNote", {"embedding": [1, 0], "note": "new text"})
        self.assertIs(graph._label_matrices["MemoryNote"][1], cached_matrix)
        graph.upsert_node("c", "MemoryNote", {"embedding": [2, 0]})
        self.assertEqual([node_id for node_id, _ in graph.search("MemoryNote", [1, 0], 3)],
                         ["a", "c", "b"])
        graph.upsert_node("a", "Keyword", {"embedding": [1, 0]})
        self.assertEqual([node_id for node_id, _ in graph.search("MemoryNote", [1, 0], 3)],
                         ["c", "b"])
        self.assertEqual([node_id for node_id, _ in graph.search("Keyword", [1, 0], 3)],
                         ["a", "keyword"])

    def test_direction_types_upsert_and_remove_nodes(self):
        graph = example_graph()
        self.assertEqual(len(graph.edges("a", direction="out")), 3)
        self.assertEqual(len(graph.edges("a", direction="in")), 2)
        self.assertEqual(len(graph.edges("a", direction="both")), 4)
        self.assertEqual(len(graph.edges("a", types={"SUPPORTS"}, direction="both")), 1)
        graph.upsert_edge("a", "b", "SUPPORTS", {"strength": 0.9})
        self.assertEqual(len(graph.edges()), 4)
        self.assertEqual(graph.edges("a", types="SUPPORTS")[0]["properties"]["strength"], 0.9)
        self.assertEqual(graph.remove_nodes("MemoryNote"), 3)
        self.assertEqual(graph.edges(), [])
        self.assertEqual([node["id"] for node in graph.nodes()], ["keyword"])

    def test_label_and_adjacency_indexes_survive_relabel_delete_and_restore(self):
        graph = example_graph()
        graph.upsert_node("topic", "Topic", {"embedding": [0, 1]})
        graph.upsert_edge("keyword", "topic", "IN_TOPIC")
        graph.upsert_edge("topic", "a", "RELATED_TO")
        graph.upsert_edge("topic", "topic", "SELF")
        self.assertEqual([node["id"] for node in graph.nodes("Topic")], ["topic"])
        graph.upsert_node("keyword", "Topic", {"embedding": [1, 0]})
        self.assertEqual(graph.nodes("Keyword"), [])
        self.assertEqual([node["id"] for node in graph.nodes("Topic")], ["keyword", "topic"])
        restored = InMemoryGraphStore.from_snapshot(graph.snapshot())
        self.assertEqual(restored.edges("topic", direction="both"),
                         graph.edges("topic", direction="both"))
        self.assertEqual(restored.remove_nodes("Topic"), 2)
        self.assertEqual(restored.nodes("Topic"), [])
        self.assertEqual(restored.edges("a", direction="in"),
                         [edge for edge in graph.edges("a", direction="in")
                          if edge["source"] in {"a", "b"}])
        self.assertEqual(len(restored.edges()), 4)
        restored.clear()
        self.assertEqual(restored.edges("a", direction="both"), [])
        self.assertEqual(restored.nodes("MemoryNote"), [])

    def test_records_clone_and_snapshot_are_isolated(self):
        graph = example_graph()
        node = graph.get_node("a")
        node["properties"]["tags"].append("external")
        graph.edges()[0]["properties"]["strength"] = -1
        self.assertEqual(graph.get_node("a")["properties"]["tags"], ["first"])
        snap = graph.snapshot()
        clone = graph.clone()
        clone.upsert_node("a", "MemoryNote", {"embedding": [0, 1]})
        clone.upsert_edge("a", "b", "SUPPORTS", {"strength": 0.1})
        self.assertEqual(graph.get_node("a")["properties"]["embedding"], [1.0, 0.0])
        self.assertEqual(graph.edges("a", types="SUPPORTS")[0]["properties"]["strength"], 0.7)
        snap["nodes"][0]["properties"]["tags"].append("snapshot")
        self.assertEqual(graph.get_node("a")["properties"]["tags"], ["first"])

    def test_save_load_and_bad_snapshots(self):
        graph = example_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "graph.json"
            graph.save(path)
            restored = InMemoryGraphStore.load(path)
            self.assertEqual(restored.snapshot(), graph.snapshot())
            self.assertEqual(json.loads(path.read_text())["score_mode"], "cosine_01")
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

        base = graph.snapshot()
        cases = []
        bad = deepcopy(base)
        bad["counts"]["nodes"] += 1
        cases.append(bad)
        bad = deepcopy(base)
        bad["edges"][0]["target"] = "missing"
        cases.append(bad)
        bad = deepcopy(base)
        bad["nodes"][0]["properties"]["embedding"] = [1, 0, 0]
        cases.append(bad)
        bad = deepcopy(base)
        bad["nodes"].append(deepcopy(bad["nodes"][0]))
        bad["counts"]["nodes"] += 1
        cases.append(bad)
        bad = deepcopy(base)
        bad["score_mode"] = "raw_cosine"
        cases.append(bad)
        bad = deepcopy(base)
        bad["nodes"][0]["id"] = []
        cases.append(bad)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                InMemoryGraphStore.from_snapshot(case)
        self.assertEqual(graph.snapshot(), base)

    def test_vector_validation_and_clear(self):
        graph = InMemoryGraphStore()
        for vector in ([0, 0], [float("nan"), 1], [1, float("inf")], [[1, 0]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                graph.upsert_node("bad", "MemoryNote", {"embedding": vector})
        graph.upsert_node("a", "MemoryNote", {"embedding": [1, 0]})
        self.assertEqual(graph.embedding_dim, 2)
        with self.assertRaisesRegex(ValueError, "dimension"):
            graph.upsert_node("b", "MemoryNote", {"embedding": [1, 0, 0]})
        with self.assertRaisesRegex(ValueError, "dimension"):
            graph.search("MemoryNote", [1, 0, 0], 1)
        with self.assertRaisesRegex(ValueError, "dangling"):
            graph.upsert_edge("a", "missing", "RELATED_TO")
        graph.clear()
        self.assertEqual(graph.nodes(), [])
        self.assertEqual(graph.edges(), [])
        self.assertEqual(graph.search("MemoryNote", [1, 0], 1), [])

    def test_finite_extreme_vectors_keep_valid_cosine(self):
        graph = InMemoryGraphStore(2)
        graph.upsert_node("tiny", "MemoryNote", {"embedding": [1e-300, 0]})
        graph.upsert_node("huge", "MemoryNote", {"embedding": [1e300, 0]})
        self.assertEqual(graph.search("MemoryNote", [1, 0], 2),
                         [("huge", 1.0), ("tiny", 1.0)])


if __name__ == "__main__":
    unittest.main()
