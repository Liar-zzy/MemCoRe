"""Hard bounds and full coverage for deterministic Leiden postprocessing."""

from copy import deepcopy
import unittest

import networkx as nx

from memcore.versions.v1.utils.topic_partition import bounded_keyword_partition


def assert_partition(test, graph, partition, minimum=3, maximum=15):
    members = [node for group in partition.values() for node in group]
    test.assertEqual(len(members), len(set(members)))
    test.assertEqual(set(members), set(graph.nodes))
    test.assertTrue(all(minimum <= len(group) <= maximum for group in partition.values()))


class TopicPartitionTests(unittest.TestCase):
    def test_oversized_clique_splits_into_balanced_groups(self):
        graph = nx.complete_graph([f"kw-{index:02d}" for index in range(19)])
        assignment = {4: list(graph.nodes)}
        original = deepcopy(assignment)
        partition, audit = bounded_keyword_partition(graph, assignment)
        assert_partition(self, graph, partition)
        self.assertEqual(sorted(map(len, partition.values())), [9, 10])
        self.assertEqual(audit["split_count"], 1)
        self.assertEqual(assignment, original)
        self.assertFalse(audit["pure_leiden_output"])

    def test_affinity_keeps_strong_communities_together(self):
        first = [f"a{index:02d}" for index in range(9)]
        second = [f"b{index:02d}" for index in range(9)]
        graph = nx.Graph()
        graph.add_nodes_from(first + second)
        for group in (first, second):
            for source in group:
                for target in group:
                    if source < target:
                        graph.add_edge(source, target, weight=10)
        graph.add_edge(first[-1], second[0], weight=0.1)
        partition, _ = bounded_keyword_partition(graph, {0: first + second})
        assert_partition(self, graph, partition)
        self.assertEqual({frozenset(group) for group in partition.values()},
                         {frozenset(first), frozenset(second)})

    def test_filtered_small_groups_and_isolates_are_recovered(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c", "d", "e", "f", "g", "h"])
        graph.add_edge("a", "b", weight=4)
        graph.add_edge("c", "d", weight=4)
        graph.add_edge("d", "e", weight=4)
        graph.add_edge("e", "f", weight=4)
        # 'g' and 'h' are isolates omitted by Leiden.
        partition, audit = bounded_keyword_partition(graph, {3: ["a", "b"], 7: ["c", "d", "e", "f"]})
        assert_partition(self, graph, partition)
        self.assertEqual(audit["input_unassigned_keywords"], 2)
        self.assertEqual(audit["filled_missing_keywords"], 2)
        self.assertEqual(audit["filled_isolated_keywords"], 2)
        self.assertEqual(audit["merge_count"], 1)
        self.assertIn({"c", "d", "e", "f"}, [set(group) & {"c", "d", "e", "f"}
                                             for group in partition.values()])

    def test_all_small_groups_and_remainder_one_or_two(self):
        for remainder in (1, 2):
            graph = nx.Graph()
            graph.add_nodes_from([f"k{index}" for index in range(15 + remainder)])
            partition, audit = bounded_keyword_partition(graph, {1: [f"k{index}" for index in range(15)]})
            assert_partition(self, graph, partition)
            self.assertEqual(audit["borrowed_nodes"], 3 - remainder)
        graph = nx.Graph()
        graph.add_nodes_from([f"x{index}" for index in range(7)])
        partition, audit = bounded_keyword_partition(
            graph, {2: ["x0"], 4: ["x1", "x2"]})
        assert_partition(self, graph, partition)
        self.assertEqual(audit["merge_count"], 2)

    def test_small_total_and_bad_assignments(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b"])
        partition, audit = bounded_keyword_partition(graph, {})
        self.assertEqual(partition, {})
        self.assertFalse(audit["feasible"])
        self.assertIn("fewer", audit["infeasible_reason"])
        graph.add_node("c")
        for invalid in ({1: ["a", "a"]}, {1: ["absent"]},
                        {1: ["a"], 2: ["a"]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bounded_keyword_partition(graph, invalid)

    def test_result_independent_of_input_order(self):
        names = [f"n{index:02d}" for index in range(19)]
        graph_a = nx.complete_graph(names)
        graph_b = nx.complete_graph(list(reversed(names)))
        first = bounded_keyword_partition(graph_a, {5: names})
        second = bounded_keyword_partition(graph_b, {5: list(reversed(names))})
        self.assertEqual(first, second)

    def test_float_affinity_is_independent_of_graph_and_member_order(self):
        names = [f"float-{index:02d}" for index in range(19)]
        edges = [(names[first], names[second],
                  0.1 + ((first * 7 + second * 11) % 9) * 0.07)
                 for first in range(len(names))
                 for second in range(first + 1, len(names))]
        graph_a, graph_b = nx.Graph(), nx.Graph()
        graph_a.add_nodes_from(names)
        graph_b.add_nodes_from(reversed(names))
        for source, target, weight in edges:
            graph_a.add_edge(source, target, weight=weight)
        for source, target, weight in reversed(edges):
            graph_b.add_edge(target, source, weight=weight)
        first = bounded_keyword_partition(graph_a, {8: names})
        second = bounded_keyword_partition(graph_b, {8: list(reversed(names))})
        self.assertEqual(first, second)
        assert_partition(self, graph_a, first[0])


if __name__ == "__main__":
    unittest.main()
