"""Deterministic cardinality repair for final Leiden keyword assignments.

This is a constraint postprocessor, not another Leiden optimization. It keeps
already valid communities, splits oversized ones by graph affinity, and places
undersized or unassigned keywords without losing graph coverage.
"""

import math

import networkx as nx


def _weight(graph, first, second):
    data = graph.get_edge_data(first, second)
    return float(data.get("weight", 1.0)) if data is not None else 0.0


def _affinity(graph, node, members):
    return math.fsum(_weight(graph, node, member) for member in sorted(members))


def _balanced_sizes(total, minimum, maximum):
    count = math.ceil(total / maximum)
    if count > total // minimum:
        raise ValueError(f"cannot partition {total} keywords into groups of {minimum}..{maximum}")
    base, extra = divmod(total, count)
    return [base + int(index < extra) for index in range(count)]


def _split_by_affinity(graph, members, sizes):
    """Choose a high-degree seed, then grow each balanced block by affinity."""
    remaining = set(members)
    groups = []
    for size in sizes:
        seed = min(remaining, key=lambda node: (
            -_affinity(graph, node, remaining - {node}), node))
        group = [seed]
        remaining.remove(seed)
        while len(group) < size:
            chosen = min(remaining, key=lambda node: (-_affinity(graph, node, group), node))
            group.append(chosen)
            remaining.remove(chosen)
        groups.append(group)
    return groups


def bounded_keyword_partition(graph: nx.Graph, clusters: dict[int, list[str]],
                              min_size: int = 3, max_size: int = 15
                              ) -> tuple[dict[int, list[str]], dict]:
    """Repair final Leiden clusters into a full, bounded keyword partition.

    Input clusters may omit isolated graph nodes. Unknown or repeated keywords
    are errors. Existing valid clusters keep their IDs; members are sorted in
    the result. Edge weights guide splits and attachment, with names breaking
    all ties. The input graph and cluster mapping are never mutated.
    """
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError("graph must be an undirected networkx.Graph")
    if (type(min_size) is not int or type(max_size) is not int or min_size < 1
            or max_size < min_size):
        raise ValueError("invalid cluster size bounds")
    if any(not isinstance(node, str) or not node for node in graph.nodes):
        raise ValueError("all keyword graph nodes must be nonempty strings")
    for first, second, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError(f"invalid graph edge weight: {first} -- {second}")
    if not isinstance(clusters, dict):
        raise ValueError("clusters must be a dictionary")
    assigned = set()
    input_sizes = {}
    for cluster_id, members in clusters.items():
        if type(cluster_id) is not int or not isinstance(members, list) or not members:
            raise ValueError("cluster IDs must be integers with nonempty member lists")
        input_sizes[cluster_id] = len(members)
        for node in members:
            if not isinstance(node, str) or node not in graph:
                raise ValueError(f"unknown keyword in Leiden assignment: {node!r}")
            if node in assigned:
                raise ValueError(f"duplicate keyword in Leiden assignment: {node}")
            assigned.add(node)
    missing = sorted(set(graph.nodes) - assigned)
    total = graph.number_of_nodes()
    audit = {
        "strategy": "deterministic_affinity_cardinality_postprocess_v1",
        "pure_leiden_output": False,
        "min_size": min_size, "max_size": max_size,
        "total_keywords": total, "input_cluster_sizes": dict(sorted(input_sizes.items())),
        "input_unassigned_keywords": len(missing),
        "input_isolated_unassigned_keywords": sum(graph.degree(node) == 0 for node in missing),
        "filled_missing_keywords": 0, "filled_isolated_keywords": 0,
        "split_count": 0, "merge_count": 0,
        "merge_count_definition": "undersized input clusters redistributed into bounded output",
        "borrowed_nodes": 0, "output_cluster_sizes": {}, "covered_keywords": 0,
        "feasible": True,
    }
    if total < min_size:
        audit["feasible"] = False
        audit["infeasible_reason"] = "fewer graph keywords than min_size"
        return {}, audit
    _balanced_sizes(total, min_size, max_size)  # fail explicitly for impossible custom bounds

    result = {}
    pending = list(missing)
    next_id = max(clusters, default=-1) + 1
    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        if min_size <= len(members) <= max_size:
            result[cluster_id] = set(members)
        elif len(members) > max_size:
            pieces = _split_by_affinity(graph, members,
                                        _balanced_sizes(len(members), min_size, max_size))
            result[cluster_id] = set(pieces[0])
            for piece in pieces[1:]:
                result[next_id] = set(piece)
                next_id += 1
            audit["split_count"] += 1
        else:
            pending.extend(members)
            audit["merge_count"] += 1

    # Attach as many pending keywords as possible while leaving either zero or
    # at least min_size for new balanced clusters.
    pending.sort(key=lambda node: (-_affinity(graph, node, graph.nodes - {node}), node))
    capacity = sum(max_size - len(members) for members in result.values())
    attach_count = (len(pending) if capacity >= len(pending)
                    else min(capacity, max(0, len(pending) - min_size)))
    for node in pending[:attach_count]:
        choices = [cluster_id for cluster_id, members in result.items()
                   if len(members) < max_size]
        selected = min(choices, key=lambda cluster_id: (
            -_affinity(graph, node, result[cluster_id]), len(result[cluster_id]), cluster_id))
        result[selected].add(node)
    remaining = pending[attach_count:]
    if len(remaining) >= min_size:
        pieces = _split_by_affinity(graph, remaining,
                                    _balanced_sizes(len(remaining), min_size, max_size))
        for piece in pieces:
            result[next_id] = set(piece)
            next_id += 1
        remaining = []
    elif remaining:
        # Full clusters can force a remainder of one or two. Borrow members
        # without taking any donor below its lower bound.
        needed = min_size - len(remaining)
        for _ in range(needed):
            donors = [(cluster_id, node) for cluster_id, members in result.items()
                      if len(members) > min_size for node in members]
            if not donors:
                raise ValueError("cannot borrow enough keywords for a bounded partition")
            cluster_id, node = min(donors, key=lambda item: (
                -_affinity(graph, item[1], remaining),
                _affinity(graph, item[1], result[item[0]] - {item[1]}),
                item[0], item[1]))
            result[cluster_id].remove(node)
            remaining.append(node)
            audit["borrowed_nodes"] += 1
        result[next_id] = set(remaining)

    output = {cluster_id: sorted(members) for cluster_id, members in sorted(result.items())}
    flattened = [node for members in output.values() for node in members]
    if (set(flattened) != set(graph.nodes) or len(flattened) != total
            or any(not min_size <= len(members) <= max_size for members in output.values())):
        raise AssertionError("bounded partition failed coverage or size invariant")
    audit["filled_missing_keywords"] = len(missing)
    audit["filled_isolated_keywords"] = audit["input_isolated_unassigned_keywords"]
    audit["output_cluster_sizes"] = {cluster_id: len(members)
                                     for cluster_id, members in output.items()}
    audit["covered_keywords"] = total
    return output, audit
