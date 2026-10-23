"""A small, serializable graph with exact in-process vector search.

Nodes and edges use application IDs; no database-generated identifier is needed.
The score is ``(1 + cosine_similarity) / 2`` so callers can use the original
gate's zero-to-one similarity range.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA_VERSION = 1
SCORE_MODE = "cosine_01"


def _name(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _json_value(value):
    """Copy supported values into plain JSON data, rejecting lossy keys/NaN."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("properties must contain finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("property keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported property value: {type(value).__name__}")


def _properties(value):
    if not isinstance(value, dict):
        raise ValueError("properties must be a dictionary")
    return _json_value(value)


def _vector(value, expected_dim=None):
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("embedding must be a numeric vector") from None
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a nonempty one-dimensional vector")
    if expected_dim is not None and vector.size != expected_dim:
        raise ValueError(f"embedding dimension {vector.size} != {expected_dim}")
    if not np.isfinite(vector).all():
        raise ValueError("embedding must contain finite numbers")
    if not np.any(vector != 0):
        raise ValueError("embedding must be nonzero")
    return vector


def _unit(vector):
    """Normalize without overflow or underflow for finite nonzero vectors."""
    scaled = vector / np.max(np.abs(vector))
    return scaled / np.linalg.norm(scaled)


class InMemoryGraphStore:
    """Directed, typed graph whose node embeddings share one dimension.

    Upserts replace the complete property dictionary for an existing node or
    edge; callers that need a partial update should read, edit, and upsert.
    """

    def __init__(self, embedding_dim=None):
        if embedding_dim is not None and (type(embedding_dim) is not int or embedding_dim <= 0):
            raise ValueError("embedding_dim must be a positive integer or None")
        self.embedding_dim = embedding_dim
        self._nodes = {}
        self._edges = {}
        self._label_nodes = {}
        self._out_edges = {}
        self._in_edges = {}
        self._label_matrices = {}

    def upsert_node(self, node_id: str, label: str, properties: dict):
        _name(node_id, "node_id")
        _name(label, "label")
        props = _properties(properties)
        if "embedding" in props:
            vector = _vector(props["embedding"], self.embedding_dim)
            props["embedding"] = vector.tolist()
            if self.embedding_dim is None:
                self.embedding_dim = vector.size
        previous = self._nodes.get(node_id)
        self._nodes[node_id] = {"id": node_id, "label": label, "properties": props}
        if previous is None:
            self._label_nodes.setdefault(label, set()).add(node_id)
        elif previous["label"] != label:
            old_ids = self._label_nodes[previous["label"]]
            old_ids.remove(node_id)
            if not old_ids:
                del self._label_nodes[previous["label"]]
            self._label_nodes.setdefault(label, set()).add(node_id)
        if previous is None or (previous["label"] != label or
                                previous["properties"].get("embedding") != props.get("embedding")):
            self._label_matrices.pop(label, None)
            if previous is not None:
                self._label_matrices.pop(previous["label"], None)

    def get_node(self, node_id: str):
        record = self._nodes.get(node_id)
        return deepcopy(record) if record is not None else None

    def nodes(self, label=None):
        if label is not None:
            _name(label, "label")
        ids = self._nodes if label is None else self._label_nodes.get(label, ())
        return [deepcopy(self._nodes[node_id]) for node_id in sorted(ids)]

    def upsert_edge(self, source: str, target: str, edge_type: str, properties=None):
        _name(source, "source")
        _name(target, "target")
        _name(edge_type, "edge_type")
        if source not in self._nodes or target not in self._nodes:
            raise ValueError(f"edge has dangling endpoint: {source} -> {target}")
        props = _properties({} if properties is None else properties)
        key = (source, target, edge_type)
        if key not in self._edges:
            self._out_edges.setdefault(source, set()).add(key)
            self._in_edges.setdefault(target, set()).add(key)
        self._edges[key] = {"source": source, "target": target,
                            "type": edge_type, "properties": props}

    def edges(self, node_id=None, types=None, direction="out"):
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be out, in, or both")
        if node_id is not None:
            _name(node_id, "node_id")
        if types is not None:
            types = {types} if isinstance(types, str) else set(types)
        if node_id is None:
            keys = self._edges
        elif direction == "out":
            keys = self._out_edges.get(node_id, ())
        elif direction == "in":
            keys = self._in_edges.get(node_id, ())
        else:
            keys = self._out_edges.get(node_id, set()) | self._in_edges.get(node_id, set())
        records = []
        for key in sorted(keys):
            edge = self._edges[key]
            if types is not None and edge["type"] not in types:
                continue
            records.append(deepcopy(edge))
        return records

    def remove_nodes(self, label):
        _name(label, "label")
        removed = set(self._label_nodes.get(label, ()))
        incident = set()
        for node_id in removed:
            incident.update(self._out_edges.get(node_id, ()))
            incident.update(self._in_edges.get(node_id, ()))
        for key in incident:
            source, target, _ = key
            del self._edges[key]
            self._out_edges[source].remove(key)
            self._in_edges[target].remove(key)
            if not self._out_edges[source]:
                del self._out_edges[source]
            if not self._in_edges[target]:
                del self._in_edges[target]
        for node_id in removed:
            del self._nodes[node_id]
        self._label_nodes.pop(label, None)
        self._label_matrices.pop(label, None)
        return len(removed)

    def clear(self):
        self._nodes.clear()
        self._edges.clear()
        self._label_nodes.clear()
        self._out_edges.clear()
        self._in_edges.clear()
        self._label_matrices.clear()

    def _matrix(self, label):
        if label not in self._label_matrices:
            items = [(node_id, node["properties"]["embedding"])
                     for node_id, node in sorted(self._nodes.items())
                     if node["label"] == label and "embedding" in node["properties"]]
            if items:
                ids = [item[0] for item in items]
                matrix = np.asarray([item[1] for item in items], dtype=np.float64)
                matrix = np.stack([_unit(row) for row in matrix])
            else:
                ids = []
                matrix = np.empty((0, self.embedding_dim or 0), dtype=np.float64)
            self._label_matrices[label] = (ids, matrix)
        return self._label_matrices[label]

    def search(self, label, query_vector, k, exclude_ids=()):
        _name(label, "label")
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        query = _vector(query_vector, self.embedding_dim)
        ids, matrix = self._matrix(label)
        if not ids or k == 0:
            return []
        query = _unit(query)
        scores = np.clip((1.0 + matrix @ query) / 2.0, 0.0, 1.0)
        excluded = set(exclude_ids)
        ranked = [(node_id, float(score)) for node_id, score in zip(ids, scores)
                  if node_id not in excluded]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def snapshot(self):
        nodes = self.nodes()
        edges = self.edges()
        return {"schema_version": SCHEMA_VERSION,
                "embedding_dim": self.embedding_dim,
                "score_mode": SCORE_MODE,
                "counts": {"nodes": len(nodes), "edges": len(edges)},
                "nodes": nodes, "edges": edges}

    @classmethod
    def from_snapshot(cls, data):
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported graph snapshot schema")
        if data.get("score_mode") != SCORE_MODE:
            raise ValueError("unsupported graph snapshot score_mode")
        nodes, edges, counts = data.get("nodes"), data.get("edges"), data.get("counts")
        if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(counts, dict):
            raise ValueError("invalid graph snapshot structure")
        if counts != {"nodes": len(nodes), "edges": len(edges)}:
            raise ValueError("graph snapshot counts do not match records")
        store = cls(data.get("embedding_dim"))
        for node in nodes:
            if not isinstance(node, dict) or set(node) != {"id", "label", "properties"}:
                raise ValueError("invalid graph snapshot node")
            _name(node["id"], "node_id")
            if node["id"] in store._nodes:
                raise ValueError(f"duplicate graph snapshot node: {node['id']}")
            store.upsert_node(node["id"], node["label"], node["properties"])
        if store.embedding_dim != data.get("embedding_dim"):
            raise ValueError("graph snapshot embedding_dim does not match vectors")
        for edge in edges:
            if not isinstance(edge, dict) or set(edge) != {"source", "target", "type", "properties"}:
                raise ValueError("invalid graph snapshot edge")
            _name(edge["source"], "source")
            _name(edge["target"], "target")
            _name(edge["type"], "edge_type")
            key = (edge["source"], edge["target"], edge["type"])
            if key in store._edges:
                raise ValueError(f"duplicate graph snapshot edge: {key}")
            store.upsert_edge(edge["source"], edge["target"], edge["type"], edge["properties"])
        return store

    def clone(self):
        return self.from_snapshot(self.snapshot())

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.snapshot(), stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_snapshot(json.load(stream))
