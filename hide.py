"""
HIDE_k: Hyperset Individualisation with Degrees and the Empty set (order k)

Dependencies
------------
  - networkx >= 2.6
  - BisPy   >= 0.2.2    (pip install BisPy)

MIT License. 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
from bispy import compute_maximum_bisimulation, Algorithms

__all__ = [
    "HypersetAPG",
    "hide_k_certificate",
    "equal_certificates",
    "hide_k_equivalent",
]

_ORD_PREFIX = "HIDE:ORD"  # ordinal gadget nodes
_POINT_PREFIX = "HIDE:POINT"  # fresh points
_SRC_PREFIX = "HIDE:SRC"  # connectivity source


def _unique_label(base, existing):
    """Return a label not present in `existing`, derived from `base`."""
    label = base
    i = 1
    existing_set = set(existing)
    while label in existing_set:
        label = (base[0], base[1] + i)
        i += 1
    return label


class HypersetAPG:

    def __init__(self, graph, point):
        self.graph = graph
        self.point = point

    def bisimilar_to(
        self,
        other,
        *,
        algorithm=Algorithms.DovierPiazzaPolicriti,
    ):
        if self is other:
            return True

        U = nx.DiGraph()
        mapA = {n: ("A", n) for n in self.graph.nodes}
        mapB = {n: ("B", n) for n in other.graph.nodes}

        U.add_nodes_from(mapA.values())
        U.add_nodes_from(mapB.values())
        U.add_edges_from((mapA[u], mapA[v]) for (u, v) in self.graph.edges)
        U.add_edges_from((mapB[u], mapB[v]) for (u, v) in other.graph.edges)

        P0 = ("HIDE:P0", 0)
        U.add_node(P0)
        U.add_edge(P0, mapA[self.point])
        U.add_edge(P0, mapB[other.point])

        classes = compute_maximum_bisimulation(U, algorithm=algorithm)
        cls_of = {}
        for i, block in enumerate(classes):
            for x in block:
                cls_of[x] = i

        return cls_of[mapA[self.point]] == cls_of[mapB[other.point]]


# ---------------------------
# Core HIDE_k implementation
# ---------------------------


def hide_k_certificate(
    G,
    k,
    *,
    connect_if_disconnected=True,
    algorithm=Algorithms.DovierPiazzaPolicriti,
):
    UG = _as_undirected_simple(G)

    if connect_if_disconnected and not nx.is_connected(UG):
        UG = _make_connected_with_universal_source(UG)

    baseDG, ord0 = _directed_with_degree_gadget(UG)

    node_list = sorted(UG.nodes(), key=repr)
    unique = []
    counts = []

    for idx, S in enumerate(combinations(node_list, k)):
        DG_S, point = _individualise_k_by_empty_set(baseDG, ord0, S, idx)
        hyp = _bisimulation_contraction(DG_S, point, algorithm=algorithm)

        for i, u in enumerate(unique):
            if hyp.bisimilar_to(u, algorithm=algorithm):
                counts[i] += 1
                break
        else:
            unique.append(hyp)
            counts.append(1)

    return [(unique[i], counts[i]) for i in range(len(unique))]


def equal_certificates(
    cert_G,
    cert_H,
    *,
    algorithm=Algorithms.DovierPiazzaPolicriti,
):
    used_H = [False] * len(cert_H)

    for hypG, countG in cert_G:
        matched = False
        for j, (hypH, countH) in enumerate(cert_H):
            if used_H[j]:
                continue
            if hypG.bisimilar_to(hypH, algorithm=algorithm):
                if countH != countG:
                    return False
                used_H[j] = True
                matched = True
                break
        if not matched:
            return False

    return all(used_H)


def hide_k_equivalent(
    G,
    H,
    k,
    *,
    connect_if_disconnected=True,
    algorithm=Algorithms.DovierPiazzaPolicriti,
):
    cert_G = hide_k_certificate(
        G, k, connect_if_disconnected=connect_if_disconnected, algorithm=algorithm
    )
    cert_H = hide_k_certificate(
        H, k, connect_if_disconnected=connect_if_disconnected, algorithm=algorithm
    )
    return equal_certificates(cert_G, cert_H, algorithm=algorithm)


# ---------------------------
# Internal utilities
# ---------------------------


def _as_undirected_simple(
    G,
):
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes)

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        undirected_pairs = {
            tuple(sorted((u, v))) for (u, v, *_) in G.edges(keys=True) if u != v
        }
    else:
        undirected_pairs = {tuple(sorted((u, v))) for (u, v) in G.edges if u != v}

    UG.add_edges_from(undirected_pairs)
    return UG


def _make_connected_with_universal_source(UG):

    if UG.number_of_nodes() == 0 or nx.is_connected(UG):
        return UG.copy()

    UG2 = UG.copy()
    src = _unique_label((_SRC_PREFIX, 0), UG2.nodes)
    UG2.add_node(src)
    for v in UG.nodes:
        UG2.add_edge(src, v)
    return UG2


def _directed_with_degree_gadget(UG):

    DG = nx.DiGraph()
    DG.add_nodes_from(UG.nodes)
    for u, v in UG.edges:
        DG.add_edge(u, v)
        DG.add_edge(v, u)

    degrees = dict(UG.degree())
    max_deg = max(degrees.values(), default=0)

    # Ordinal chain: ORD_d -> ORD_{d-1} for d >= 1
    ord_nodes = [(_ORD_PREFIX, d) for d in range(max_deg + 1)]
    DG.add_nodes_from(ord_nodes)
    for d in range(1, max_deg + 1):
        DG.add_edge((_ORD_PREFIX, d), (_ORD_PREFIX, d - 1))

    # Connect vertex v to ORD_{deg(v)}
    for v, d in degrees.items():
        DG.add_edge(v, (_ORD_PREFIX, d))

    ord0 = (_ORD_PREFIX, 0)  # empty set
    return DG, ord0


def _individualise_k_by_empty_set(
    baseDG,
    ord0,
    S,
    idx,
):
    DG = baseDG.copy()
    point = (_POINT_PREFIX, idx)
    DG.add_node(point)
    for w in S:
        DG.add_edge(point, w)
        DG.add_edge(w, ord0)
    return DG, point


def _bisimulation_contraction(
    DG,
    point,
    *,
    algorithm=Algorithms.DovierPiazzaPolicriti,
):
    classes = compute_maximum_bisimulation(DG, algorithm=algorithm)

    class_of = {}
    for i, block in enumerate(classes):
        for x in block:
            class_of[x] = i

    Q = nx.DiGraph()
    Q.add_nodes_from(range(len(classes)))

    def _is_ordinal_nonzero(u):
        return (
            isinstance(u, tuple) and len(u) == 2 and u[0] == _ORD_PREFIX and u[1] != 0
        )

    for u, v in DG.edges:
        if _is_ordinal_nonzero(u) or _is_ordinal_nonzero(v):
            continue
        cu, cv = class_of[u], class_of[v]
        Q.add_edge(cu, cv)

    p_class = class_of[point]

    reachable = {p_class}
    reachable |= nx.descendants(Q, p_class)  # descendants excludes the source
    Q = Q.subgraph(reachable).copy()

    return HypersetAPG(Q, p_class)
