"""
Certificate drawing helpers for HIDE_k
===============================================

These routines plot the multiset of hypersets returned by:
  - hide_k_certificate(...)   -> [(HypersetAPG, multiplicity), ...]

They visualize each hyperset's *bisimulation contraction* (APG) and annotate:
  - point (distinguished node of the APG)
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Hashable, Optional, Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import numpy as np


# ---------- Public API --------------------------------------------------------


def draw_certificate_grid(
    cert: Sequence[Tuple[Any, int]],
    *,
    title: Optional[str] = None,
    max_cols: int = 4,
    figsize_per_cell: Tuple[float, float] = (3.2, 3.0),
    layout: str = "kamada_kawai",  # "kamada_kawai" | "spring" | "shell" | "random"
    sort: str = "multiplicity_desc",  # "multiplicity_desc" | "size_desc" | "none"
    node_size: int = 450,
    font_size: int = 8,
    show_labels: bool = True,  # label only point(not all nodes)
    show_legend: bool = True,
    arrows: bool = True,
    savepath: Optional[str] = None,
):
    """
    Draw a certificate (HIDE_k) as a grid of APGs.

    Parameters
    ----------
    cert : sequence of (hyp, multiplicity)
        hyp is HypersetAPG (HIDE_k)
        We only rely on these attributes if present:
          - hyp.graph : nx.DiGraph (bisimulation contraction)
          - hyp.point : hashable node id within hyp.graph
    title : str, optional
        Figure title.
    max_cols : int
        Maximum number of columns in the grid.
    figsize_per_cell : (w, h)
        Size in inches per cell.
    layout : str
        Node layout algorithm for each APG.
    sort : str
        Reorder certificate items before plotting.
    node_size : int
        Base node size for drawing.
    font_size : int
        Font size for labels and titles.
    show_labels : bool
        If True, annotate point/chosen near their nodes.
    show_legend : bool
        If True, add a legend (only if choosen present in any item).
    arrows : bool
        If True, draw directed edges with arrows.
    savepath : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    (fig, axes)
    """
    if len(cert) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(figsize_per_cell[0], figsize_per_cell[1]))
        ax.axis("off")
        ax.set_title("Empty certificate", fontsize=font_size)
        if title:
            fig.suptitle(title, fontsize=font_size + 2)
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=180)
        return fig, [ax]

    items = list(cert)
    if sort == "multiplicity_desc":
        items.sort(
            key=lambda t: (
                -t[1],
                -t[0].graph.number_of_nodes(),
                -t[0].graph.number_of_edges(),
            )
        )
    elif sort == "size_desc":
        items.sort(
            key=lambda t: (
                -t[0].graph.number_of_nodes(),
                -t[0].graph.number_of_edges(),
                -t[1],
            )
        )
    # else: keep order

    n = len(items)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig_w = max(1, cols) * figsize_per_cell[0]
    fig_h = max(1, rows) * figsize_per_cell[1]

    # fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    # if not isinstance(axes, (list, tuple)):
    #     axes = [axes]
    # axes = [
    #     ax
    #     for row in (axes if isinstance(axes, list) else [axes])
    #     for ax in (row if hasattr(row, "__len__") else [row])
    # ]
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    any_chosen = any(
        hasattr(hyp, "chosen_classes") and len(getattr(hyp, "chosen_classes", ())) > 0
        for hyp, _ in items
    )

    for i, (hyp, mult) in enumerate(items):
        ax = axes[i]
        G = hyp.graph
        point = getattr(hyp, "point", None)
        chosen = (
            tuple(getattr(hyp, "chosen_classes", ()))
            if hasattr(hyp, "chosen_classes")
            else ()
        )

        _draw_apg(
            ax,
            G,
            point=point,
            chosen=chosen,
            layout=layout,
            node_size=node_size,
            font_size=font_size,
            show_labels=show_labels,
            arrows=arrows,
        )
        ax.set_title(f"x{mult}", fontsize=font_size)

    # Hide any spare axes
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=font_size + 3)

    if show_legend and (any_chosen):
        handles = [
            Line2D(
                [0],
                [0],
                marker="p",
                color="w",
                markerfacecolor="#1f77b4",
                markersize=10,
                label="point",
                markeredgecolor="k",
            ),
        ]
        if any_chosen:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="#2ca02c",
                    markersize=9,
                    label="chosen v_i",
                    markeredgecolor="k",
                )
            )
        fig.legend(
            handles=handles, loc="upper right", frameon=False, fontsize=font_size
        )

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=180)
    return fig, axes


def draw_hide_certificate(cert: Sequence[Tuple[Any, int]], **kwargs):
    """Wrapper specialised for HIDE_k certificates."""
    return draw_certificate_grid(
        cert, title=kwargs.pop("title", "HIDE_k certificate"), **kwargs
    )


# ---------- Internals ---------------------------------------------------------


def _layout(G: nx.DiGraph, which: str) -> dict:
    which = (which or "").lower()
    if which == "spring":
        return nx.spring_layout(G, seed=0)
    if which == "shell":
        return nx.shell_layout(G)
    if which == "random":
        return nx.random_layout(G)
    # default:
    return nx.kamada_kawai_layout(G)


def _draw_apg(
    ax,
    G: nx.DiGraph,
    *,
    point: Optional[Hashable],
    chosen: Sequence[Hashable] = (),
    originals: Sequence[Hashable] = (),
    layout: str = "kamada_kawai",
    node_size: int = 450,
    font_size: int = 8,
    show_labels: bool = True,
    arrows: bool = True,
):
    """Draw a single APG with role-aware styling."""
    ax.axis("off")
    pos = _layout(G, layout)

    # Draw edges first (directed)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=arrows,
        arrowstyle="-|>",
        arrowsize=12,
        width=1.2,
        connectionstyle="arc3,rad=0.06",
    )

    # Node role sets
    nodes_all = set(G.nodes)
    set_chosen = set(chosen) & nodes_all  # avoid overlaps in styling
    set_point = {point} & nodes_all if point is not None else set()
    set_rest = set(G.nodes) - (set_chosen | set_point)

    # Base nodes
    if set_rest:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sorted(set_rest),
            node_size=node_size,
            node_color="#cccccc",
            edgecolors="k",
            linewidths=0.8,
            ax=ax,
        )

    # Chosen: triangles
    if set_chosen:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(set_chosen),
            node_shape="^",
            node_size=node_size,
            node_color="#2ca02c",
            edgecolors="k",
            linewidths=0.9,
            ax=ax,
        )

    # Point: pentagon
    if set_point:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(set_point),
            node_shape="p",
            node_size=int(node_size * 1.1),
            node_color="#1f77b4",
            edgecolors="k",
            linewidths=1.0,
            ax=ax,
        )

    # Self-loops indicator (small ring)
    for n in G.nodes:
        if G.has_edge(n, n):
            (x, y) = pos[n]
            circ = plt.Circle((x, y), radius=0.04, fill=False, lw=1.0, ec="k")
            ax.add_patch(circ)

    # Minimal labels (only roles)
    if show_labels:
        labels = {}
        if set_point:
            for n in set_point:
                labels[n] = "p"
        # Number chosen similarly
        if chosen:
            idxc = {n: i for i, n in enumerate(chosen, start=1)}
            for n in set_chosen:
                labels[n] = f"v{idxc.get(n, '?')}"

        if labels:
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)
