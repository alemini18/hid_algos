import networkx as nx
from hide_old import hide_k_certificate, hide_k_equivalent


def shrikhande_graph():
    """
    Construct the Shrikhande graph:
    - Vertex set: Z4 x Z4
    - Adjacency: (a,b) connected to (c,d) if (a-c, b-d) ∈ S
      where S = {(±1,0), (0,±1), (±1,±1)} in Z4 x Z4
    """
    G = nx.Graph()
    n = 4
    # Vertices are pairs (i,j) mod 4
    vertices = [(i, j) for i in range(n) for j in range(n)]
    G.add_nodes_from(vertices)

    # Generating set for adjacency (mod 4 arithmetic)
    generators = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]

    for a, b in vertices:
        for c, d in vertices:
            if ((a - c + n) % n, (b - d + n) % n) in generators:
                G.add_edge((a, b), (c, d))

    return G


# G = nx.cycle_graph(6)  # C6
# H = nx.path_graph(6)  # P6

G = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
H = shrikhande_graph()

print("HIDE_1 equal?", hide_k_equivalent(G, H, k=1))  # HID vs HID
# print("HIDE_2 equal?", hide_k_equivalent(G, H, k=2))

cert = hide_k_certificate(G, k=1)
for hyp, mult in cert:
    print(
        "Multiplicity:",
        mult,
        "APG nodes:",
        hyp.graph.number_of_nodes(),
        "APG edges:",
        hyp.graph.number_of_edges(),
    )
