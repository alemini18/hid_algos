import networkx as nx
from hide import hide_k_certificate
from draw_cert import draw_hide_certificate

# Example graphs
G = nx.Graph([(0, 1), (0, 3), (1, 2), (2, 5), (2, 3), (3, 4), (4, 5)])
G = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
# G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (0, 4), (1, 3)])

G = nx.Graph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)])


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
        for dx, dy in generators:
            neighbor = ((a + dx) % n, (b + dy) % n)
            G.add_edge((a, b), neighbor)

    return G


H = shrikhande_graph()

# HIDE_k
cert_hide = hide_k_certificate(G, k=2)

draw_hide_certificate(
    cert_hide,
    max_cols=3,
    sort="multiplicity_desc",
    title="HIDE_2(G)",
    savepath="hide.png",
)
