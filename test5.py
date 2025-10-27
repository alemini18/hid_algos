import networkx as nx
from bispy import compute_maximum_bisimulation, Algorithms


def shrikhande_graph():
    """
    Construct the Shrikhande graph:
    - Vertex set: Z4 x Z4
    - Adjacency: (a,b) connected to (c,d) if (a-c, b-d) ∈ S
      where S = {(±1,0), (0,±1), (±1,±1)} in Z4 x Z4
    """
    G = nx.DiGraph()
    n = 4
    vertices = [(i, j) for i in range(n) for j in range(n)]
    G.add_nodes_from(vertices)

    generators = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]

    for a, b in vertices:
        for dx, dy in generators:
            neighbor = ((a + dx) % n, (b + dy) % n)
            G.add_edge((a, b), neighbor)
            G.add_edge(neighbor, (a, b))

    return G


G = shrikhande_graph()

G.add_node("special")
G.add_node("null")
G.add_edge("special", (0, 1))
G.add_edge("special", (1, 0))
G.add_edge((0, 1), "null")
G.add_edge((1, 0), "null")

""" for u, v in G.edges:
    print(u, v) """


intial_partition = [tuple([i for i in range(0, 16)]), (16,), (17,)]

classes = compute_maximum_bisimulation(
    G, initial_partition=intial_partition, algorithm=Algorithms.DovierPiazzaPolicriti
)

class_of = {}
for i, block in enumerate(classes):
    for x in block:
        class_of[x] = i
Q = nx.DiGraph()
Q.add_nodes_from(range(len(classes)))
for u, v in G.edges:
    cu, cv = class_of[u], class_of[v]
    Q.add_edge(cu, cv)

p_class = class_of["special"]

reachable = {p_class}
reachable |= nx.descendants(Q, p_class)
Q = Q.subgraph(reachable).copy()

""" for u, v in Q.edges:
    print(u, v) """
