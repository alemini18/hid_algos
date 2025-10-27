import networkx as nx
from hide import hide_k_certificate
from print_cert import print_certificate

G = nx.cycle_graph(6)

# HIDE_2
cert_hide = hide_k_certificate(G, k=2)
print_certificate(cert_hide, name="HIDE_2(G)")
