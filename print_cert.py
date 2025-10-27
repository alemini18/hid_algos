def print_certificate(cert, name="certificate"):
    """
    Pretty-print a HIDE_k certificate.

    Parameters
    ----------
    cert : list of (hyp, multiplicity)
        hyp is HypersetAPG (HIDE_k).
    name : str
        Label for the certificate (optional).
    """
    print(f"=== {name} ===")
    if not cert:
        print("(empty)")
        return

    for idx, (hyp, mult) in enumerate(cert, start=1):
        print(f"\nHyperset {idx}  (multiplicity ×{mult})")
        print("Nodes:", list(hyp.graph.nodes))
        print("Edges:", list(hyp.graph.edges))
        print("Point:", hyp.point)
        if hasattr(hyp, "chosen_classes") and getattr(hyp, "chosen_classes", ()):
            print("Chosen:", hyp.chosen_classes)
