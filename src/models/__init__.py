"""Model definitions, one module per phase.

    encoder_classifier.py  Phase 1  — transformer encoder + mean-pool + linear
    lsa_classifier.py      Phase 2  — + Label-Specific Attention + gated fusion
    hetero_gcn.py          Phase 3  — heterogeneous GCN backbone
    graph_classifier.py    Phase 3  — Hetero-GCN + Label Correlation Matrix
"""
