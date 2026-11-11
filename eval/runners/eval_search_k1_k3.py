def compute_recall(golden_dois: set[str], retrieved_dois: set[str]) -> float:
    if not golden_dois:
        return 0.0
    return len(golden_dois & retrieved_dois) / len(golden_dois)
