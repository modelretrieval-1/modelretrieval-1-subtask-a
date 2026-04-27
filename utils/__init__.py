import numpy as np


def dcg_at_k(r: list, k: int) -> float:
    """Compute DCG@k

    Args:
        r (list): Relevance scores in rank order (first element is the first item)
        k (int): Rank position to calculate DCG up to

    Example:
        r = [3, 2, 3, 0, 1, 2] # Relevance scores for the ranked items
        k = 5 # Calcualte DCG at rank 5
        dcg_at_k(r, k)

    Returns:
        float: DCG@k value
    """
    r = r[:k]
    return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(r))


def ndcg_at_k(r: list, k: int) -> float:
    """Compute nDCG@k

    Args:
        r (list): Relevance scores in rank order (first element is the first item)
        k (int): Rank position to calculate nDCG up to

    Example:
        r = [3, 2, 3, 0, 1, 2] # Relevance scores for the ranked items
        k = 5 # Calcualte nDCG at rank 5
        ndcg_at_k(r, k)

    Returns:
        float: nDCG@k value
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max
