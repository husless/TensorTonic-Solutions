import numpy as np

def pad_sequences(
    seqs: list[list[int]],
    pad_value: int=0,
    max_len:int|None=None
):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    N = len(seqs)
    result = np.full((N, max_len), pad_value, dtype=int)
    for i, seq in enumerate(seqs):
        k = min(max_len, len(seq))
        result[i,:k] = seq[:k]

    return result