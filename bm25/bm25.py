import numpy as np
from collections import Counter
import math

def bm25_score(
    query_tokens: list[str],
    docs: list[list[str]],
    k1: float=1.2,
    b:float=0.75
):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([])
    Q = len(query_tokens)

    counters = [Counter(doc) for doc in docs]

    # idf
    dft = (
        sum(token in c for c in counters)
        for token in query_tokens
    )
    idf = [
        math.log(1 + (N - df_t + 0.5)/(df_t + 0.5))
        for df_t in dft
    ]

    avgdl = sum(c.total() for c in counters) / N
    bm25 = np.zeros(N)
    c1 = 1 + k1
    for d, c in enumerate(counters):
        tf = (c.get(tok, 0) for tok in query_tokens)
        c2 = k1 * (1 - b + b * c.total()/avgdl)
        bm25[d] = sum(idf_t*tf_t*c1/(tf_t+ c2) for idf_t, tf_t in zip(idf, tf))

    return bm25