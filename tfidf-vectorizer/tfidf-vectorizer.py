import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    n_docs = len(documents)
    counters = [
        Counter(tok.lower() for tok in doc.split())
        for doc in documents
    ]
    vocab = set()
    for c in counters:
        vocab |= c.keys()
    
    vocabulary = sorted(vocab)
    n_vocab = len(vocabulary)
    tifidf = np.zeros((n_docs, n_vocab))
    for t, tok in enumerate(vocabulary):
        # documents containing term tok
        df_t = sum(tok in c for c in counters)
        idf_t = math.log(n_docs / df_t)

        for d, c in enumerate(counters):
            occ = c.get(tok, 0)
            if occ > 0:
                tifidf[d, t] = idf_t * occ / c.total()

    return tifidf, vocabulary
    
    