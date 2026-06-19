import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    pred = np.asarray(predictions)
    n_class, n_sample = pred.shape

    majority = [0]*n_sample

    for i in range(n_sample):
        votes = dict()
        for t in range(n_class):
            label = pred[t, i]
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1
        highest = max(votes.values())
        majority[i] = min(label for label, cnt in votes.items() if cnt==highest)

    return majority
        

    