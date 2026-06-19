def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    D = len(points[0])
    d_squared = (
        [sum((p[d]-c[d])**2 for d in range(D)) for c in centroids]
        for p in points
    )
    ass = [0 for _ in points]
    for i, dists in enumerate(d_squared):
        best_d = float('inf')
        best_idx = 0
        for j, d in enumerate(dists):
            if d < best_d:
                best_idx, best_d = j, d
        ass[i] = best_idx

    return ass