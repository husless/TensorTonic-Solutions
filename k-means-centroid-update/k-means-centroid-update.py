def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    d = len(points[0])
    centroids = [
        [0]*d for _ in range(k)
    ]
    count = [0 for _ in range(k)]
    for p, assign in zip(points, assignments):
        for i in range(d):
            centroids[assign][i] += p[i]

        count[assign] += 1

    for i, cc in enumerate(count):
        if cc > 0:
            for j in range(d):
                centroids[i][j] /= cc

    return centroids
        