import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.

    X_train: (n_train, d), training features
    y_train: (n_train,), taining labels
    X_test: (n_test, d), test features
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_test, d = X_test.shape

    log_likelihood = np.zeros((n_test, n_classes))
    n_train = len(y_train)

    for idx, c in enumerate(classes):
        # Subset training data belonging only to class 'c'
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]

        # Compute Log Prior: log P(y = c)
        log_prior = np.log(n_c / n_train)
        
        # Compute Log Likelihood parameters with Laplace smoothing (+1)
        # Count occurrences of feature j in class c
        feature_counts = np.sum(X_c, axis=0)
        
        # theta_cj = P(x_j = 1 | y = c)
        theta = (feature_counts + 1) / (n_c + 2)
        
        # 4. Compute posterior score for all test samples for class 'c'
        # Formula: X_test * log(theta) + (1 - X_test) * log(1 - theta)
        # Using dot products for optimal execution speed
        log_likelihood[:, idx] = log_prior + np.dot(X_test, np.log(theta)) + np.dot(1 - X_test, np.log(1-theta))
        
    return log_likelihood