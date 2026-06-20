import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.

    X_train: (n_samples, n_features), training features matrix
    y_train: (n_samples,) labels
    X_test: (m_samples, n_features) test features matrix
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_test, d = X_test.shape

    log_likelihood = np.zeros((n_test, n_classes))
    n_train = len(y_train)

    eps = 1.0E-9
    
    for idx, c in enumerate(classes):
        # Subset training data belonging only to class 'c'
        X_c = X_train[y_train == c]
        n_c = X_c.shape[0]

        # For each class c and feature j, 
        # compute the mean and population variance:
        mu = np.mean(X_c, axis=0)
        std = np.mean((X_c-mu)**2, axis=0) + eps
        
        # Compute Log Prior: log P(y = c)
        log_prior = np.log(n_c / n_train)

        # gaussian log posterior
        g = - np.sum(0.5*np.log(2*np.pi*std) + (X_test-mu)**2/(2*std), axis=1)
        log_likelihood[:, idx] = log_prior + g

    # highest likelyhood
    idx = np.argmax(log_likelihood, axis=1)
    return classes[idx].tolist()
        