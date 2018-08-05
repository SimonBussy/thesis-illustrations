import numpy as np

def generate_data(n_samples,seed): 
    np.random.seed(seed)
    sigma = 1
    cov = [[sigma, 0], [0, sigma]]  # diagonal covariance
    # generate class 1
    mean1 = [2, 2]
    x1, y1 = np.random.multivariate_normal(mean1, cov, n_samples).T
    # generate class 2
    mean2 = [4, 4]
    x2, y2 = np.random.multivariate_normal(mean2, cov, n_samples).T
    Y = np.array([0] * n_samples + [1] * n_samples)
    x_1 = np.append(x1, x2)
    x_2 = np.append(y1, y2)
    X = np.array([x_1,
                  x_2,
                  x_1 ** 2 * x_2,
                  x_2 ** 2 * x_1,
                  x_1 ** 3,
                  x_2 ** 3,
                  x_1** 3 * x_2** 3,
                  x_1 ** 3 * x_2 ** 2,
                  x_2 ** 3 * x_1 ** 2,
                  x_1 ** 2 * x_2 ** 2,
                  x_1 ** 4 * x_2 ** 2,
                  x_2 ** 4 * x_1 ** 2,
                  x_1 ** 4 * x_2 ** 3,
                  x_2 ** 4 * x_1 ** 3,
                  x_1 ** 4 * x_2 ** 4
                 ]).T
    
    # Shuffle arrays
    perm = np.random.permutation(2 * n_samples)
    X = X[perm]
    Y = Y[perm]
    return X, Y, mean1, mean2