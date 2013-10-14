#! /usr/bin/python

import numpy as np

    

def linear_regression (x1, y1, x2, y2, training_size):
    # generate a line to use as the target function 
    M = np.matrix([[x1, x2], [y1, y2]])
    is_sing = np.linalg.det(M) == 0

    if is_sing:
        coeffs = np.array([1, -x1/y1])
    else:
        coeffs = np.array([1, 1]).dot(M.I)

    coeffs = np.array(coeffs).flatten()
#    print coeffs

    # create a training data matrix. Rows are (x, y)'
    X = np.matrix(np.random.uniform(-1, 1, (training_size, 2)))
    ones = np.matrix([[1]] * training_size)
    X = np.concatenate((ones, X), axis=1)

    y = np.array(training_size * [-1])
    ones_idx = np.array(coeffs[0] * X[:,1] + coeffs[1] * X[:,2] < 1).flatten()
    y[ones_idx] = 1
    
    # compute X dagger
    Xd = (X.T.dot(X)).I.dot(X.T)

    w = Xd.dot(y)
    g = np.sign(w.dot(X.T))
    
    n_misses = np.sum(g != y)

    prob = float(n_misses) / float(training_size)
#    print "Iter: %d, prob: %f" % (it, prob)
    return (w, coeffs, prob) 





def linear_regression_test(w, coeffs, test_size):
    # create a data matrix. Rows are (x, y)'
    X = np.matrix(np.random.uniform(-1, 1, (test_size, 2)))
    ones = np.matrix([[1]] * test_size)
    X = np.concatenate((ones, X), axis=1)
    
    f = np.array(test_size * [-1])
    ones_idx = np.array(coeffs[0] * X[:,1] + coeffs[1] * X[:,2] < 1).flatten()
    f[ones_idx] = 1

    g = np.sign(w.dot(X.T))
    
    n_misses = np.sum(g != f)
    prob = float(n_misses) / float(test_size)
    return (prob)
    


def main():
    x1 = -0.5
    y1 = -0.5
    x2 = 0.2
    y2 = 0.5
    training_size = 10
    it, prob = linear_regression(x1, y1, x2, y2, training_size)


if __name__ == "__main__":
    main()
