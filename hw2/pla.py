#! /usr/bin/python

import random
import numpy as np

def calc_prob (coeffs, w):
    n_test_points = 1000
    f_test = np.array(n_test_points * [-1])

    test_data = np.array(np.random.uniform(-1,1,(2, n_test_points)))
    ones = np.array(n_test_points * [[1]])
    test_data = np.concatenate((ones.T, test_data), axis=0)

#    f_ones_idx = test_data[2,:] < (slope * test_data[1,:] + icept)
    f_ones_idx = np.array(coeffs[0] * test_data[1,:] + coeffs[1] * test_data[2,:] < 1).flatten()
    f_test[f_ones_idx] = 1
 
    g_test = np.array(np.sign(w.dot(test_data)))
    g_test[g_test == 0] = 1
    g_test = g_test.flatten()

    n_misses = sum(g_test != f_test)
    print n_misses
    prob = float(n_misses) / float(n_test_points)

    return prob




def perceptron (x1, y1, x2, y2, training_size, w):
    
    M = np.matrix([[x1, x2], [y1, y2]])
    is_sing = np.linalg.det(M) == 0

    if is_sing:
        coeffs = np.array([1, -x1/y1])
    else:
        coeffs = np.array([1, 1]).dot(M.I)

    coeffs = np.array(coeffs).flatten()

    # create a 3xN data array. columns are (1, x, y)'
    data = np.array(np.random.uniform(-1, 1, (2, training_size)))
    ones = np.array(training_size * [[1]])
    data = np.concatenate((ones.T, data), axis=0)

    # f is the target function
    # g is the current hypothesis
    # w is the weight vector
    f = np.array(training_size * [-1])
    g = np.array(training_size * [-1])
    ones_idx = np.array(coeffs[0] * data[:,1] + coeffs[1] * data[:,2] < 1).flatten()
    f[ones_idx] = 1

    maxIter = 100
    it = 0
    prob = 0
    while (np.any(f != g)) and (it < maxIter):
        it += 1

        # pick a random misclassified index
        miss_idx = np.argwhere(g != f)

        if len(miss_idx > 0):
            change_idx = random.sample(miss_idx, 1)
            change_idx = change_idx[0]
        
            w = w + np.squeeze(f[change_idx]*data[:,change_idx])
            g = np.sign(np.dot(w, data))

            g[g == 0] = 1 # account for any possible "ties"
        
        prob = calc_prob(coeffs, w)

#    print "Iter: %d, prob: %f" % (it, prob)
    return (it, prob) 




def perceptron2 (coeffs, w, training_size):
    
    # create a 3xN data array. columns are (1, x, y)'
    data = np.array(np.random.uniform(-1, 1, (2, training_size)))
    ones = np.array(training_size * [[1]])
    data = np.concatenate((ones.T, data), axis=0)

    # f is the target function
    # g is the current hypothesis
    # w is the weight vector
    f = np.array(training_size * [-1])
#    g = np.array(training_size * [-1])
    g = np.sign(w.dot(data))

    ones_idx = np.array(coeffs[0] * data[:,1] + coeffs[1] * data[:,2] < 1).flatten()
    f[ones_idx] = 1

    maxIter = 100
    it = 0
    prob = 0
    while (np.any(f != g)) and (it < maxIter):
        it += 1

        # pick a random misclassified index
        miss_idx = np.argwhere(g != f)

        if len(miss_idx > 0):
            change_idx = np.array(random.sample(miss_idx, 1)).flatten()
            change_idx = change_idx[0]
            
            w = w + np.squeeze(f[change_idx]*data[:,change_idx])
            g = np.sign(np.dot(w, data))

            g[g == 0] = 1 # account for any possible "ties"
        
        prob = calc_prob(coeffs, w)

#    print "Iter: %d, prob: %f" % (it, prob)
    return (it, prob) 
