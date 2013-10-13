#! /usr/bin/python

import random
import numpy as np

def calc_prob (slope, icept, w):
    n_test_points = 1000
    f_test = np.array(n_test_points * [-1])
    g_test = np.array(n_test_points * [-1])

    test_data = np.array(np.random.uniform(-1,1,(2, n_test_points)))
    ones = np.array(n_test_points * [[1]])
    test_data = np.concatenate((ones.T, test_data), axis=0)

    f_ones_idx = test_data[2,:] < (slope * test_data[1,:] + icept)
    f_test[f_ones_idx] = 1
 
    g_test = np.sign(np.dot(w, test_data))
    g_test[g_test == 0] = 1

    prob = float(sum(g_test != f_test)) / float(n_test_points)

    return prob




def perceptron (x1, y1, x2, y2, training_size):
    slope = (y2 - y1) / (x2 - x1)
    icept = y2 - (slope * x2)

    # create a 3xN data array. columns are (1, x, y)'
    data = np.array(np.random.uniform(-1, 1, (2, training_size)))
    ones = np.array(training_size * [[1]])
    data = np.concatenate((ones.T, data), axis=0)

    # f is the target function
    # g is the current hypothesis
    # w is the weight vector
    f = np.array(training_size * [-1])
    g = np.array(training_size * [-1])
    ones_idx = data[2,:] < (slope * data[1,:] + icept)
    f[ones_idx] = 1
    w = np.array([0, 0, 0])

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
        
        prob = calc_prob(slope, icept, w)

#    print "Iter: %d, prob: %f" % (it, prob)
    return (it, prob) 
