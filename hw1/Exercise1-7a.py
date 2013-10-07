#! /usr/bin/python
#
# This is support material for the course "Learning from Data" on edX.org
# https://www.edx.org/course/caltechx/cs1156x/learning-data/1120
#
# The software is intented for course usage, no guarantee whatsoever
# Date: Sep 30, 2013
#
# Template for a LIONsolver parametric table script.
#
# Generates a table based on input parameters taken from another table or from user input
#
# Syntax:
# When called without command line arguments:
#    number_of_inputs
#    name_of_input_1 default_value_of_input_1
#    ...
#    name_of_input_n default_value_of_input_n
# Otherwise, the program is invoked with the following syntax:
#    script_name.py input_1 ... input_n table_row_number output_file.csv
# where table_row_number is the row from which the input values are taken (assume it to be 0 if not needed)
#
# To customize, modify the output message with no arguments given and insert task-specific code
# to insert lines (using tmp_csv.writerow) in the output table.

import sys
import os
import random
import numpy as np

#
# If there are not enough parameters, optionally write out the number of required parameters
# followed by the list of their names and default values. One parameter per line,
# name followed by tab followed by default value.
# LIONsolver will use this list to provide a user friendly interface for the component's evaluation.
#
if len(sys.argv) < 2:
    sys.stdout.write ("2\nNumber of tests\t1000\nNumber of training points\t10\n")
    sys.exit(0)

#
# Retrieve the input parameters, the input row number, and the output filename.
#
in_parameters = [float(x) for x in sys.argv[1:-2]]
in_rownumber = int(sys.argv[-2])
out_filename = sys.argv[-1]

#
# Retrieve the output filename from the command line; create a temporary filename
# and open it, passing it through the CSV writer
#
tmp_filename = out_filename + "_"
tmp_file = open(tmp_filename, "w")

#############################################################################
#
# Task-specific code goes here.
#

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

# The following function is a stub for the perceptron training function required in Exercise1-7 and following.
# It currently generates random results.
# You should replace it with your implementation of the
# perceptron algorithm (we cannot do it otherwise we solve the homework for you :)
# This functon takes the coordinates of the two points and the number of training samples to be considered.
# It returns the number of iterations needed to converge and the disagreement with the original function.
def perceptron_training (x1, y1, x2, y2, training_size):
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
#        print "Iter: %d, prob: %f" % (it, prob)

#    print "Iter: %d, prob: %f" % (it, prob)
    return (it, prob) 



tests = int(in_parameters[0])
points = int(in_parameters[1])

# Write the header line in the output file, in this case the output is a 3-columns table containing the results
# of the experiments
# The syntax  name::type  is used to identify the columns and specify the type of data
header = "Test number::label,Number of iterations::number,Disagreement::number\n"
tmp_file.write (header)


# Repeat the experiment n times (tests parameter) and store the result of each experiment in one line of the output table
for t in range(1,tests+1):
    x1 = random.uniform (-1, 1)
    y1 = random.uniform (-1, 1)
    x2 = random.uniform (-1, 1)
    y2 = random.uniform (-1, 1)
    print t
    iterations, disagreement = perceptron_training (x1, y1, x2, y2, points)
    line = str(t) + ',' + str(iterations) + ',' + str(disagreement) + '\n'
    tmp_file.write (line)

#
#############################################################################

#
# Close all input files and the temporary output file.
#
tmp_file.close()

#
# Rename the temporary output file into the final one.
# It's important that the output file only appears when it is complete,
# otherwise LIONsolver might read an incomplete table.
#
os.rename (tmp_filename, out_filename)
