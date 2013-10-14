#! /usr/bin/python
#
# Coin flip simulation for homework 2 in Learning from Data

import numpy as np

n_coins = 1000  # number of coins
n_flips = 10    # number of flips
n_runs = 100000 # number of experiments
filename = "coins.csv"

outfile = open(filename, "w")
header = "Number,First,Minimum,Random\n"
outfile.write(header)

c_1 = np.empty(n_runs)
c_min = np.empty(n_runs)
c_rand = np.empty(n_runs)

for i in np.arange(0, n_runs):
    if np.mod(i, 10000) == 0:
        print i
    # flip each coin n_flips times
    outcomes = np.random.rand(n_coins, n_flips)
    is_heads = outcomes > 0.5
    
    # number of heads in the n_flips flips for each coin
    n_heads = is_heads.sum(axis=1)

    min_idx = np.argwhere(n_heads == np.min(n_heads))
    min_idx = min_idx[0].astype(int)

    rand_idx = np.random.randint(0, n_coins)

    c_1[i] = float(n_heads[0]) / float(n_flips)
    c_min[i] = float(n_heads[min_idx]) / float(n_flips)
    c_rand[i] = float(n_heads[rand_idx]) / float(n_flips)
    
    line = "%d,%f,%f,%f\n" % (i, c_1[i], c_min[i], c_rand[i])
    outfile.write(line)
    #print (line)

outfile.close()

print "c_1 mean: %f, sd: %f" % (np.mean(c_1), np.std(c_1))
print "c_min mean: %f, sd: %f" % (np.mean(c_min), np.std(c_min))
print "c_rand mean: %f, sd: %f" % (np.mean(c_rand), np.std(c_rand))
