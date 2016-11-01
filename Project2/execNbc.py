#!/usr/bin/python
import sys, nbc, splitdata
from subprocess import call
from math import sqrt
from numpy import std

def run(clabel, sample, newsample):
    fname = ""
    cindex = 0
    if clabel == "f":
        fname = "funny_data.csv"
        cindex = 5
    else:
        fname = "stars_data.csv"
        cindex = 7

    if newsample == 1:
        splitdata.split(fname, str(sample))

    trainName = "datasets/" + fname + "_" + str(sample) + "_train.csv"
    testName = "datasets/" + fname + "_" + str(sample) + "_test.csv"

    return nbc.main(trainName, testName, cindex - 1, 1)


def stderr(lst, folds = 10):
    stdev = std(lst)
    return stdev / sqrt(folds)

def main(clabel, sample, newsample):
    run(clabel, sample, newsample)

if __name__ == "__main__":
    trainingSetSize = [100, 250, 500, 1000, 2000]

    starsZOLoss = {100: [], 250: [], 500: [], 1000: [], 2000: []}

    fout = open('run_output.log', 'w+')

    # Compute each of 0.1, 0.5, 0.9 training sizes
    for size in trainingSetSize:
        print "Running for sample size " + str(size)
        for y in range(0, 10):
            # Stars data
            print "  >> Run #" + str(y + 1) + " for stars data"
            # Redirect output
            stdout_ = sys.stdout
            sys.stdout = fout
            # Run
            starsErrors = run("s", size, 1)
            # Reset output
            sys.stdout = stdout_
            # Record result
            starsZOLoss[size].append(starsErrors[0])   # 0-1 loss

    fout.close()

    print "Done.\n"
    # Print stars data
    print "Stats for stars data:"
    for size in trainingSetSize:
        average = sum(starsZOLoss[size]) / len(starsZOLoss[size])
        err = stderr(starsZOLoss[size])
        print str(size) + ": L0/1 = "  + str(average) + ", stderr = " + str(err)
