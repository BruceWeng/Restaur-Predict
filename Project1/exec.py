#!/usr/bin/python
import sys, nbc, splitdata
from subprocess import call

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


def main(clabel, sample, newsample):
    run(clabel, sample, newsample)

if __name__ == "__main__":
    trainingSetSize = [0.1, 0.5, 0.9]
    funnyZOLoss = {'0.1': [], '0.5': [], '0.9': []}
    funnyBSLoss = {'0.1': [], '0.5': [], '0.9': []}

    starsZOLoss = {'0.1': [], '0.5': [], '0.9': []}
    starsBSLoss = {'0.1': [], '0.5': [], '0.9': []}

    fout = open('run_output.log', 'w+')

    # Compute each of 0.1, 0.5, 0.9 training sizes
    for size in trainingSetSize:
        print "Running for sample size " + str(size)
        for y in range(0, 10):
            # Funny data
            print "  >> Run #" + str(y + 1) + " for funny data"

            # Redirect output
            stdout_ = sys.stdout
            sys.stdout = fout
            # Run
            funnyErrors = run("f", size, 1)
            # Reset output
            sys.stdout = stdout_
            # Record result
            funnyZOLoss[str(size)].append(str(funnyErrors[0]))   # 0-1 loss
            funnyBSLoss[str(size)].append(str(funnyErrors[1]))   # Baseline loss

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
            starsZOLoss[str(size)].append(str(starsErrors[0]))   # 0-1 loss
            starsBSLoss[str(size)].append(str(starsErrors[1]))   # Baseline loss

    fout.close()

    print "Done.\n"

    # Print funny data
    print "Stats for funny data:"
    for size in trainingSetSize:
        print "  >> Zero-One Loss for Size " + str(size) + ": [" + ", ".join(funnyZOLoss[str(size)]) + "]"
        print "   | Baseline Loss for Size " + str(size) + ": [" + ", ".join(funnyBSLoss[str(size)]) + "]"

    print "\n"
    # Print stars data
    print "Stats for stars data:"
    for size in trainingSetSize:
        print "  >> Zero-One Loss for Size " + str(size) + ": [" + ", ".join(starsZOLoss[str(size)]) + "]"
        print "   |Baseline Loss for Size " + str(size) + ": [" + ", ".join(starsBSLoss[str(size)]) + "]"
