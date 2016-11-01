#!/usr/bin/python
import sys, nbc, nbcCluster
from subprocess import call
from util import *

# def run(clabel, sample, newsample):
#     fname = ""
#     cindex = 0
#     if clabel == "f":
#         fname = "funny_data.csv"
#         cindex = 5
#     else:
#         fname = "stars_data.csv"
#         cindex = 7

#     if newsample == 1:
#         splitdata.split(fname, str(sample))

#     trainName = "datasets/" + fname + "_" + str(sample) + "_train.csv"
#     testName = "datasets/" + fname + "_" + str(sample) + "_test.csv"

#     return nbc.main(trainName, testName, cindex - 1, 1)


# def main(clabel, sample, newsample):
    # run(clabel, sample, newsample)

def q1run():
    dprt("Running Q1...")

    ksize = [10, 20, 50, 100, 200]
    kmeansScore = {"kwp": {}, "kwnp": {}}
    sphKmeansScore = {"sknwp": {}, "sknwnp": {}}

    fout = open('ans/_run_output_q1.log', 'w+')

    # Compute for each cluster size
    for size in ksize:
        dprt("  >> Running for cluster size " + str(size))
        wordsOut = open('ans/words_q1_' + str(size), 'w+')

        # Redirect output
        stdout_ = sys.stdout
        sys.stdout = fout

        res = nbcCluster.nbcCluster(size)
        kmeansScore["kwp"][size] = res["kwpScore"]
        kmeansScore["kwnp"][size] = res["kwnpScore"]
        sphKmeansScore["sknwp"][size] = res["sknwpScore"]
        sphKmeansScore["sknwnp"][size] = res["sknwnpScore"]

        # Print each word in separate file so we can get the words with best K
        sys.stdout = wordsOut

        toPrint = ["kwpClusters", "kwnpClusters", "sknwpClusters", "sknwnpClusters"]
        for item in toPrint:
            print item
            print "------------------------------------------------------------"
            for x in res[item]:
                print x
            print "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
            print "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"


        sys.stdout = stdout_
        wordsOut.close()

    fout.close()

    # Print all the scores in one file
    scoresOut = open('ans/cluster_scores', 'w+')
    stdout_ = sys.stdout
    sys.stdout = scoresOut
    print "Cluster Sizes:"
    print ksize
    print ""

    print "KWP:"
    tmp = []
    for size in ksize:
        tmp.append(kmeansScore["kwp"][size])
    print tmp
    print ""

    print "KWNP:"
    tmp = []
    for size in ksize:
        tmp.append(kmeansScore["kwnp"][size])
    print tmp
    print ""

    print "SKNWP:"
    tmp = []
    for size in ksize:
        tmp.append(sphKmeansScore["sknwp"][size])
    print tmp
    print ""

    print "SKNWNP:"
    tmp = []
    for size in ksize:
        tmp.append(sphKmeansScore["sknwnp"][size])
    print tmp
    print ""

    scoresOut.close()
    sys.stdout = stdout_

if __name__ == "__main__":
    q1run()
