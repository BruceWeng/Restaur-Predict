import random, sys
from math import sqrt
from util import *
from sklearn.metrics.pairwise import cosine_similarity

numIterations = 20

def normalizeFeatures(wp, wnp):
    """
    Normalize features

    Parameters:
        wp: listof(listof(int))
            For each word w, the list of documents that contain
            w and is positive (stars == 5)
        wnp: listof(listof(int))
            For each word w, the list of documents that contain
            w and is negative (stars == 1)

    Returns:
        Dictionary structure:
            { "nwp" : [ _index = word, value = listof(doc having the word) ]
              "nwnp": [ same as above ] }
    """

    nwp = []
    for docLst in wp:
        nwp.append(normalize(docLst, smoothing = 0.00000000001))

    nwnp = []
    for docLst in wnp:
        nwnp.append(normalize(docLst, smoothing = 0.00000000001))

    return {"nwp": nwp, "nwnp": nwnp}


def cosineSimilarity(v1, v2):
    if len(v1) != len(v2):
        dprt("  >> ERORR: Cannot compute cosine similarity of vectors with unequal length...")
        sys.exit(-1)

    num = 0
    v1denum = 0
    v2denum = 0
    for i in range(0, len(v1)):
        num += v1[i] * v2[i]
        v1denum += v1[i] * v1[i]
        v2denum += v2[i] * v2[i]

    return num / (sqrt(v1denum) * sqrt(v2denum))


def sphKmeans(data, k, max_run = 300):
    """

    Returns: dictionary of list
        { "centres": listof coordinates of centres
          "labels": listof which word belongs to which cluster }
    """

    # Randomly select initial centroids
    curCentroids = random.sample(data, k)
    dataCluster = []
    nextCentroids = []
    scores = 0

    while True:
        # Centre as index, list of data in the centre as value
        clusters = [[]] * k
        # List of scores for each data point to its centre
        scores = 0
        dataCluster = []

        # Cluster based on cosine similarity
        for dtInd, dt in enumerate(data):
            bestCentroidInd = 0
            bestScore = -1
            for cInd, centroid in enumerate(curCentroids):
                score = cosineSimilarity(dt, centroid)
                # score = cosine_similarity(dt, centroid)[0][0]
                if score > bestScore:
                    bestCentroidInd = cInd
                    bestScore = score

            setlst(clusters, bestCentroidInd, [dt], lambda x, y: x + y)
            dataCluster.append(bestCentroidInd)
            scores += bestScore

        if max_run == 0:
            dprt("  >> Aborting because max_run reached")
            break
        max_run -= 1

        # Recompute centre
        for cInd, cluster in enumerate(clusters):
            clusterSum = []
            for docList in cluster:
                for docId, doc in enumerate(docList):
                    setlst(clusterSum, docId, doc, lambda x, y: x + y, 0)

            if len(clusterSum) == 0:
                setlst(nextCentroids, cInd, curCentroids[cInd])
            else:
                setlst(nextCentroids, cInd, normalize(clusterSum, smoothing = 0.00000000001))

        # Check if converged
        isConverged = True
        for i in range(0, len(curCentroids)):
            for j in range(0, len(curCentroids[i])):
                if curCentroids[i][j] != nextCentroids[i][j]:
                    isConverged = False
                    break
            if not isConverged:
                curCentroids = nextCentroids
                break

        if isConverged:
            break

    return {"centres": curCentroids, "labels": dataCluster, "scores": scores}


def computeSphKmeans(nwp, nwnp, k):
    dprt("  >> Computing spherical kmeans on nwp...")

    sknwp = sphKmeans(nwp, k)
    for i in range(1, numIterations):
        tmp = sphKmeans(nwp, k)
        if tmp["scores"] > sknwp["scores"]:
            sknwp = tmp

    dprt("  >> Computing spherical kmeans on nwnp...")
    sknwnp = sphKmeans(nwnp, k)
    for i in range(1, numIterations):
        tmp = sphKmeans(nwnp, k)
        if tmp["scores"] > sknwnp["scores"]:
            sknwnp = tmp

    return {"sknwp": sknwp, "sknwnp": sknwnp}
