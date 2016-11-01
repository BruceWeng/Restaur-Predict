import sys
from binarify import *
from scipy.stats import chi2_contingency
from util import *

# Constants
FILENAME = "stars_data.csv"
COLINDEX = 6
SUPPORT = 0.03
CONFIDENCE = 0.25
SIGNIFICANCE = 0.05
MAX_ISET_SIZE = 3

# Test related constants
ENV = ""
DICT_SIZE = 200 if ENV == "test" else 2000
FEAT_SIZE = 250 if ENV == "test" else -1

# Chi count
chiCount = 0

def mapRules(rule, dictList):
    rule = listify(rule, "=>", str)
    ant = [dictList[i] for i in listify(rule[0])]
    con = [dictList[i] for i in listify(rule[1])]
    return stringify(ant, " AND ") + " THEN " + stringify(con)


def getCandidates(itemsets, size):
    last = len(itemsets[0]) - 1
    returnset = []
    for i in range(len(itemsets)):
        p = itemsets[i]
        for j in range(i, len(itemsets)):
            q = itemsets[j]

            if p[:last] == q[:last] and p[last] < q[last]:
                newset = p[:] + [q[last]]

                # Pruning - remove those whose subsets are not in the original itemset
                canAppend = True
                for subsetIndex in range(len(newset)):
                    canAppend = newset[:subsetIndex] + newset[subsetIndex + 1:] in itemsets
                    if not canAppend:
                        break
                if canAppend:
                    returnset.append(newset)
    return returnset


def pruneBySupport(itemsets, binFeatures, support):
    numPruned = 0
    frequencies = [0] * len(itemsets)

    dprt("   Itemsets size: " + str(len(itemsets)), 3)
    # Increase count for support of an itemset if a doc
    # has every word in the set
    for doc in binFeatures:
        for ind, iset in enumerate(itemsets):
            hasAll = True
            for item in iset:
                if doc[item] != 1:
                    hasAll = False
                    break
            frequencies[ind] += 1 if hasAll else 0

    for ind, freq in enumerate(frequencies):
        if freq < support * len(binFeatures):
            numPruned += 1
            frequencies[ind] = None
            itemsets[ind] = None

    itemsets = [ item for item in itemsets if item != None ]
    frequencies = [ freq for freq in frequencies if freq != None ]

    dprt("Itemsets removed: " + str(numPruned), 3)
    dprt("   Itemsets left: " + str(len(itemsets)), 3)
    return { "itemsets": itemsets, "frequencies": frequencies}


def getRules(allSets, allFreq, confidence, maxsize):
    rules = {}
    for k in range(2, maxsize + 1):
        lastItemsets = allSets[k - 1]
        lastFrequencies= allFreq[k - 1]
        curItemsets = allSets[k]
        curFrequencies = allFreq[k]

        for ind, iset in enumerate(curItemsets):
            for i in range(len(iset) - 1, -1, -1):
                ant = iset[:i] + iset[i + 1:]
                con = [iset[i]]
                consup = allFreq[1][allSets[1].index(con)]

                antsup = lastFrequencies[lastItemsets.index(ant)]
                cursup = curFrequencies[ind]

                conf = cursup / float(antsup)
                if conf >= confidence:
                    key = stringify(ant) + "=>" + stringify(con)
                    rules[key] = [(antsup) / float(5000), conf, consup / float(5000)]

    ret = sorted(rules.iteritems(), key = lambda e: e[1][1])
    ret.reverse()
    return ret


def getChi(cursup, antsup, consup):
    size = 500 if ENV == "test" else 5000
    ac = cursup
    anc = antsup - ac
    nac = consup - ac
    nanc = size - consup - anc

    chi = chi2_contingency([[ac, anc],[nac, nanc]])

    return [chi[0], chi[1]]


def getRulesByChi(allSets, allFreq, confidence, maxsize, bonferroni = 1):
    global chiCount
    rules = {}
    for k in range(2, maxsize + 1):
        lastItemsets = allSets[k - 1]
        lastFrequencies= allFreq[k - 1]
        curItemsets = allSets[k]
        curFrequencies = allFreq[k]

        for ind, iset in enumerate(curItemsets):
            for i in range(len(iset) - 1, -1, -1):
                ant = iset[:i] + iset[i + 1:]
                con = [iset[i]]

                antsup = lastFrequencies[lastItemsets.index(ant)]
                consup = allFreq[1][allSets[1].index(con)]
                cursup = curFrequencies[ind]

                chi = getChi(cursup, antsup, consup)
                chiCount += 1
                if chi[1] < SIGNIFICANCE / float(bonferroni):
                    key = stringify(ant) + "=>" + stringify(con)
                    rules[key] = [chi[0], chi[1]]

    ret = sorted(rules.iteritems(), key = lambda e: e[1][1])
    return ret


def apriori(binFeatures, support, confidence, dictList, maxsize, mode = "q1"):
    allSets = {}
    allFreq = {}

    # L in slides pseudo code, initialize L1 to be [[0], [1], ..., [2002]]
    itemsets = [[single] for single in range(len(binFeatures[0]))]

    for k in range(1, maxsize + 1):
        dprt("Running " + str(k) + "-th run of apriori...", 1)

        dprt("Pruning anything with support < " + str(support), 2)
        results = pruneBySupport(itemsets, binFeatures, support)
        itemsets = results["itemsets"]

        allSets[k] = itemsets
        allFreq[k] = results["frequencies"]

        if k == maxsize:
            break

        # C_k+1 on slides
        dprt("Generating next candidate sets...", 2)
        itemsets = getCandidates(itemsets, k)
        dprt(str(len(itemsets)) + " itemsets generated.", 3)
        dprt("")

    global chiCount


    dprt("Generating rules...", 2)
    rules = {}
    if mode == "q1":
        rule = getRules(allSets, allFreq, confidence, maxsize)
    elif mode == "q3a" or "q3e":
        rules = getRulesByChi(allSets, allFreq, confidence, maxsize)

        if mode == "q3e":
            dprt("Ran " + str(chiCount) + " chi-square tests...", 2)
            rules = getRulesByChi(allSets, allFreq, confidence, maxsize, chiCount)

    dprt("Generated " + str(len(rules)) + " rules.", 3)
    dprt("Saving rules to ./results...", 1)

    f = open("results", "w+")
    for rule in rules:
        f.write(mapRules(rule[0], dictList) + " " + str(rule[1]) + "\n")

def run(mode = "q1"):
    dprt("Generating dictionary...")
    dataList = getColList(FILENAME, COLINDEX)
    dictList = getMostFrequentWords(dataList, 100, DICT_SIZE)
    dprt("Generating binary features...")
    binFeatures = classify(dataList, dictList, limit = FEAT_SIZE)
    if ENV == "test":
        binFeatures.extend(classify(dataList, dictList, limit = FEAT_SIZE, start = 2500))

    dprt("Running apriori algorithm...")
    dictList.extend(["isPositive", "isNegative"])
    apriori(binFeatures, SUPPORT, CONFIDENCE, dictList, MAX_ISET_SIZE, mode)


if __name__ == "__main__":
    mode = "q1"
    if len(sys.argv) > 1 and sys.argv[1] in ["q1", "q3a", "q3e"]:
        mode = sys.argv[1]

    dprt("Running in mode " + mode + "...")
    run(mode)
