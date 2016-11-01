import copy, nbc, random, sys
from numpy import std
from math import sqrt
from sklearn.naive_bayes import BernoulliNB
# for q4, not needed else where
spaceWrappedDict = [' o ', ' bill ', ' stained ', ' whose ', ' cucumber ', ' kick ',
            ' weed ', ' jar ', ' dark ', ' class ', ' upon ', ' chemically ', ' churches ',
            ' royal ', ' removed ', ' mysterious ', ' piece ', ' offer ', ' emptiness ',
            ' advised ', ' whenever ', ' away ', ' shopper ', ' dispenser ', ' slice ', ' breast ',
            ' plugged ', ' air ', ' chain ', ' attitude ', ' loved ', ' deliveries ', ' flavorful ',
            ' bleach ', ' written ', ' shortly ', ' youre ', ' show ', ' brings ', ' taquitos ',
            ' terrific ', ' 72 ', ' terminating ', ' sounded ', ' inquiry ', ' local ', ' 22 ',
            ' puppy ', ' essentially ', ' losing ', ' gtl ', ' wash ', ' staffing ', ' sunday ',
            ' start ', ' energy ', ' 82612830 ', ' atmosphere ', ' raised ', ' thin ', ' chinese ',
            ' performances ', ' point ', ' civility ', ' jack ', ' job ', ' frequent ', ' surprised ',
            ' gravy ', ' whats ', ' ugh ', ' soup ', ' highschooler ', ' calls ', ' remembers ', ' 150 ',
            ' forced ', ' practically ', ' line ', ' kabob ', ' meats ', ' reviewbad ', ' brats ', ' zero ',
            ' silent ', ' halfprice ', ' produce ', ' corrupt ', ' fries ', ' modern ', ' thatafter ', ' fit ',
            ' split ', ' gringos ', ' poured ', ' home ', ' apart ', ' wifi ', ' pitcherthe ', ' anniversary ']


def partition(dataList):
    # Get the length, we'll get the samples from there
    partitionSize = len(dataList['text']) / 10
    indexRef = range(0, len(dataList['text']))

    partitions = []
    for i in range(0, 10):
        random.shuffle(indexRef)
        part = indexRef[0:partitionSize]
        indexRef = indexRef[partitionSize:]

        partitions.append(part)

    return partitions


def flatten(partitions):
    ret = []
    for part in partitions:
        ret.extend(part)
    return ret


def stderr(lst, folds = 10):
    stdev = std(lst)
    return stdev / sqrt(folds)


def binarify(dataList, dataSet, topics, words):
    halfData =  len(dataList['clabel']) / 2

    q4data = []
    for doc in dataSet:
        q4data.append(dataList['text'][doc])

    nbcbin = nbc.buildBinaryClassLabelString(q4data, spaceWrappedDict)
    binaryDoc = []
    binaryInt = []
    for i, doc in enumerate(dataSet):
        docTopic = []

        classLabel = 1 if int(dataList['clabel'][doc]) > 1 else 0
        normalizedIndex = doc if doc < halfData else (doc - halfData)
        for topic in topics:
            inTopic = 0
            for word in topic:
                if classLabel == 1:
                    inTopic = 1 if words["wp"][word][normalizedIndex] else 0
                else:
                    inTopic = 1 if words["wnp"][word][normalizedIndex] else 0
            docTopic.append(inTopic)

        # For q 4, not needed
        # docTopic.extend(nbcbin[i])

        binaryInt.append(classLabel)
        binaryDoc.append(docTopic)
    return {"binaryDoc": binaryDoc, "clabel": binaryInt}


def cValidation(dataList, topics, words, folds = 10):
    tss = [100, 250, 500, 1000, 2000]
    partitions = partition(dataList)

    ret = {'scores': [], 'err': []}

    for t in tss:
        scores = []
        for i in range(0, folds):
            # We don't want to change the partitions
            trainPool = copy.deepcopy(partitions)
            testSet = trainPool.pop(i)
            trainPool = flatten(trainPool)

            # Randomly sample t values
            random.shuffle(trainPool)
            trainSet = trainPool[0:t]
            trainBin = binarify(dataList, trainSet, topics, words)
            testBin = binarify(dataList, testSet, topics, words)
            print (len(trainBin['binaryDoc'][0]))
            model = nbc.nbcLearn(trainBin['binaryDoc'], trainBin['clabel'])
            zeroOneLoss = nbc.nbcTest(model, testBin['binaryDoc'], testBin['clabel'])

            scores.append(zeroOneLoss)
        ret['scores'].append(sum(scores) / len(scores))
        ret['err'].append(stderr(scores))
    return ret
