import crossValidation, kmeans, nbc, pickle, random, re, sphKmeans, sys
from os import path
from util import *

# Number of most frequently used words to use
NUM_FREQ = 2000
SAVE_DATA_DIR = "data/full"

# NUM_FREQ = 400
# SAVE_DATA_DIR = "data/test"

# stars_data file
dataFilename = "stars_data.csv"
# stars attribute and its properties
classLabel = 6
clabelPositive = 5
clabelNegative = 1

def buildW(dataList, dictList):
    """
    Builds Wp and Wnp (word positive & word not positive)
    based on the provided data set and dictionary

    Parameters:
        dataList: listof(listof(str))
            Data from dataFilename, with fields 'text' and 'clabel'
        dictList: listof(str)
            The list of top most frequently used words

    Returns:
        A list of dictioanries: [[WP], [WNP]]
    """

    wp = []
    wnp = []

    dprt("Building WP and WNP...")
    for word in dictList:
        docPos = []
        docNeg = []

        for j in range(0, 400):# + range(2500, 2600):
        # for j in range(0, len(dataList['text'])):
            # text = dataList['text'][j].split(" ")
            text = dataList['text'][j]
            clabel = int(dataList['clabel'][j])
            cnt = text.count(word)

            # if wrappedWord in text:
            if clabel == clabelPositive:
                docPos.append(int(cnt))
            else:
                docNeg.append(int(cnt))

        wp.append(docPos)
        wnp.append(docNeg)

    return {"wp": wp, "wnp": wnp}


def mapWordsToCluster(clusters, dictList, numClusters):
    clustersOfWords = [[]] * numClusters
    for wInd, cluster in enumerate(clusters):
        setlst(clustersOfWords, cluster, [dictList[wInd]], lambda x, y: x + y, [])

    return clustersOfWords


def generateTopics(wpl, wnpl, wp, wnp, k):
    dprt("  >> Constructing topics...")

    # Group things together
    wplGroup = [[]] * k
    for wInd, cluster in enumerate(wpl):
        setlst(wplGroup, cluster, [wInd], lambda x, y: x + y, [])

    wnplGroup = [[]] * k
    for wInd, cluster in enumerate(wnpl):
        setlst(wnplGroup, cluster, [wInd], lambda x, y: x + y, [])

    ret = wplGroup + wnplGroup
    return ret



def generateNewData(k, save = True):
    dprt("[Feature Building]")
    # Get list of comments for each data
    dprt("  >> Getting list of comments...")
    # nbc.getColList returns {'text' = {}, 'col' = {}}
    dataList = nbc.getColList(dataFilename, classLabel)

    # Below code is to limit data size
    dataList['text'] = dataList['text'][0:200] + dataList['text'][2500:2700]
    dataList['clabel'] = dataList['clabel'][0:200] + dataList['clabel'][2500:2700]

    # Process train data for unique word counts
    dprt("  >> Extracting most frequently used words...")
    splittedList = re.findall(r'\w+', " ".join(dataList['text']))
    dictList = nbc.getMostFrequentWords(splittedList, 200, NUM_FREQ)

    # Build words matrices for wp and wnp
    words = buildW(dataList, dictList)

    dprt("[Standard KMeans]")
    kmean = kmeans.computeKmeans(words["wp"], words["wnp"], k)
    # kfeaturized = kmeans.binarify(kmean["kwp"], kmean["kwnp"], words["wp"], words["wnp"])
    # kbinary = kfeaturized['binaryDoc']
    ktopics = generateTopics(kmean["kwp"]['labels'], kmean["kwnp"]['labels'], words["wp"], words["wnp"], k)
    # ktopics = kfeaturized['topics']

    # Spherical kmeans
    dprt("[Spherical KMeans]")
    dprt("  >> Normalizing features...")
    normalized = sphKmeans.normalizeFeatures(words["wp"], words["wnp"])
    sphkmean = sphKmeans.computeSphKmeans(normalized["nwp"], normalized["nwnp"], k)
    # sphfeaturized = kmeans.binarify(sphkmean["sknwp"], sphkmean["sknwnp"], words["wp"], words["wnp"])
    # sphbinary = sphfeaturized['binaryDoc']
    sphtopics = generateTopics(sphkmean["sknwp"]['labels'], sphkmean["sknwnp"]['labels'], words["wp"], words["wnp"], k)
    # sphtopics = sphfeaturized['topics']

    if save:
        pickle.dump(dataList, open(SAVE_DATA_DIR + '/dataList', 'wb'))
        pickle.dump(dictList, open(SAVE_DATA_DIR + '/dictList', 'wb'))
        pickle.dump(words, open(SAVE_DATA_DIR + '/words', 'wb'))
        pickle.dump(kmean, open(SAVE_DATA_DIR + '/kmean_' + str(k), "wb"))
        # pickle.dump(kbinary, open(SAVE_DATA_DIR + '/kbinary_' + str(k), "wb"))
        pickle.dump(ktopics, open(SAVE_DATA_DIR + '/ktopics_' + str(k), "wb"))
        pickle.dump(sphkmean, open(SAVE_DATA_DIR + '/sphkmean_' + str(k), "wb"))
        # pickle.dump(sphbinary, open(SAVE_DATA_DIR + '/sphbinary_' + str(k), "wb"))
        pickle.dump(sphtopics, open(SAVE_DATA_DIR + '/sphtopics_' + str(k), "wb"))

    return [dataList, dictList, words, kmean, ktopics, sphkmean, sphtopics]


def loadDataFromFiles(k):
    dprt("Loading from saved data...")
    # Load data list (from stars data)
    dataList = pickle.load(open(SAVE_DATA_DIR + '/dataList', 'rb'))
    # Load dictionary
    dictList = pickle.load(open(SAVE_DATA_DIR + '/dictList', 'rb'))
    # Load word
    words = pickle.load(open(SAVE_DATA_DIR + '/words', 'rb'))
    # Load kmean save data
    kmean = pickle.load(open(SAVE_DATA_DIR + '/kmean_' + str(k), "rb"))
    ktopics = pickle.load(open(SAVE_DATA_DIR + '/ktopics_' + str(k), "rb"))
    # Load spherical kmean save data
    sphkmean = pickle.load(open(SAVE_DATA_DIR + '/sphkmean_' + str(k), "rb"))
    sphtopics = pickle.load(open(SAVE_DATA_DIR + '/sphtopics_' + str(k), "rb"))

    # Q4
    # dictList = ['o', 'bill', 'stained', 'whose', 'cucumber', 'kick', 'weed', 'jar', 'dark', 'class', 'upon', 'chemically', 'churches', 'royal', 'removed', 'mysterious', 'piece', 'offer', 'emptiness', 'advised', 'whenever', 'away', 'shopper', 'dispenser', 'slice', 'breast', 'plugged', 'air', 'chain', 'attitude', 'loved', 'deliveries', 'flavorful', 'bleach', 'written', 'shortly', 'youre', 'show', 'brings', 'taquitos', 'terrific', '72', 'terminating', 'sounded', 'inquiry', 'local', '22', 'puppy', 'essentially', 'losing', 'gtl', 'wash', 'staffing', 'sunday', 'start', 'energy', '82612830', 'atmosphere', 'raised', 'thin', 'chinese', 'performances', 'point', 'civility', 'jack', 'job', 'frequent', 'surprised', 'gravy', 'whats', 'ugh', 'soup', 'highschooler', 'calls', 'remembers', '150', 'forced', 'practically', 'line', 'kabob', 'meats', 'reviewbad', 'brats', 'zero', 'silent', 'halfprice', 'produce', 'corrupt', 'fries', 'modern', 'thatafter', 'fit', 'split', 'gringos', 'poured', 'home', 'apart', 'wifi', 'pitcherthe', 'anniversary']

    return [dataList, dictList, words, kmean, ktopics, sphkmean, sphtopics]


def nbcCluster(k):
    # objects = generateNewData(k, save = False)
    objects = loadDataFromFiles(k)

    dataList = objects[0]
    dictList = objects[1]
    words = objects[2]
    kmean = objects[3]
    # kbinary = objects[4]
    ktopics = objects[4]
    sphkmean = objects[5]
    # sphbinary = objects[7]
    sphtopics = objects[6]

    dprt("  >>  WP Dimension: " + str(len(words["wp"])) + " rows x " + str(len(words["wp"][0])) + " cols")
    dprt("  >> WNP Dimension: " + str(len(words["wnp"])) + " rows x " + str(len(words["wnp"][0])) + " cols")
    dprt("  >>     KWP Score: " + str(kmean["kwp"]["scores"]))
    dprt("  >>    KWNP Score: " + str(kmean["kwnp"]["scores"]))
    dprt("  >>   SKNWP Score: " + str(sphkmean["sknwp"]["scores"]))
    dprt("  >>  SKNWNP Score: " + str(sphkmean["sknwnp"]["scores"]))

    # Question 2
    partitions = crossValidation.cValidation(dataList, ktopics, words);
    print partitions
    partitions = crossValidation.cValidation(dataList, sphtopics, words);
    print partitions

    return {
            # Question 1 returns
            "kwpScore": kmean["kwp"]["scores"],
            "kwnpScore": kmean["kwnp"]["scores"],
            "sknwpScore": sphkmean["sknwp"]["scores"],
            "sknwnpScore": sphkmean["sknwnp"]["scores"],
            "kwpClusters": mapWordsToCluster(kmean["kwp"]["labels"], dictList, k),
            "kwnpClusters": mapWordsToCluster(kmean["kwnp"]["labels"], dictList, k),
            "sknwpClusters": mapWordsToCluster(sphkmean["sknwp"]["labels"], dictList, k),
            "sknwnpClusters": mapWordsToCluster(sphkmean["sknwnp"]["labels"], dictList, k)
    }


if __name__ == "__main__":
    if not(path.isfile(dataFilename)):
        print "Data file does not exist: " + dataFilename
        sys.exit(-1)

    # nbcCluster(10)
    # nbcCluster(20)
    nbcCluster(50)
    # nbcCluster(100)
    # nbcCluster(200)
