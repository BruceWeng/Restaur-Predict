import csv, random, re, sys
from os import path
from math import log

LENGTH = 2000
BCL_LIMIT = -1

# Print debug statement if true
debug = False

def dprt(message):
    """
    Prints debug message

    Parameters
        message: The message to print
    """

    if debug:
        print "\033[01;32m[DEBUG] " + message + "\033[00m"


def getColList(fname, clColIndex):
    """
    Return a list of values of the column in the CSV file

    Parameters
    ----------
        fname: str
            The name of the CSV file
        columns: str
            The name of the column in the CSV file to get

    Returns
    -------
        List of values of the specified columns
    """
    colList = {'text': [], 'clabel': []}

    with open(fname) as f:
        reader = csv.reader(f)
        header = reader.next()
        textColIndex = header.index('text')

        for row in reader:
            text = re.sub(r'[^a-zA-Z0-9 ]', '', row[textColIndex].lower())
            clabel = row[clColIndex]

            colList['text'].append(text)
            colList['clabel'].append(clabel)

    return colList


def getMostFrequentWords(data, start = 0, length = 0):
    """
    Returns the most frequently appeared words in the data

    Parameters
    ----------
        data: list(str)
            The data to count unique words from
        start: int, optional
            Cut off the first start most frequent words
        length: int, optional
            The number of most frequent words to return
            from start

    Returns
    -------
        List of strings that are the start-th to (start + length)-th most
        frequently used words
    """

    if length == 0 or start + length > len(data):
        length = len(data) - start
        dprt("Setting length to " + str(length))

    uniqueWords = {}

    for keyword in data:
        if keyword == '':
            continue
        uniqueWords[keyword] = uniqueWords.get(keyword, 0) + 1

    # Obtains the sorted list from index 200 to 2199 (201st to 2200th item)
    lst = sorted(uniqueWords, key = uniqueWords.get, reverse = True)

    return lst[start:start + length]


def buildBinaryClassLabelString(data, dictionary, limit = BCL_LIMIT):
    """
    Constructs bag of words in binary format.
    The corresponding word will have 1 if the data has the word,
    or 0 if data does not have the word.

    Parameters
    ----------
        data: list(str)
            The data to build on
        dictionary: listof(str)
            The dictionary to build on
        limit: int, optional
            To limit the number of row. Good for testing.
            -1 means no limit

    Returns
    -------
        A key-value pair with word being the key and
        whether the word appeared in dictionary being value
        (0 = did not appear, 1 = appeared)
    """

    bagOfWords = []

    for colval in data:
        if limit == 0:
            break
        else:
            limit -= 1

        document = []
        for word in dictionary:
            document.append(1 if word in colval else 0)

        bagOfWords.append(document)

    return bagOfWords

def buildBinaryClassLabelInt(data, clabel, limit = BCL_LIMIT):
    """
    Constructs bag of words in binary format.
    The corresponding word will have 1 if the data is > gt,
    or 0 if not.

    Parameters
    ----------
        data: list(str)
            The data to build on
        gt: int
            The value to be greater than
        limit: int, optional
            To limit the number of row. Good for testing.
            -1 means no limit

    Returns
    -------
        A list same dimension as data with 1 being the
        value > gt and 0 being the value <= gt
    """

    bagOfWords = []
    numPositive = 0

    for colval in data:
        if limit == 0:
            break
        else:
            limit -= 1

        if clabel == "funny":
            bagOfWords.append(1 if int(colval) > 0 else 0)
        else:
            bagOfWords.append(1 if int(colval) == 5 else 0)

    return bagOfWords

def nbcLearn(words, cl):
    """
    Learn a model from our training set

    Parameters
    ----------
        trainingSet: list(list(int))
            Training set, each element is a list of int mapping to the existence of
            words in dictionary (1 = existed, 0 otherwise)
        cl: list(int)
            Class label, 1 = positive and 0 = negative

    Returns
    -------
        Learned model, list(list(list(float))), in the following form:
        [[ List of negative probabilities of word i, List of positive probabilities of word i]]
    """

    # Allocate lists of 0
    posHasWord = [0] * len(words[0])
    negHasWord = [0] * len(words[0])

    # Count the occurence of each word corresponding to the class label
    for i in range(0, len(words)):
        for j in range(0, len(words[i])):
            if words[i][j] == 1:
                if cl[i] == 1:
                    posHasWord[j] += 1
                else:
                    negHasWord[j] += 1

    # Get number of positive/negative labels
    numPos = cl.count(1)
    numNeg = len(cl) - numPos

    # Calculate the probability for each of the following categories
    probPositive = map((lambda(x): ((x + 1) / float(numPos + 2))), posHasWord)
    probNegative = map((lambda(x): ((x + 1) / float(numNeg + 2))), negHasWord)
    probPositiveNoWord = map(lambda(x): (1 - x), probPositive)
    probNegativeNoWord = map(lambda(x): (1 - x), probNegative)

    model = [[probNegative, probNegativeNoWord], [probPositive, probPositiveNoWord]]
    return model

def nbcTest(model, testingSet, cl):
    """
    Tests the provided dataset using the model

    Parameters
    ----------
        model:  list(list(list(float)))
            The model we learned from training dataset in the following form:
            [[ List of negative probabilities of word i, List of positive probabilities of word i]]
        testingSet: list(list(int))
            The testing set
        cl: list(int)
            Class label; 1 = positive and 0 = negative

    Returns
    -------
        Accruacy of our model
    """

    # Calculate base probability (P(positive) and P(negative))
    basePosProb = log(sys.float_info.min if cl.count(1) == 0 else cl.count(1) / float(len(cl)))
    baseNegProb = log(sys.float_info.min if cl.count(0) == 0 else cl.count(0) / float(len(cl)))

    numCorrect = 0

    for i in range(0, len(testingSet)):
        # Reset probability for new document
        posProb = basePosProb
        negProb = baseNegProb

        for j in range(0, len(testingSet[i])):
            if testingSet[i][j] == 1:
                # If the word exists, count it towards the probability for both negative and positive
                posProb += log(float(model[1][0][j]))
                negProb += log(float(model[0][0][j]))
            else:
                posProb += log(float(model[1][1][j]))
                negProb += log(float(model[0][1][j]))

        # Find the winner
        isPositive = 1 if posProb > negProb else 0
        # Keep track of correct guesses
        numCorrect += 1 if cl[i] == isPositive else 0

    return 1 - (numCorrect / float(len(cl)))

def bsLoss(trainingCL, testingCL):
    positiveCount = trainingCL.count(1);
    largerPrior = 1
    if positiveCount < len(trainingCL) - positiveCount:
        largerPrior = 0

    numCorrect = 0
    for x in testingCL:
        if x == largerPrior:
            numCorrect += 1

    return 1 - (numCorrect / float(len(testingCL)))


def main(trainDataFile, testDataFile, classLabelIndex, topTen):
    """
    Calculates the top words and find the 0-1 loss
    """

    # Process command line inputs
    # trainDataFile = argv[1]
    # testDataFile = argv[2]
    # classLabelIndex = int(argv[3]) - 1
    # topTen = True if int(argv[4]) == 1 else False

    # Input validation
    if not(path.isfile(trainDataFile)):
        print "Train data file does not exist: " + trainDataFile
        sys.exit(-1)

    if not(path.isfile(testDataFile)):
        print "Test data file does not exist: " + testDataFile
        sys.exit(-1)

    if classLabelIndex != 4 and classLabelIndex != 6:
        print "Class label index is neither 5 nor 7: " + str(classLabelIndex + 1)
        sys.exit(-1)

    # See output for command line arguments
    dprt("Train data file: " + trainDataFile)
    dprt(" Test data file: " + testDataFile)
    dprt("    Class label: " + ("funny" if classLabelIndex == 4 else "stars"))
    dprt("Print top words: " + ("true" if topTen else "false"))

    # Get list of comments for each data
    dprt("Getting list of comments...")
    trainList = getColList(trainDataFile, classLabelIndex)
    testList = getColList(testDataFile, classLabelIndex)

    # Process train data for unique word counts
    dprt("Extracting most frequently used words...")
    splittedTrainList = re.findall(r'\w+', " ".join(trainList['text']))
    dictList = getMostFrequentWords(splittedTrainList, 200, LENGTH)

    # Print top 10 if required
    if topTen:
        for i in range(0, 10):
            print "WORD" + str(i + 1) + " " + dictList[i]

    # Q3 from HW4
    # spaceWrappedDict = map(lambda(x): " " + x + " ", dictList)
    # Q4 from HW 4
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

    # Build class label for each word in dictionary on whether the word
    # appeared in the comment or not
    dprt("Building binary class label for training set on dictionary...")
    trainCommentBCL = buildBinaryClassLabelString(trainList['text'], spaceWrappedDict)
    dprt("Building binary class label for test set on dictionary...")
    testCommentBCL = buildBinaryClassLabelString(testList['text'], spaceWrappedDict)

    trainIntBCL = []
    testIntBCL = []

    # Determine to use funny or stars as class label
    column = ""
    if classLabelIndex == 4:
        column = "funny"
    elif classLabelIndex == 6:
        column = "stars"

    # Get the binary labels for the corresponding class labels
    dprt("Building binary class label for training set on " + column)
    trainIntBCL = buildBinaryClassLabelInt(trainList['clabel'], column)
    dprt("Building binary class label for test set on " + column)
    testIntBCL = buildBinaryClassLabelInt(testList['clabel'], column)

    # Learn
    dprt("Learning from training set...")
    model = nbcLearn(trainCommentBCL, trainIntBCL)
    # Test
    dprt("Testing learned model with testing set...")

    zeroOneLoss = nbcTest(model, testCommentBCL, testIntBCL)
    print "ZERO-ONE-LOSS " + str(zeroOneLoss)

    baselineLoss = bsLoss(trainIntBCL, testIntBCL)
    dprt("Baseline Loss: " + str(baselineLoss))
    return [zeroOneLoss, baselineLoss]

# Only run this if this is the script being called
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) - 1, int(sys.argv[4]))
