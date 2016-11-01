import csv, re, sys, string
from util import *
from os import path
from math import log

# Limit the number of features for testing
BCL_LIMIT = -1

def getColList(fname, clColIndex):
    """ Return a list of values of the column in the CSV file

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
            text = row[textColIndex].lower().replace('\n', ' ').translate(None, string.punctuation)
            while '  ' in text:
                text = text.replace('  ', ' ')

            clabel = row[clColIndex]

            colList['text'].append(text)
            colList['clabel'].append(int(clabel))

    return colList


def getMostFrequentWords(dataList, start = 0, length = 0):
    """ Returns the most frequently appeared words in the data

    Parameters
    ----------
        dataList: dict(text: list(str), clabel: list(int))
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

    splittedText = " ".join(dataList['text']).split(" ")

    if length == 0 or start + length > len(splittedText):
        length = max(0, len(splittedText) - start)
        dprt("Setting length to " + str(length))

    uniqueWords = {}

    for keyword in splittedText:
        if keyword == '':
            continue
        uniqueWords[keyword] = uniqueWords.get(keyword, 0) + 1

    lst = sorted(uniqueWords, key = uniqueWords.get, reverse = True)

    return lst[start:start + length]


def classify(dataList, dictionary, limit = -1, start = 0):
    """
    Constructs bag of words in binary format.
    The corresponding word will have 1 if the data has the word,
    or 0 if data does not have the word.

    Parameters
    ----------
        dataList: dict(text: list(str), clabel: list(int))
            The data to build on
        dictionary: listof(str)
            The dictionary to build on
        limit: int, optional
            To limit the number of row. Good for testing.
            -1 means no limit
        start: int
            Specify where to start building in from in dataList

    Returns
    -------
        A key-value pair with word being the key and
        whether the word appeared in dictionary being value
        (0 = did not appear, 1 = appeared)
    """

    bagOfWords = []

    if start >= len(dataList['text']):
        dprt("Start >= length of datalist: " + str(start) + " >= " + str(len(dataList['text'])))
        return bagOfWords

    for i in range(start, len(dataList['text'])):
        if limit == 0:
            break
        else:
            limit -= 1

        doc = dataList['text'][i]
        label = dataList['clabel'][i]

        docFeature = []
        for word in dictionary:
            docFeature.append(1 if " " + word + " " in doc else 0)

        docFeature.append(1 if label == 5 else 0)
        docFeature.append(1 if label == 1 else 0)
        bagOfWords.append(docFeature)

    return bagOfWords
