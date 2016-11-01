import csv, os, sys
from random import sample

def split(fname, percent):
    """
    Return a list of values of the column in the CSV file

    Parameters
    ----------
        fname: str
            The name of the CSV file
        column: str
            The name of the column in the CSV file

    Returns
    -------
        List of values of the specified column
    """
    try:
        os.mkdir('./datasets')
    except:
        pass

    with open(fname) as f:
        reader = csv.reader(f)
        # Count rows, -1 for header
        numRows = sum(1 for row in reader) - 1
        # Reset cursor
        f.seek(0)

        header = reader.next()

        train = open('./datasets/' + fname + '_' + str(percent) + '_train.csv', 'w+')
        trainWriter = csv.writer(train, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        trainWriter.writerow(header)

        test = open('./datasets/' + fname + '_' + str(percent) + '_test.csv', 'w+')
        testWriter = csv.writer(test, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        testWriter.writerow(header)

        sampleCount = int(numRows * float(percent))
        # print "Sampling " + str(sampleCount) + " rows from data..."
        sampleRows = sample(xrange(0, numRows), sampleCount)
        sampleRows.sort()

        rowCount = 0
        trainSample = sampleRows.pop(0)
        for data in reader:
            if rowCount == trainSample:
                # This is for the training set
                trainWriter.writerow(data)
                # Pop the next row number in train sample
                if len(sampleRows) > 0:
                    trainSample = sampleRows.pop(0)
            else:
                testWriter.writerow(data)

            rowCount += 1

        train.close()
        test.close()

if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2]);
