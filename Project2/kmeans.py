import copy
from util import *
from sklearn.cluster import KMeans

# Number of threads to use for KMeans, -2 = all but one
CPU_NUM = -2

def computeKmeans(wp, wnp, k = 50):
    """
    Caluculates the kmeans

    Parameters:
        wp: listof(listof(int))
            For each word w, the list of documents that contain
            w and is positive (stars == 5)
        wnp: listof(listof(int))
            For each word w, the list of documents that contain
            w and is negative (stars == 1)
        k: int
            The number of clusters to do kmeans on

    Returns:
        Dictionary structure:
            { "kwp" : { "centres": listof coordinates for the cluster centre,
                        "labels": listof cluster number for each word }
              "kwnp": { same as above, except for wnp } }
    """

    dprt("  >> Running kmeans on WP...")
    kwp = KMeans(n_clusters = k, n_init = 20, n_jobs = CPU_NUM).fit(wp)
    dprt("  >> Running kmeans on WNP...")
    kwnp = KMeans(n_clusters = k, n_init = 20, n_jobs = CPU_NUM).fit(wnp)

    retval = {
            "kwp": {
                "centres": kwp.cluster_centers_,
                "labels": kwp.labels_,
                "scores": kwp.inertia_
            },
            "kwnp": {
                "centres": kwnp.cluster_centers_,
                "labels": kwnp.labels_,
                "scores": kwnp.inertia_
            }
    }

    return retval


# def binarify(kwp, kwnp, wp, wnp):
#     """
#     Creates binary attributes for the dataset

#     Parameters:
#         kwp: listof(clustered data)
#             The kmeans-clustered data, with kwp.centres (coordinates of centroids)
#             and kwp.labels (which word belongs to which cluster), for positive words

#         kwnp: listof(clustered data)
#             Same as kwp, except for negative words

#         wp: listof(listof int)
#             List of list of documents that contains the word at the same index as dictionary
#             e.g. wp[i][j] = 5 means document j has 5 occurence of word at dictList[i]
#             For positive words.

#         wnp: listof(listof int)
#             Same as wp, except for negative words

#     Returns: listof(listof int)
#         Binary attribute of whether a document contains a word in topic i
#         e.g. ret[i][j] = 1 means document i contains a word from topic j
#         Topic is a collection of words from dictionary, partitioned by kmeans cluster
#     """

#     dprt("  >> Constructing topics...")
#     topics = []
#     for i in range(0, len(kwp['labels'])):
#         setlst(topics, kwp['labels'][i], [wp[i]], lambda x, y: x + y, [])
#     for i in range(0, len(kwnp['labels'])):
#         setlst(topics, kwnp['labels'][i] + len(kwp['centres']), [wnp[i]], lambda x, y: x + y, [])

#     # Make a backup for return
#     origTopics = copy.deepcopy(topics)

#     # Compress - combine all word-doc within 1 topic into 1 list
#     # This is because as long as a doc has a word in a topic, it is 1,
#     # so it doesn't really matter which word that is as long as the word
#     # appears in the topic
#     for topicInd, topic in enumerate(topics):
#         # topic is a list of list of docs
#         newDoc = []
#         for word in topic:
#             # word is a list of document
#             for docInd, doc in enumerate(word):
#                 setlst(newDoc, docInd, 1 if doc > 0 else 0, lambda x, y: max(x, y), 0)

#         # Replace with compressed version
#         topics[topicInd] = newDoc

#         if len(topics[topicInd]) == 0:
#             topics[topicInd] = [0] * len(topics[topicInd - 1])

#     # Transpose topic-doc to doc-topic
#     dprt("  >> Building binary features...")
#     docBinary = []
#     for docInd in range(0, len(topics[0])):
#         document = []
#         for topic in topics:
#             # Each topic now is a list of doc that has word in the topic
#             document.append(topic[docInd])

#         docBinary.append(document)

#     return {"topics": origTopics, "binaryDoc": docBinary}
