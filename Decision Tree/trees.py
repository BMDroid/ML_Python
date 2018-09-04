from math import log
import operator


# 3.1 Function to calculate the Shannon entropy of a dataset
def calcShannonEnt(dataset):
    """ Calculate the Shannon Entropy of one dateset
    Args:
        dataSet::[[]]
            A list of lists
            assumed each lists has the same size
            and the last element of each list is the class label
    Returns:
        shannonEnt::float
            The shannon entropy of the dataset
            larger the entropy, more mixed dataset is
    """
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = 1 + labelCounts.get(currentLabel, 0)
#       if currentLabel not in labelCounts.keys():
#       labelCounts[currentLabel] = 0
#       labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts.get(key) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 3.2 Dataset splitting on a given feature
def splitDataSet(dataSet, axis, value):
    """ Split the dataset by certain axis
    Args:
        dataSet::[[]]
            A list of lists
            has the same assumption with the calShannonEnt
        axis::int
            The feature you want to split
        value::any
            The class label
    Returns:
        retDataSet::[[]]
            The new dataSet cut out the feature that you split on
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # slice to the axis(not including)
            reducedFeatVec.extend(featVec[axis + 1:])  # add two lists up
            # append new element to the dataset
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 3.3 Choosing the best feature to split on
def chooseBestFeatureToSplit(dataSet):
    """ Choose the best feature to split the dataset
    Args:
        dataSet::[[]]
            The data set stored in a list of lists
    Returns:
        bestFeature::int
            The index of the best feature of each instance
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # create a list of the choosen feature
        featList = [example[i] for example in dataSet]
        # create a set from the featList, removing reptetive elements
        uniqeVals = set(featList)
        newEntropy = 0
        for val in uniqeVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """ Count the majority element in the list
    Args:
        classList::[]
            A list of different classes
    Returns:
        sortedClassCount[0][0]::[tuple(classLable, counts)][0][0]
            The class label with the maximum occurency
            classLable::any type
                The class lable
            counts::int
                The number of each class label
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = 1 + classCount.get(vote, 0)
#       if vote not in classCount.keys():
#           classCount[vote] = 0
#       classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)  # sorted list contains tuples
    return sortedClassCount[0][0]


# 3.4 Tree-building
def createTree(dataSet, labels):
    """ Creat Tree
    Args:
        dataSet::[[]]
            A list of lists
        labels::
            A list of labels, contains a label for each of the features in the dataset
    Returns:
        myTree::{{}}
            The tree stored in the nested dictionary
    """
    classList = [example[-1] for example in dataSet]
    # base condition 1: all the instances in a brance are the same class
    if classList.count(classList[0]) == len(classList):
        return classList
    # base condition 2: run out of the features
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = labels[bestFeat]
    myTree = {bestFeatLable: {}}
    del (labels[bestFeat])
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    for val in uniqueVals:
        subLables = labels[:]  # make a copy of labels
        myTree[bestFeatLable][val] = createTree(
            splitDataSet(dataSet, bestFeat, val), subLables)
    return myTree


# 3.8 Classification function for an existing decision tree
def classify(inputTree, featLabels, testVec):
    """ Classify a instance using decision tree
    Args:
        inputTree:{{}}
            The tree stored in the nested dictionary
        featLabels:[]
            The list of feature labels
        testVec::[]
            The test instance stored in the list which size is len(dataSet[0] - 1)
    Returns:
        classLabel::any
            The predicted class label of the testVec
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    testFeat = testVec[featIndex]
    for key in list(secondDict.keys()):
        if testFeat == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 3.9 Methods for persisting the decision tree with pickle
def storeTree(inputTree, fileName):
    """ Store the decision in a file
    Args:
        inputTree::{{}}
            The tree stored in the nested dictionary
        fileName::str
            The file name you want to use
    """
    import pickle
    fw = open(fileName, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(fileName):
    """ Grab the decision tree from a file
    Args:
        fileName::str
            The name of the file which stores the decision tree
    Returns:
        pickle.load(fr)::{{}}
            Decition tree stored in the nested dictionary
    """
    import pickle
    fr = open(fileName)
    return pickle.load(fr)


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':
                                                  {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers':
                                                  {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = retrieveTree(0)
    print(classify(myTree, labels, [1, 0]))
