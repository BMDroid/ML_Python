# 3.5 Plotting tree nodes with text annotations]
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrowArgs = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrowArgs)


def createPlot1():
    fig = plt.figure(1, facecolor="white")
    fig.clf()  # clear fig
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


# 3.6 Identify the num of leaves in a tree and the depth
def getNumLeafs(myTree):
    """ Get the number of leaf nodes in a decision tree
    Args:
        myTree:{{}}
            Desicion tree stored in a nested dictionary
    Returns:
        numLeafs::int
            The number of leaf nodes
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]  # first decision node of the tree
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """ Get the maximum depth of a decision tree
    Args:
        myTree:{{}}
            Desicion tree stored in a nested dictionary
    Returns:
        maxDepth::int
            maximum depth
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]  # first decision node of the tree
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 3.7 The plotTree function
def plotMidText(cntrPt, parentPt, txtString):
    """ Plot text between child and parent
    Args:
        cntrPt::[float]
            The coordinate of child node stored in a 2-elements list
            The first element is the X coordinate, the second element is the Y coordinate
        parentPt::[float]
            The coordinate of parent node stored in a 2-elements list
            The first element is the X coordinate, the second element is the Y coordinate
        txtString::str
            The text you wanted to add
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    """ Plot the decision tree
    Args:
        myTree::{{}}
            Decision tree stored in nested dictionary
        parentPt::[float]
            The coordinate of parent node stored in a 2-elements list
            The first element is the X coordinate, the second element is the Y coordinate
        nodeTxt::str
            The text of node
    """
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + numLeafs) /
              2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # keep track of currently ploted and make a note that are about to draw children nodes
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
        plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # two gloabal vars
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # keep track of what has already been plotted and the appropricate coordinate to place the next node
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers':
                                                  {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers':
                                                  {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


createPlot(retrieveTree(1))
