from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator


def createDataSet():
  group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return group, labels

def file2matrix(filename):
  love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
  fr = open(filename)
  numberOfLines = len(fr.readlines()) # this is the sample size
  returnMat = zeros((numberOfLines, 3)) # create the Numpy matrix to return, the 3 here is for the example of the dating site
  classLabelVector = []
  fr = open(filename)
  index = 0
  for line in fr.readlines():
    line = line.strip() # remove the white space from the line
    listFromLine = line.split('\t') # split the line with tab
    returnMat[index,:] = listFromLine[0:3]
    if(listFromLine[-1].isdigit()):
      classLabelVector.append(int(listFromLine[-1]))
    else:
      classLabelVector.append(love_dictionary.get(listFromLine[-1])) # append the last element of each line to the label vector
    index += 1
  return returnMat, classLabelVector

def autoNorm(dataSet):
  minVals = dataSet.min(0) # get the minimum values of each column and place in the miniVals
  maxVals = dataSet.max(0)
  ranges = maxVals - minVals
  normDataSet = zeros(shape(dataSet))
  m = dataSet.shape[0]
  normDataSet = dataSet - tile(minVals, (m, 1)) # tile() function create a matrix the same size as our input matrix and then fill it up with many copies, or tiles
  normDataSet = normDataSet / tile(ranges, (m, 1))
  return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0] # the row numbers of the dataSet which is the data sets size
  diffMat = tile(inX, (dataSetSize, 1)) - dataSet # create an array that has the same dimension of the dataSet containning inX
  sqDiffMat = diffMat ** 2
  sqDistances = sqDiffMat.sum(axis=1) # sum each row
  distances = sqDistances ** 0.5
  # print(f"The distance array is: {distances}")
  sortedDistIndicies = distances.argsort() # return the indices of the sorted array from the smallest to the largest
  classCount = {}
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
  # print(f"The classCount is: {classCount}")
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # take the classCount dict and decompose it into a list of tuples and then sort the tuples by the second item in the tuple using the itemgetter method from the operator module
  # print(f"The sorted classCount is: {sortedClassCount}")
  return sortedClassCount[0][0] # return the label of the item occurring the most frequently.


def datingClassTest():
  hoRatio = 0.20 # the ratio of the test set of the whole data set

  datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
  normMat, _, _ = autoNorm(datingDataMat)
  m = normMat.shape[0]
  numTestVecs = int(m * hoRatio) # the size of the test set
  errorCount = 0.0 # count the error]
  k = 5
  for i in range(numTestVecs):
    classfierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], k)
    print(f"The classifier came back with: {classfierResult}, the real answer is: {datingLabels[i]}")
    if(classfierResult != datingLabels[i]): errorCount += 1.0
  print(f"The total error rate is: {errorCount / float(numTestVecs)}")

def classifyPerson():
  resultList = ['not at all', 'in small doses', 'in large doses']
  ffMiles = float(input("Frequent flier miles earned per year: "))
  percentTats = float(input("Percentage of time spent playing video games: "))
  iceCream = float(input("Liters of ice cream consumed per year: "))
  datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
  normMat, ranges, minVals = autoNorm(datingDataMat)
  inArr = array([ffMiles, percentTats, iceCream])
  classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
  print(f"You will probably like this person: {resultList[classifierResult - 1]}")


def testkNN():
  group, _ = createDataSet()
  _, labels = createDataSet()
  inX = [0, 0]
  print(classify0(inX, group, labels, 3))
  # Plot the datingTestData
  datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels)) # Figure 2.5
  plt.show()
  #Classify a Person
  classifyPerson()


