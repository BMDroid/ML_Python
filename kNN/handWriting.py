from os import listdir
from numpy import *
import kNN

def img2vector(filename):
  returnVect = zeros(shape=(1, 1024))
  fr = open(filename)
  for i in range(32):
    lineStr = fr.readline()
    for j in range(32):
      returnVect[0, 32 * i + j] = int(lineStr[j])
  return returnVect

def handWritingClassTest():
  hwLabels = []
  trainingFileList = listdir('trainingDigits')
  m = len(trainingFileList)
  trainingMat = zeros(shape=(m, 1024))
  for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0] # take off the '.txt' from the filename
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)
  testFileList = listdir('testDigits')
  errorCount = 0.0
  mTest = len(testFileList)
  for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = img2vector('trainingDigits/%s' % fileNameStr)
    classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
    print(f"The classifier came back with: {classifierResult}, the real answer is {classNumStr}.")
    if(classifierResult != classNumStr): errorCount += 1
  print(f"\nThe total number of errors is: {errorCount}.")
  print(f"\nThe total error rate is: {errorCount / float(mTest)}.")