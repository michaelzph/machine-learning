import numpy as np
from os import listdir
import operator
import time

# img2vector 方法
# 参数：文件名
# 返回：numpy 数组
# 功能：将 32x32 的二进制图像矩阵转换为 1x1024 的向量


def img2vector(file):
    returnVec = np.zeros((1, 1024))
    with open(file) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0, 32*i+j] = int(lineStr[j])
    return  returnVec
	
# classify 分类方法
# 功能：对待测数据进行分类
# 参数：
#    inX：待分类数据
#    dataSet：训练数据集
#    labels：标签向量
#    k：近邻数量
# 返回：类标签

def classify(inX, dataSet, labels, k=3):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet   # tile(A,n) -- 将数据组 A 重复n次 
    sqDiffMat = np.power(diffMat, 2)
    sqDistance = sqDiffMat.sum(axis=1)
    distance = np.sqrt(sqDistance)
    sortedDistIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
	

# 读取图像数据文件并转换成矩阵
def read_and_convert(filePath):
    dataLabel = []
    fileList = listdir(filePath)
    fileAmount = len(fileList)
    dataMat = np.zeros((fileAmount, 1024))
    for i in range(fileAmount):
        fileNameStr = fileList[i]
        classTag = int(fileNameStr.split(".")[0].split("_")[0])
        dataLabel.append(classTag)
        dataMat[i,:] = img2vector(filePath+"/{}".format(fileNameStr))
    return dataMat, dataLabel

	
# 手写数字识别
def handwrittingClassify():
    hwlabels = []
    trainFilePath = "trainingDigits"
    trainFileList = listdir(trainFilePath)
    m = len(trainFileList)
    trainMat = np.zeros((m, 1024))
    st = time.clock()
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNum = int(fileStr.split("_")[0])
        hwlabels.append(classNum)
        #fpath = trainFilePath + "/" + fileNameStr
        trainMat[i,:] = img2vector(trainFilePath+"/{}".format(fileNameStr))
    #return trainMat, hwlabels
    testFilePath = "testDigits"
    testFileList = listdir(testFilePath)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNum = int(fileStr.split("_")[0])
        vectorTest = img2vector(testFilePath+"/{}".format(fileNameStr))
        classifyResult = classify(vectorTest, trainMat, hwlabels, 3)
        #print("the classifier came back with: {0}, the real answer is: {1}".format(classifyResult, classNum))
        if (classifyResult != classNum):
            errorCount += 1.0
    et = time.clock()
    print("cost {:.4f} s".format(et-st))
    print("the total numbers of error is: {}".format(errorCount))
    print("the total error rate is: {:.6f}".format(errorCount/float(mTest)))


	
	
