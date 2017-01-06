from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm


# 获取指定路径下的所有 .png 文件
def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
	
		
# 解析出 .png 图件文件的名称
def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]


# 将 20px * 20px 的图像数据转换成 1*400 的 numpy 向量
# 参数：imgFile--图像名  如：0_1.png
# 返回：1*400 的 numpy 向量
def img2vector(imgFile):
    #print("in img2vector func--para:{}".format(imgFile))
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normalization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1,-1)) # 1 * 400 矩阵
    return img_arr2
    
# 读取一个类别的所有数据并转换成矩阵 
# 参数：
#    basePath: 图像数据所在的基本路径
#       Mnist-image/train/
#       Mnist-image/test/
#    cla：类别名称
#       0,1,2,...,9
# 返回：某一类别的所有数据----[样本数量*(图像宽x图像高)] 矩阵
def read_and_convert(imgFileList):
    dataLabel = [] # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 400)) # dataNum * 400 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = get_img_name_str(imgNameStr)  # 得到 数字_实例编号.png
        #print("imgName: {}".format(imgName))
        classTag = imgName.split(".")[0].split("_")[0] # 得到 类标签(数字)
        #print("classTag: {}".format(classTag))
        dataLabel.append(classTag)
        dataMat[i,:] = img2vector(imgNameStr)
    return dataMat, dataLabel
	
	
# 读取训练数据
def read_all_data():
    cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_data_path = "Mnist-image\\train\\0"
    flist = get_file_list(train_data_path)
    dataMat, dataLabel = read_and_convert(flist)
    for c in cName:
        train_data_path_ = "Mnist-image\\train\\" + c
        flist_ = get_file_list(train_data_path_)
        dataMat_, dataLabel_ = read_and_convert(flist_)
        dataMat = np.concatenate((dataMat, dataMat_), axis=0)
        dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
    #print(dataMat.shape)
    #print(len(dataLabel))
    return dataMat, dataLabel
	

# create model
def create_svm(dataMat, dataLabel, decision='ovr'):
    clf = svm.SVC(decision_function_shape=decision)
    clf.fit(dataMat, dataLabel)
    return clf


#clf = svm.SVC(decision_function_shape='ovr')
st = time.clock()
clf = create_svm(dataMat, dataLabel, decision='ovr')
et = time.clock()
print("Training spent {:.4f}s.".format((et-st)))


# 对10个数字进行分类测试
def main():
    tbasePath = "Mnist-image\\test\\"
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.clock()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    for tcn  in tcName:
        testPath = "Mnist-image\\test\\" + tcn
        #print("class " + tcn + " path is: {}.".format(testPath))
        tflist = get_file_list(testPath)
        #tflist
        tdataMat, tdataLabel = read_and_convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)))

        #print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.clock()
        preResult = clf.predict(tdataMat)
        pre_et = time.clock()
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et-pre_st)))
        #print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x!=tcn])
        print("errorCount: {}.".format(errCount))
        allErrCount += errCount
        score_st = time.clock()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.clock()
        print("computing score spent {:.6f}s.".format(score_et-score_st))
        allScore += score
        print("score: {:.6f}.".format(score))
        print("error rate is {:.6f}.".format((1-score)))
        print("---------------------------------------------------------")


    tet = time.clock()
    print("Testing All class total spent {:.6f}s.".format(tet-tst))
    print("All error Count is: {}.".format(allErrCount))
    avgAccuracy = allScore/10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    print("Average error rate is: {:.6f}.".format(1-avgScore))











	