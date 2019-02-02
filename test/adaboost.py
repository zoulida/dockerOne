__author__ = 'zoulida'
from numpy import *
import matplotlib.pyplot as plt


# 加载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 返回分类预测结果  根据阈值所以有两种返回情况
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 返回 该弱分类器单层决策树的信息  更新D向量的错误率 更新D向量的预测目标
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}  # 字典用于保存每个分类器信息
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # 初始化最小误差最大
    for i in range(n):  # 特征循环  （三层循环，遍历所有的可能性）
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # (大-小)/分割数  得到最小值到最大值需要的每一段距离
        for j in range(-1, int(numSteps) + 1):  # 遍历步长 最小值到最大值的需要次数
            for inequal in ['lt', 'gt']:  # 在大于和小于之间切换
                threshVal = (rangeMin + float(j) * stepSize)  # 最小值+次数*步长  每一次从最小值走的长度
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # 最优预测目标值  用于与目标值比较得到误差
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                if weightedError < minError:  # 选出最小错误的那个特征
                    minError = weightedError  # 最小误差 后面用来更新D权值的
                    bestClasEst = predictedVals.copy()  # 最优预测值

                    bestStump['dim'] = i  # 特征
                    bestStump['thresh'] = threshVal  # 到最小值的距离 （得到最优预测值的那个距离）
                    bestStump['ineq'] = inequal  # 大于还是小于 最优距离为-1
    return bestStump, minError, bestClasEst


# 循环构建numIt个弱分类器
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []  # 保存弱分类器数组
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # D向量 每条样本所对应的一个权重
    aggClassEst = mat(zeros((m, 1)))  # 统计类别估计累积值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 加入单层决策树

        # 得到运算公式中的向量+/-α，预测正确为-α，错误则+α。每条样本一个α
        # multiply对应位置相乘  这里很聪明，用-1*真实目标值*预测值，实现了错误分类则-，正确则+
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # 这三步为更新概率分布D向量 拆分开来了，每一步与公式相同
        D = D / D.sum()

        # 计算停止条件错误率=0 以及计算每次的aggClassEst类别估计累计值
        aggClassEst += alpha * classEst
        # 很聪明的计算方法 计算得到错误的个数，向量中为1则错误值
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # sign返回数值的正负符号，以1、-1表示
        errorRate = aggErrors.sum() / m  # 错误个数/总个数
        # print("错误率：", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# 预测 累加 多个弱分类器获得预测值*该alpha 得到 结果
def adaClassify(datToClass, classifierArr):  # classifierArr是元组，所以在取值时需要注意
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    # 循环所有弱分类器
    for i in range(len(classifierArr[0])):
        # 获得预测结果
        classEst = stumpClassify(dataMatrix, classifierArr[0][i]['dim'], classifierArr[0][i]['thresh'],
                                 classifierArr[0][i]['ineq'])
        # 该分类器α*预测结果 用于累加得到最终的正负判断条件
        aggClassEst += classifierArr[0][i]['alpha'] * classEst  # 这里就是集合所有弱分类器的意见，得到最终的意见
    return sign(aggClassEst)  # 提取数据符号


datArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArr = adaBoostTrainDS(datArr, labelArr, 15)

testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr, classifierArr)
errArr = mat(ones((67, 1)))  # 一共有67个样本
cnt = errArr[prediction10 != mat(testLabelArr).T].sum()
print(cnt / 67)




from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

datArr, labelArr = loadDataSet('horseColicTraining2.txt')  # 这里使用了上面获取数据的函数得到数据集和目标变量
clf = AdaBoostClassifier(n_estimators=10)  # 指定10个弱分类器
label = array(labelArr)
scores = cross_val_score(clf, mat(datArr), label)  # 模型 数据集 目标变量
print(scores.mean())