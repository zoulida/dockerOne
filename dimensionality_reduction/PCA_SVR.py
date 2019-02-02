__author__ = 'zoulida'

import pandas as pd
import numpy as np
from sklearn import datasets,decomposition,manifold

def loadData():
    #data_path = '../small_HFT1.csv'
    data_path = '/volume/HFT_XY_unselected.csv'
    csv_data = pd.read_csv(data_path)  # 读取训练数据
    #print(csv_data.shape)  # (189, 9)
    N = 5
    #csv_batch_data = csv_data.tail(N)  # 取后5条数据
    #print(csv_batch_data.shape)  # (5, 9)

    #print(csv_data)  # (5, 9)
    csv_data
    return csv_data.drop(['index', 'realY', 'predictY'], axis=1), csv_data['realY']

def transform_PCA(*data):
    X,Y=data
    pca = decomposition.PCA(n_components=20)
    #pca=decomposition.IncrementalPCA(n_components=None) #超大规模分批加载内存
    pca.fit(X)
    print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

    X_r = pca.transform(X)
    #print(X_r)
    return X_r

def SVR_train(*data):
    X, Y = data
    ####3.1决策树回归####
    from sklearn import tree
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    ####3.2线性回归####
    from sklearn import linear_model
    model_LinearRegression = linear_model.LinearRegression()
    ####3.3SVM回归####
    from sklearn import svm
    model_SVR = svm.SVR()
    model_SVR2 = svm.SVR(kernel='rbf', C=100, gamma=0.1)
    ####3.4KNN回归####
    from sklearn import neighbors
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    ####3.5随机森林回归####
    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
    ####3.6Adaboost回归####
    from sklearn import ensemble
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
    ####3.7GBRT回归####
    from sklearn import ensemble
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
    ####3.8Bagging回归####
    from sklearn.ensemble import BaggingRegressor
    model_BaggingRegressor = BaggingRegressor()
    ####3.9ExtraTree极端随机树回归####
    from sklearn.tree import ExtraTreeRegressor
    model_ExtraTreeRegressor = ExtraTreeRegressor()

    # Create the (parametrised) models
    # print("Hit Rates/Confusion Matrices:\n")
    models = [
        (
            "model_DecisionTreeRegressor", model_DecisionTreeRegressor
        ),
        (
            "model_LinearRegression", model_LinearRegression
        ),
        (
            "model_SVR", model_SVR2#model_SVR
        ),
        (
            "model_KNeighborsRegressor", model_KNeighborsRegressor
        ),
        (
            "model_RandomForestRegressor", model_RandomForestRegressor
        ),
        (
            "model_AdaBoostRegressor", model_AdaBoostRegressor
        ),
        (
            "model_GradientBoostingRegressor", model_GradientBoostingRegressor
        ),
        (
            "model_BaggingRegressor", model_BaggingRegressor
        ),
        (
            "model_ExtraTreeRegressor", model_ExtraTreeRegressor
        )
    ]

    for m in models:

        #X = X.reset_index(drop=True)
        #print(X)
        # y = y.reset_index(drop=True)
        # print(y)

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=2, shuffle=False)

        for train_index, test_index in kf.split(X):
            # print(train_index, test_index)
            # print(X.loc[[0,1,2]])

            X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[
                test_index]  # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
            #print(X_test, y_test)
            #print(X_train, y_train)
            print('======================================')

            import datetime
            starttime = datetime.datetime.now()


            print("正在训练%s模型：" % m[0])
            m[1].fit(X_train, y_train)

            # Make an array of predictions on the test set
            pred = m[1].predict(X_test)

            # Output the hit-rate and the confusion matrix for each model
            score = m[1].score(X_test, y_test)
            print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
            # print("%s\n" % confusion_matrix(y_test, pred, labels=[-1.0, 1.0]))#labels=["ant", "bird", "cat"]

            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, pred)
            print('r2: ', r2)

            endtime = datetime.datetime.now()
            print('%s训练，预测耗费时间，单位秒：'%m[0], (endtime - starttime).seconds)

            #result = m[1].predict(X_test)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(np.arange(len(pred)), y_test, 'go-', label='true value')
            plt.plot(np.arange(len(pred)), pred, 'ro-', label='predict value')
            plt.title('score: %f' % score)
            plt.legend()
            plt.show()

if __name__=="__main__":
    X, Y = loadData()
    #print(X, Y)

    X_t = transform_PCA(X, Y)

    SVR_train(X_t, Y)