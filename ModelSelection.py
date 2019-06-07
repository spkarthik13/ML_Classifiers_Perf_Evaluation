import numpy as np
import pandas as pd
from scipy import stats
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

################################removing highest co related values######################################################
trainingDataSet = pd.read_csv("A3_training_dataset.tsv",delimiter="\t", header=None)
testDataSet = pd.read_csv("A3_test_dataset.tsv",delimiter="\t", header=None)

classLabel = trainingDataSet.iloc[:,-1]
trainingCorrData = trainingDataSet.iloc[:,:-1]

correlation, pValue = stats.spearmanr(trainingCorrData)
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation[i, j] > 0.3:  # Features below this threshold value are eliminated
            if columns[j]:
                columns[j] = False

columns_Selected = trainingCorrData.columns[columns]
print("Columns selected:",len(columns_Selected))
# trainingData = pd.DataFrame(trainingCorrData[columns_Selected])
# trainingData.insert(loc=len(columns_Selected), column="class",value=classLabel)
#
# trainingData = trainingData.astype(float).values.tolist()
# random.shuffle(trainingData)
# trainingData = pd.DataFrame(trainingData)
#
# X = trainingData.iloc[:,:-1]
# Y = trainingData.iloc[:,-1]
#
model = KNeighborsClassifier()
scores = cross_val_score(model, X, Y, cv=5)
numNeighbors  = np.arange(5,15)
# rfe = RFE(model, 3)
# rfe = rfe.fit(trainingCorrData,classLabel)
# print("RFE Support: ",rfe.support_)
# print("RFE Ranking: ",rfe.ranking_)
param_grid    = dict(n_neighbors=numNeighbors)
cv            = StratifiedKFold(5)
grid = GridSearchCV(model,param_grid=param_grid,cv=cv)
# #grid.fit(dataImpNew,yNew)
# grid.fit(X,Y)
# print("The parameters combination that would give best accuracy is : ")
# print(grid.best_params_)
# print("The best accuracy achieved after parameter tuning via grid search is : ", grid.best_score_)