import numpy as np
import pandas as pd
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import sys

trainingDataSet = pd.read_csv(sys.argv[1],delimiter="\t", header=None)
testDataSet = pd.read_csv(sys.argv[2],delimiter="\t", header=None)

classLabel = trainingDataSet.iloc[:,-1]
trainingCorrData = trainingDataSet.iloc[:,:-1]

correlation, pValue = stats.spearmanr(trainingCorrData)
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation[i, j] >= 0.2:  # Features below this threshold value are eliminated
            if columns[j]:
                columns[j] = False

columns_Selected = trainingCorrData.columns[columns]
print("Columns selected:",len(columns_Selected))
trainingData = pd.DataFrame(trainingCorrData[columns_Selected])
trainingData.insert(loc=len(columns_Selected), column="class",value=classLabel)
testData = pd.DataFrame(testDataSet[columns_Selected])

trainingData = trainingData.astype(float).values.tolist()
trainingData = pd.DataFrame(trainingData)
X = trainingData.iloc[:,:-1]
Y = trainingData.iloc[:,-1]

model = RandomForestClassifier()
model.fit(X,Y)

# 'min_samples_split': [1.0, 10],
#     'min_samples_leaf' : [0.5, 5],
param_grid = {
    'n_estimators': [750, 800, 850, 900],
    'max_depth' : [12,15]
}
cv = StratifiedKFold(5,shuffle=True)


grid1 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='precision')

grid1.fit(X,Y)
print("Tuning parameters for precision")
print("The parameters combination that would give best accuracy is : ")
best_params = grid1.best_params_
print(grid1.best_params_)
print("Grid scores:")
# RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'])
meanPrecScore = grid1.cv_results_['mean_test_score']
#precRank = grid1.cv_results_['rank_test_score']

std_Prec = grid1.cv_results_['std_test_score']
meanprecition = np.average(meanPrecScore)
DeviationPrecision = np.average(std_Prec)
print("meanprecition", meanPrecScore)
print("Mean of precitions: ",meanprecition)
print("Standard Deviation of precision: ", DeviationPrecision)

print('-----------------------------------------------------------------------------')

grid2 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='recall')

grid2.fit(X,Y)
print("Tuning parameters for recall")
print("The parameters combination that would give best accuracy is : ")
print(grid2.best_params_)
print("Grid scores:")

meanRecScore = grid2.cv_results_['mean_test_score']
#recRank = grid2.cv_results_['rank_test_score']
print("meanRecScore : ",meanRecScore)
std_rec = grid2.cv_results_['std_test_score']
DeviationRecall = np.average(std_rec)
meanRecall = np.average(meanRecScore)
print("Mean of recall: ", meanRecall)
print("Standard Deviation of Recall: ", DeviationRecall)
# print("std Recall: ", std_rec)

#plt.step(meanRecScore, meanPrecScore)
#plt.fill_between(meanRecScore, meanPrecScore, step="pre")
# plt.plot(meanRecScore, meanPrecScore)
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title('randomForest')
# plt.ylim(0.5,0.85)
# plt.show()

modelChosen = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'])
modelChosen.fit(X,Y)
featureImp = modelChosen.feature_importances_
print(featureImp)

prob = modelChosen.predict_proba(testData)
probAvg = np.average(prob)
print("Average prediction Probability: ", probAvg)
print("Writing probabilities to file")

file = open("RandomForestProb.txt","w+")
probClass1 = prob[:,-1]
for i in probClass1:
    file.write(str(i))
    file.write("\n")
file.close()