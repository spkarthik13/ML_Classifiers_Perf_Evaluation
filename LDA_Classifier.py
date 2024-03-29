import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

trainingDataSet = pd.read_csv("A3_training_dataset.tsv",delimiter="\t", header=None)
testDataSet = pd.read_csv("A3_test_dataset.tsv",delimiter="\t", header=None)

classLabel = trainingDataSet.iloc[:,-1]
trainingCorrData = trainingDataSet.iloc[:,:-1]

correlation, pValue = stats.spearmanr(trainingCorrData)
columns = np.full((correlation.shape[0],), True, dtype=bool)
for i in range(correlation.shape[0]):
    for j in range(i + 1, correlation.shape[0]):
        if correlation[i, j] >= 0:  # Features below this threshold value are eliminated
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

model = LinearDiscriminantAnalysis()
model.fit(X,Y)


param_grid = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'n_components': [0, 1],
    'store_covariance': [True, False]
}

cv = StratifiedKFold(5,shuffle=True)


grid1 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='precision')

grid1.fit(X,Y)
print("Tuning parameters for precision")
print("The parameters combination that would give best accuracy is : ")
print(grid1.best_params_)
print("Grid scores:")

meanPrecScore = grid1.cv_results_['mean_test_score']
print("meanPrecScore",meanPrecScore)
precRank = grid1.cv_results_['rank_test_score']

std_Prec = grid1.cv_results_['std_test_score']
meanPrecScore_avg = np.average(meanPrecScore)
DeviationPrecision = np.average(std_Prec)

print("Standard Deviation of precision: ", DeviationPrecision)

print('-----------------------------------------------------------------------------')

grid2 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='recall')

grid2.fit(X,Y)
print("Tuning parameters for recall")
print("The parameters combination that would give best accuracy is : ")
print(grid2.best_params_)
print("Grid scores:")


meanRecScore = grid2.cv_results_['mean_test_score']
print("meanRecScore : ", meanRecScore)
recRank = grid2.cv_results_['rank_test_score']

std_rec = grid2.cv_results_['std_test_score']
meanRecScore_avg = np.average(meanRecScore)
print(meanRecScore_avg)
DeviationRecall = np.average(std_rec)

print("Standard Deviation of Recall: ", DeviationRecall)
# print("std Recall: ", std_rec)


plt.plot(meanRecScore, meanPrecScore)
plt.title("LDA Classifier")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.2,0.8])
plt.show()
#
modelChosen = LinearDiscriminantAnalysis(n_components=0, solver="svd", store_covariance=True)
modelChosen.fit(X,Y)

prob = modelChosen.predict_proba(testData)

probAvg = np.average(prob)

print("Average prediction Probability: ", probAvg)
print("Writing probabilities to file")
file = open("LDA_Prob.txt","w+")
probClass1 = prob[:,-1]
for i in probClass1:
    file.write(str(i))
    file.write("\n")
file.close()
#print(prob.iloc[:,-1])