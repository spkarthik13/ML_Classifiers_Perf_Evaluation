import numpy as np
import pandas as pd
from scipy import stats
import random
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
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
        if correlation[i, j] > 0:  # Features below this threshold value are eliminated
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

model = KNeighborsClassifier(n_neighbors=6,metric='manhattan',weights='uniform')
model.fit(X,Y)

# cv__results = cross_validate(model, X, Y, cv = 5, return_train_score= True)
# print("Train score: ",cv__results['train_score'])
#
# prec_scores = cross_val_score(model, X, Y, cv = 5, scoring='precision')
# rec_scores = cross_val_score(model, X, Y, cv = 5, scoring='recall')
#
# print("CrossValid Precision Score: ", prec_scores)
# print("CrossValid Recall Scores: ", rec_scores)

numNeighbors = np.arange(5,15)
metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance']
param_grid = dict(n_neighbors=numNeighbors,metric=metrics,weights=weights)
cv = StratifiedKFold(5)
scores = ['precision', 'recall']


grid1 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='precision')

grid1.fit(X,Y)
print("Tuning parameters for precision")
print("The parameters combination that would give best accuracy is : ")
print(grid1.best_params_)
print("Grid scores:")

# precArray = []

# split0Score = np.array(grid1.cv_results_['split0_test_score'])
# split0rank = np.array(grid1.cv_results_['rank_test_score'])
# s0PrecScore = np.take(split0Score, np.argmin(split0rank))
#
# split1Score = grid1.cv_results_['split1_test_score']
# s1PrecScore = np.take(split1Score, np.argmin(split0rank))
#
# split2Score = grid1.cv_results_['split2_test_score']
# s2PrecScore = np.take(split2Score, np.argmin(split0rank))
#
# split3Score = grid1.cv_results_['split3_test_score']
# s3PrecScore = np.take(split3Score, np.argmin(split0rank))
#
# split4Score = grid1.cv_results_['split4_test_score']
# s4PrecScore = np.take(split4Score, np.argmin(split0rank))
#
# Prec_scores = [s0PrecScore,s1PrecScore,s2PrecScore,s3PrecScore,s4PrecScore]
# for score in Prec_scores:
#     precArray.append(score)

meanPrecScore = grid1.cv_results_['mean_test_score']
precRank = grid1.cv_results_['rank_test_score']
# print("Precision: ",precArray)
# meanScore = np.take(meantest, np.argmin(split0rank))

# stdScore = np.take(stdTest, np.argmin(split0rank))
std_Prec = grid1.cv_results_['std_test_score']

DeviationPrecision = np.take(std_Prec, np.argmin(precRank))

print("Standard Deviation of precision: ", DeviationPrecision)

print('-----------------------------------------------------------------------------')

grid2 = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='recall')

grid2.fit(X,Y)
print("Tuning parameters for recall")
print("The parameters combination that would give best accuracy is : ")
print(grid2.best_params_)
print("Grid scores:")

# split0ScoreRec = np.array(grid2.cv_results_['split0_test_score'])
# split0rankRec = np.array(grid2.cv_results_['rank_test_score'])
# s0PrecScoreRec = np.take(split0ScoreRec, np.argmin(split0rankRec))
#
# split1ScoreRec = grid2.cv_results_['split1_test_score']
# s1PrecScoreRec = np.take(split1ScoreRec, np.argmin(split0rankRec))
#
# split2ScoreRec = grid2.cv_results_['split2_test_score']
# s2PrecScoreRec = np.take(split2ScoreRec, np.argmin(split0rankRec))
#
# split3ScoreRec = grid2.cv_results_['split3_test_score']
# s3PrecScoreRec = np.take(split3ScoreRec, np.argmin(split0rankRec))
#
# split4ScoreRec = grid2.cv_results_['split4_test_score']
# s4PrecScoreRec = np.take(split4ScoreRec, np.argmin(split0rankRec))
#
# rec_Scores = [s0PrecScoreRec, s1PrecScoreRec, s2PrecScoreRec, s3PrecScoreRec, s4PrecScoreRec]
#
# for rscore in rec_Scores:
#     recArray.append(rscore)
#
# print("Recall: ", recArray)
#
# precision = np.sort(precArray)
# recall = np.sort(recArray)

meanRecScore = grid2.cv_results_['mean_test_score']
recRank = grid2.cv_results_['rank_test_score']
# meanScoreRec = np.take(meantestRec, np.argmin(recRank))

std_rec = grid2.cv_results_['std_test_score']

DeviationRecall = np.take(std_rec, np.argmin(recRank))

print("Standard Deviation of Recall: ", DeviationRecall)
# print("std Recall: ", std_rec)
# stdTestRec = grid2.cv_results_['std_test_score']
# stdScoreRec = np.take(stdTestRec, np.argmin(recRank))


# print("Mean score precision from grid: ", meanScore)
# print("Std score precision from grid: ", stdScore)

# print("Mean score precision from grid: ", meanScoreRec)
# print("Std score precision from grid: ", stdScoreRec)

# plt.step(recArray, precArray, marker = 'o', color = 'r')

plt.fill_between(meanRecScore, meanPrecScore, step="pre")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


prob = model.predict_proba(testData)
print("Writing probabilities to file")
file = open("knnProb.txt","w+")
probClass1 = prob[:,-1]
for i in probClass1:
    file.write(str(i))
    file.write("\n")
file.close()
#print(prob.iloc[:,-1])