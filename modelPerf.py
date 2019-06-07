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
        if correlation[i, j] > 0.1:  # Features below this threshold value are eliminated
            if columns[j]:
                columns[j] = False

columns_Selected = trainingCorrData.columns[columns]
print("Columns selected:",len(columns_Selected))
trainingData = pd.DataFrame(trainingCorrData[columns_Selected])
trainingData.insert(loc=len(columns_Selected), column="class",value=classLabel)

trainingData = trainingData.astype(float).values.tolist()
trainingData = pd.DataFrame(trainingData)
X = trainingData.iloc[:,:-1]
Y = trainingData.iloc[:,-1]

model = KNeighborsClassifier(n_neighbors=6,metric='manhattan',weights='uniform')
model.fit(X,Y)
# predictclass = model.predict(testDataSet)
# prob = model.predict_proba(testDataSet)

cv__results = cross_validate(model, X, Y, cv = 5, return_train_score= True)
print("Train score: ",cv__results['train_score'])

prec_scores = cross_val_score(model, X, Y, cv = 5, scoring='precision')
rec_scores = cross_val_score(model, X, Y, cv = 5, scoring='recall')

print("CrossValid Precision Score: ", prec_scores)
print("CrossValid Recall Scores: ", rec_scores)
numNeighbors = np.arange(5,15)
metrics = ['minkowski','euclidean','manhattan']
weights = ['uniform','distance']
param_grid = dict(n_neighbors=numNeighbors,metric=metrics,weights=weights)
cv = StratifiedKFold(5)
scores = ['precision', 'recall']

mean_precArray = []
mean_recArray = []

scores = ['precision', 'recall']
means = {'precision' :[] ,'recall': []}
stds = {'precision' :[] ,'recall': []}
for score in scores:
    grid = GridSearchCV(model,param_grid=param_grid,cv=cv, scoring='%s_macro' % score)

    grid.fit(X,Y)
    print("Tuning parameters for %s" %score)
    print("The parameters combination that would give best accuracy is : ")
    print(grid.best_params_)
    print("Grid scores:")
    means[score] = grid.cv_results_['mean_test_score']
    stds[score] = grid.cv_results_['std_test_score']
    # print("Mean_Test_Score: ", means)
    # print("Standard deviations: ", stds)

split0Score = np.array(grid.cv_results_['split0_test_score'])
split0rank = np.array(grid.cv_results_['rank_test_score'])
s0PrecScore = np.take(split0Score, np.argmin(split0rank))

split1Score = grid.cv_results_['split1_test_score']
s1PrecScore = np.take(split1Score, np.argmin(split0rank))

split2Score = grid.cv_results_['split2_test_score']
s2PrecScore = np.take(split2Score, np.argmin(split0rank))

split3Score = grid.cv_results_['split3_test_score']
s3PrecScore = np.take(split3Score, np.argmin(split0rank))

split4Score = grid.cv_results_['split4_test_score']
s4PrecScore = np.take(split4Score, np.argmin(split0rank))


#
#
# print("split0Score: ", split0Score)
# print(("split1Score: ", split1Score))
# print("Mean test score", means)
# print("Standard deviation", stds)

# plt.plot(means['recall'], means['precision'])
# plt.xlabel('recall')
# plt.ylabel('precision')
# plt.show()


# print("Precision: ",prec_scores)
# print("Recall: ", rec_scores)
# # print(predictclass)
# # print(prob)