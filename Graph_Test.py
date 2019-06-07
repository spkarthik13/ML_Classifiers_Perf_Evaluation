import matplotlib.pyplot as plt

meanPrecScore_LDA = [0.38935682, 0.38935682, 0.38935682, 0.38935682, 0.06868139, 0.06868139, 0.38935682, 0.38935682, 0.38935682, 0.38935682, 0.06868139, 0.06868139]
meanRecScore_LDA =  [0.09467583, 0.09467583, 0.09467583, 0.09467583, 0.577845, 0.577845, 0.09467583, 0.09467583, 0.09467583, 0.09467583, 0.577845, 0.577845]
meanPrecScore_KNN = [0.6276418,  0.6174389,  0.61458399, 0.58161092, 0.62242118, 0.59655136, 0.62635902, 0.58218756, 0.62426611]
meanRecScore_KNN =  [0.31036428, 0.39089374, 0.31344573, 0.35647954, 0.31326511, 0.3449667, 0.31622604, 0.3334915,  0.30189525]
meanprecition = [0.79066453, 0.78792943, 0.79155158, 0.80153625, 0.79476015, 0.77540298,0.77981973, 0.79010388]
meanRecScore = [0.37927903, 0.39078428, 0.3850655,  0.38792489, 0.39371516, 0.39657835, 0.40805354, 0.38792869]

plt.plot( meanRecScore_LDA, meanPrecScore_LDA,  markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label="LDA")
plt.plot( meanRecScore_KNN, meanPrecScore_KNN,  color='olive', linewidth=2,label="KNN")
plt.plot( meanRecScore, meanprecition,  color='black', linewidth=2, linestyle='dashed', label="Random forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Cross-validation performance of the best models")
plt.xlim([0.3,0.42])
plt.legend()
plt.show()
print("in process")