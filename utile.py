#imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from xgboost import XGBClassifier, Booster, DMatrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def correlationMatrix(data):
    data = data.dropna('columns')
    data = data[[column for column in data if data[column].nunique()>1]]#keep columns where there are more than 1 unique values
    if data.shape[1] < 2:
        print(f'No correlation plots ')
        return
    corr = data.corr()
    plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix', fontsize=15)
    plt.show()

def scatterMatrix(data):
    data = data.select_dtypes(include = [np.number])
    data = data.dropna('columns')
    data = data[[column for column in data if data[column].nunique()>1]]
    columnNames = list(data)
    if len(columnNames) > 10: #reduce the nr of column for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    data = data[columnNames]
    ax = pd.plotting.scatter_matrix(data, alpha=0.75, figsize=[30, 30], diagonal='kde')
    corrs = data.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center')
    plt.suptitle('Scatter and Density Plot')
    plt.show()

def confusionMatrix(y_pred, y_test, title):
    conf_matrix = metrics.confusion_matrix(y_pred, y_test, [1,0])
    sns.heatmap(conf_matrix, cmap='RdPu', annot=True, fmt='.0f', xticklabels=["Fraudulent", "Legitimate"], yticklabels=["Fraudulent", "Legitimate"])
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.title(title)
    plt.show()
    print("\033[1m The result is telling us that we have: ",(conf_matrix[0,0] + conf_matrix[1,1]),"correct predictions\033[1m")
    print("\033[1m We have: ",(conf_matrix[0,1] + conf_matrix[1,0]),"incorrect predictions\033[1m")
    print("\033[1m And a total predictions of: ",(conf_matrix.sum()))

def model_result(y_test, y_pred):
    print('AUPRC :', (metrics.average_precision_score(y_test, y_pred)))
    print('F1 - score :',(metrics.f1_score(y_test,y_pred)))
    print('Confusion_matrix : ')
    print(metrics.confusion_matrix(y_test,y_pred))
    print("accuracy_score")
    print(metrics.accuracy_score(y_test,y_pred))
    print("classification_report")
    print(metrics.classification_report(y_test,y_pred))

def xgboost_search(X, y, search_verbose=1):
    params = {
    "gamma":[0.5, 1, 1.5, 2, 5],
    "max_depth":[3,4,5,6],
    "min_child_weight": [100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "learning_rate": [0.1, 0.01, 0.001]
    }
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)

    grid_search = GridSearchCV(estimator=xgb, param_grid=params, scoring="roc_auc", n_jobs=1, cv=skf.split(X,y), verbose=search_verbose)

    grid_search.fit(X, y)

    print("Best estimator: ")
    print(grid_search.best_estimator_)
    print("Parameters: ", grid_search.best_params_)
    print("Highest AUC: %.2f" % grid_search.best_score_)

    return grid_search.best_params_