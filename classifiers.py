import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = [30,40]
LOG_SPACE = np.logspace(start = 0, stop = 10, num = 100, base = 2 , dtype = 'float64') 

classifiers = [
    {
        'name':'SVM',
        'parameters':{
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'C':np.logspace(start = 0, stop = 60, num = 100, base = 2 , dtype = 'float64')
        },
        'cached': SVC(kernel ='rbf', C=50),
        'method':SVC
    },
    {
        'name':'Random Forest' ,
        'parameters':{
            'max_depth':[1,2,3,4,5],
            'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120, 300, 400, 600],
            'max_features':[1,2,3]
        },
        'cached': RandomForestClassifier(max_depth=6, n_estimators=50),
        'method':RandomForestClassifier
    },
    {
        'name':'XGboost' ,
        'parameters':{
            'max_depth':[1,2,3,4,5],
            'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120, 300, 400, 600],
            'max_features':[1,2,3]
        },
        'cached': XGBClassifier(max_depth=6, n_estimators=50),
        'method':XGBClassifier
    },

]
