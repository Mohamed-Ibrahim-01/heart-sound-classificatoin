from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# SVC GridSearchCV
parameters = {'kernel' : ('poly', 'rbf'), 'C':[0.01, 0.1, 0.5, 1, 10]}
svc = svm.SVC()
clf_svm = GridSearchCV(svc, parameters)
clf_svm.fit(X_train,y_train)
print(clf_svm.best_params_)

# SVC Classifier
svc = svm.SVC(kernel ='linear',C=1)
svc.fit(X_train,y_train)
print(f"Training score = {svc.score(X_train,y_train)}")
print(f"Testing score  = {svc.score(X_test,y_test)}")
plot_confusion_matrix(svc, X_test, y_test) 
plt.show()




# RFC GridSearchCV
rfc = RandomForestClassifier()
forest_params = [{'n_estimators':list(range(10, 100)),'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]
clf_rfc = GridSearchCV(rfc, forest_params, cv = 10, scoring='accuracy')
clf_rfc.fit(X_train, y_train)
print(clf_rfc.best_params_)


# RFC Classifier
rfc = RandomForestClassifier(max_depth=7, random_state=0)
rfc.fit(X_train,y_train)
print(f"Training score = {rfc.score(X_train,y_train)}")
print(f"Testing score  = {rfc.score(X_test,y_test)}")
plot_confusion_matrix(rfc, X_test, y_test) 
plt.show()







