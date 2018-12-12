# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# =============================================================================
data = pd.read_csv("data2.csv") 
data.drop("GDP per Capita", axis=1, inplace=True)  
X = data.iloc[:,0:-2]
y = data.iloc[:,-2]
del data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder = LabelEncoder()
X.iloc[:,0] = labelencoder.fit_transform(X.iloc[:,0]) 
onehotencoder = OneHotEncoder(categorical_features=[0]) 
X = onehotencoder.fit_transform(X).toarray()
X  = np.delete(X,[0],1)
# =============================================================================
for i in range(0,len(X[0])):
    X[:,i] = (X[:,i] - min(X[:,i])) / (max(X[:,i]) - min(X[:,i]))    
del i
# =============================================================================

from sklearn.feature_selection import RFECV 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

model = SVC(kernel="linear")
selector = RFECV(estimator=model,step=1,cv=20)
selector = selector.fit(X_train, y_train)
print(len(selector.ranking_))

# No features dropped
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

SVMmodel = SVC(kernel="linear")
SVMmodel.fit(X_train,y_train)

parameters = [ {"C":list(np.logspace(-1,1,20)), "kernel":["linear"]},
               {"C":list(np.logspace(-1,1,20)), "kernel":["rbf"], "gamma":list(np.logspace(-3,1,20))} ]

search = GridSearchCV(estimator = SVMmodel, 
                      param_grid= parameters, 
                      scoring = "accuracy",
                      cv = 10)
search = search.fit(X_train,y_train)
print(search.best_score_)
print(search.best_params_)
y_pred = search.predict(X_test)
print()
print(accuracy_score(y_test, y_pred))
# =============================================================================
error = list()
Cvalue= list()
for i in range(0, 20):
    error.append(1-search.grid_scores_[i][1])
    Cvalue.append(search.grid_scores_[i][0]["C"])
    
plt.plot(Cvalue,error)
plt.xlabel("C value")
plt.ylabel("Error")
plt.title("C value - Error plot")
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
Clist = np.logspace(-1,1,20)
train_errors = list()
test_errors = list()
for C in Clist:
    SVMmodel = SVC(C = C, kernel="linear", random_state = 109)
    SVMmodel.fit(X_train,y_train)
    train_errors.append(1-SVMmodel.score(X_train, y_train))
    test_errors.append(1-SVMmodel.score(X_test, y_test))    
plt.plot(Clist,train_errors) 
plt.plot(Clist,test_errors)
plt.title("C values - Errors")
plt.legend(["Training", "Test"], loc='upper left')
plt.show()
