# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
os.chdir("C:\\Users\\MONSTER\\Google Drive\\ders\\Data Mining\\project")
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
# =============================================================================
# =============================================================================

def class_fun(X,y,run=50):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    LRscore = []
    KNNscore = []
    SVMscore = []
    SVMKscore = []
    NBscore = []
    Treescore = []
    Forestscore=[]
    
    for i in range(0,run):
        X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.25,shuffle=True)
        
        LRmodel = LogisticRegression()
        LRmodel.fit(X_train,y_train)
        y_pred = LRmodel.predict(X_test)
        LRscore.append(accuracy_score(y_test, y_pred))
    
        KNNmodel = KNeighborsClassifier(n_neighbors=5)
        KNNmodel.fit(X_train,y_train)
        y_pred = KNNmodel.predict(X_test)
        KNNscore.append(accuracy_score(y_test, y_pred))
        
        SVMmodel = SVC(kernel="linear")
        SVMmodel.fit(X_train,y_train)
        SVMmodel.predict(X_test)
        SVMscore.append(accuracy_score(y_test, y_pred))

        SVMkernel = SVC(kernel="rbf")
        SVMkernel.fit(X_train,y_train)
        y_pred = SVMkernel.predict(X_test)
        SVMKscore.append(accuracy_score(y_test, y_pred))
 
        NBmodel = GaussianNB()
        NBmodel.fit(X_train,y_train)
        y_pred = NBmodel.predict(X_test)
        NBscore.append(accuracy_score(y_test, y_pred))
     
        Treemodel = DecisionTreeClassifier(criterion = "entropy")
        Treemodel.fit(X_train,y_train)
        y_pred = Treemodel.predict(X_test)
        Treescore.append(accuracy_score(y_test, y_pred))
      
        Forestmodel = RandomForestClassifier(n_estimators=10, criterion = "entropy")
        Forestmodel.fit(X_train,y_train)
        y_pred = Forestmodel.predict(X_test)
        Forestscore.append(accuracy_score(y_test, y_pred))
    
    print(np.mean(LRscore))
    print(np.mean(KNNscore))
    print(np.mean(SVMscore))
    print(np.mean(SVMKscore))
    print(np.mean(NBscore))
    print(np.mean(Treescore))
    print(np.mean(Forestscore))

# =============================================================================
class_fun(X,y)   
# =============================================================================
from sklearn.feature_selection import RFECV 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

model = SVC(kernel="linear")
selector = RFECV(estimator=model,step=1,cv=20)
selector = selector.fit(X_train, y_train)
print(len(selector.ranking_))
# =============================================================================
#without regions
class_fun(X[:,10:],y)
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
# =============================================================================
#data = pd.read_csv("GDP_column.csv", header=None) 
#X_plus = np.append(X, data, axis=1)
#class_fun(X,y)   
#class_fun(X_plus,y)   
#
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
#
#SVMmodel = SVC(kernel="linear")
#SVMmodel.fit(X_train,y_train)
#
#parameters = [ {"C":list(np.logspace(-1,2,30)), "kernel":["linear"]},
#               {"C":list(np.logspace(-1,1,20)), "kernel":["rbf"], "gamma":list(np.logspace(-3,1,20))} ]
#
#search = GridSearchCV(estimator = SVMmodel, 
#                      param_grid= parameters, 
#                      scoring = "accuracy",
#                      cv = 10)
#search = search.fit(X_train,y_train)
#print(search.best_score_)
#print(search.best_params_)
#y_pred = search.predict(X_test)
#print()
#print(accuracy_score(y_test, y_pred))

# =============================================================================

#from sklearn.decomposition import PCA
#pca = PCA(n_components=None)
#X_train_pca = pca.fit_transform(X)
#X_test_pca = pca.transform(X_test)
#variances = pca.explained_variance_ratio_
#
#pca = PCA(n_components=2)
#X_train_pca = PCA(n_components=2).fit_transform(X)
#X_test_pca = pca.transform(X_test)
#variances = pca.explained_variance_ratio_

# import xgboost 
