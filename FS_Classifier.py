from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def Split_scalar(indep,dep):
    X_train,X_test,y_train,y_test=train_test_split(indep,dep,test_size=0.25,random_state=0)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    return X_train,X_test,y_train,y_test,sc

def cm_prediction(classifier,X_test,y_test):
    y_pred=classifier.predict(X_test)
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
    accuracy=accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    score=roc_auc_score(y_test,classifier.predict_proba(X_test)[:,1])
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    from sklearn.metrics import classification_report
    report=classification_report(y_test,y_pred)
    return classifier,accuracy,precision,recall,f1,score,cm,report

def logistic(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression
    grid_param={'solver':['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'],'penalty':['elasticnet','l1','l2']}
    classifier=GridSearchCV(LogisticRegression(),grid_param,refit=True,verbose=3,n_jobs=-1,scoring='f1_weighted')
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    print("The best model is {}".format(classifier.best_params_))
    return classifier,accuracy,precision,recall,f1,score,cm,report

def naive_bayes(X_train,y_train,X_test,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier=GaussianNB()
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    return classifier,accuracy,precision,recall,f1,score,cm,report

def KNN(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    param_grid={'n_neighbors':[5,10]}
    classifier=GridSearchCV(KNeighborsClassifier(),param_grid,verbose=3,refit=True,n_jobs=-1,scoring='f1_weighted')
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    print("The best model is {}".format(classifier.best_params_))
    return classifier,accuracy,precision,recall,f1,score,cm,report

def svc(X_train,y_train,X_test,y_test):
    from sklearn.svm import SVC
    param_grid={'kernel':['rbf','poly','sigmoid'],'gamma':['scale','auto'],'C':[1,10]}
    classifier=GridSearchCV(SVC(),param_grid,verbose=3,refit=True,n_jobs=-1,scoring='f1_weighted')
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    print("The best model is {}".format(classifier.best_params_))
    return classifier,accuracy,precision,recall,f1,score,cm,report

def randomforestclassifier(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestClassifier
    param_grid={'n_estimators':[100],'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2']}
    classifier=GridSearchCV(RandomForestClassifier(),param_grid,verbose=3,refit=True,n_jobs=-1,scoring='f1_weighted')
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    print("The best model is {}".format(classifier.best_params_))
    return classifier,accuracy,precision,recall,f1,score,cm,report

def decisiontreeclassifier(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    param_grid={'splitter':['best','random'],'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2']}
    classifier=GridSearchCV(DecisionTreeClassifier(),param_grid,verbose=3,refit=True,n_jobs=-1,scoring='f1_weighted')
    classifier.fit(X_train,y_train)
    classifier,accuracy,precision,recall,f1,score,cm,report=cm_prediction(classifier,X_test,y_test)
    print("The best model is {}".format(classifier.best_params_))
    return classifier,accuracy,precision,recall,f1,score,cm,report