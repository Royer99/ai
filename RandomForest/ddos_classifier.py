
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy import asarray
import pickle

#Data prep and feature drop
df = pd.read_csv('AppDDoS.csv',header=0)
df=df.drop(['srcip','srcport','dstip','dstport','proto','dscp'], axis=1)
#numerical encoding
df=df.replace(["normal","ddossim","goldeneye","hulk","rudy","slowbody2","slowheaders","slowloris","slowread"],[1,2,3,4,5,6,7,8,9])
X = df.drop(['class'], axis=1)
#split data
y = df[['class']]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20)

#classfier setup & trainning
rnd_clf = RandomForestClassifier(n_estimators=300,criterion="entropy",max_depth=8,oob_score=True,max_leaf_nodes=40,max_features=16,n_jobs=1)
rnd_clf.fit(X_train, y_train)

#save model
pickle.dump(rnd_clf, open("sklearModel", "wb"))
sklearModel = pickle.load(open("sklearModel", "rb"))

#accuracy score
y_pred_rf= sklearModel.predict(X_test)
print("random forest", accuracy_score(y_test, y_pred_rf))
print(sklearModel)
'''
test
proba=sklearModel.predict_proba(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
pred=sklearModel.predict(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
print(pred)
print(proba)
#ddossim
proba=sklearModel.predict_proba(np.array([[5,772,4,1664,40,154,568,231,52,416,1500,722,90,48,105,56,90,65,105,56,195,195,195,195,0,0,0,0,0,5,772,4,1664,1,0,0,0,256,216,1276321883,1276322078,1276322078]]))
pred=sklearModel.predict(np.array([[5,772,4,1664,40,154,568,231,52,416,1500,722,90,48,105,56,90,65,105,56,195,195,195,195,0,0,0,0,0,5,772,4,1664,1,0,0,0,256,216,1276321883,1276322078,1276322078]]))
print(pred)
print(proba)
'''


"""#XGBOOSTRandomForestClassifier"""
'''
work in progress
X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values
model = XGBRFClassifier(n_estimators=300, subsample=0.9, colsample_bynode=0.2)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model.fit(X_train, y_train)
pickle.dump(model, open("xgbModel", "wb"))
cols_when_model_builds = model.get_booster().feature_names
print(cols_when_model_builds)
#n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

xgb_model_loaded = pickle.load(open("xgbModel", "rb"))

row = [15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]
row = asarray([row])
# make a prediction
yhat = xgb_model_loaded.predict(np.array([[5,772,4,1664,40,154,568,231,52,416,1500,722,90,48,105,56,90,65,105,56,195,195,195,195,0,0,0,0,0,5,772,4,1664,1,0,0,0,256,216,1276321883,1276322078,1276322078]]))
print('Predicted Class: %d' % yhat[0])
row = [15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]
row = asarray([row])
# make a prediction
yhat = xgb_model_loaded.predict(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
print('Predicted Class: %d' % yhat[0])
'''