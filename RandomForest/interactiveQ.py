import numpy as np
from numpy import asarray
import pickle

#load the model
sklearModel = pickle.load(open("sklearModel", "rb"))
'''
testing model using xgboost (work in progress)
xgb_model_loaded = pickle.load(open("xgbModel", "rb"))
'''

#interactive query
while(1):
  print("DDOS Classifier")
  print("Please provide  a flow separated by commas")
  data= list(map(int,input().split(",")))
  data= [data]
  data = asarray(data)
  data= np.array(data)
  proba=sklearModel.predict_proba(data)
  pred=sklearModel.predict(data)
  print(pred)
  print(proba)
'''
while(1):
  print("DDOS Classifier")
  print("Please provide  a flow separated by commas")
  data= list(map(int,input().split(",")))
  data= [data]
  data = asarray(data)
  data= np.array(data)
  proba=xgb_model_loaded.predict_proba(data)
  #pred=sklearModel.predict(data)
  print(pred)
  print(proba)
'''

