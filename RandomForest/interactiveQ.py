import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#from google.colab import files
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#import graphviz
from numpy import asarray
#from sklearn_porter import Porter
import ipaddress
#from xgboost import XGBRFClassifier
#from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
#import xgboost as xgb
import pickle


sklearModel = pickle.load(open("sklearModel", "rb"))
#xgb_model_loaded = pickle.load(open("xgbModel", "rb"))


while(1):
  print("DDOS Classifier")
  print("Please provide  a flow separated by commas")
  data= list(map(int,input().split(",")))
  data= [data]
  data = asarray(data)
  data= np.array(data)
  #print(data)
  #print(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
  proba=sklearModel.predict_proba(data)
  pred=sklearModel.predict(data)
  #proba=sklearModel.predict_proba(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
  #pred=sklearModel.predict(np.array([[15,4038,13,1836,52,269,1500,507,52,141,628,216,1,25,299,79,1,27,299,85,351,351,351,351,0,0,0,0,0,15,4038,13,1836,1,1,0,0,788,684,1276318784,1276319135,1276319111]]))
  print(pred)
  print(proba)
  #print(pred)
  #print(proba)


