import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

columns=["T","AP","RH","V","EP"]
df = pd.read_csv('Folds5x2_pp.csv',names = columns,skiprows=1)
#split features and label
df_x=df[["T","AP","RH","V"]]
df_y=df[["EP"]]

images_dir = './plotsP'
#data exploration
fig=plt.figure(1)
plt.scatter(df[["T"]],df[["EP"]],marker=1)
fig.savefig(f"{images_dir}/T.png")
fig2=plt.figure(2)
plt.scatter(df[["AP"]],df[["EP"]],marker=2)
fig2.savefig(f"{images_dir}/AP.png")
fig3=plt.figure(3)
plt.scatter(df[["RH"]],df[["EP"]],marker=3)
fig3.savefig(f"{images_dir}/RH.png")
fig4=plt.figure(4)
plt.scatter(df[["V"]],df[["EP"]],marker=4)
fig4.savefig(f"{images_dir}/V.png")
#based on the observations, the temperature variable is highly related to the target class

#split the data using the 80-20 rule
numberOfInstances=9568
fd_cement=df_x[["T"]]
x_train=fd_cement[:round(80*numberOfInstances/100)+10].to_numpy()
y_train=df_y[:round(80*numberOfInstances/100)+10].to_numpy()
x_test=fd_cement[round(80*numberOfInstances/100)-10:].to_numpy()
y_test=df_y[round(80*numberOfInstances/100)-10:].to_numpy()
#print(y_train.shape[0])
#print(x_test.shape[0])
#print(y_test.shape[0])

n=x_train.shape[0]
sum1=0
sum2=0
sum3=0
sum4=0
for x in range(n):
  sum1+=(x_train[x]*y_train[x])
  sum2+=x_train[x]*x_train[x]
  sum3+=x_train[x]
  sum4+=y_train[x]
#print(n)
#print(sum1)
#print(sum2)
#print(sum3)
#print(sum4)
xhat=sum3/n
yhat=sum4/n
#print(xhat)
#print(yhat)
print(((n*sum2)-(sum3*sum3)))
a1=((n*sum1)-(sum3*sum4))/((n*sum2)-(sum3*sum3))
a0=yhat-(a1*xhat)
print("a1",a1)
print("a0",a0)
y=a0+a1*x_train
#plt.scatter(df[["T"]],df[["EP"]],marker=1)
fig5=plt.figure(5)
plt.plot(x_train,y,label='y = a0+a1*x')
plt.xlabel('Temperature (T) C°')
plt.ylabel('Net hourly electrical energy (EP) MW')
plt.legend()
plt.title('Linear regression plot')
fig5.savefig(f"{images_dir}/lin.png")

#interactive query
while(1):
  print("please provide a temperature to predict the Net hourly electrical energy")
  print("ranges from 1.81°C and 37.11°C")
  temp=float(input())
  res=a0+a1*temp
  print(res)
