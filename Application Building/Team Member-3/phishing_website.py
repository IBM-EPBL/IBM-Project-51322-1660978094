#Importing the libraries
 
import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

#Reading the dataset

ds=pd.read_csv("dataset_website.csv")
print(ds.head())


#Splitting data into independent and dependent variables


x = ds.iloc[:,1:31].values
y = ds.iloc[:,-1].values
print(x,y)


#Split the datasets into train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#Instantiate the model
lr = LogisticRegression()
lr.fit(x_train,y_train)
lr_predict= lr.predict(x_test)
print('The accurcy of Logistic Regression Model is : ', 100.0 * accuracy_score(lr_predict,y_test))
print(classification_report(lr_predict,y_test))

#Make pickle file of our model
file = "Phishing_website.pkl"
fileobject= open(file ,"wb")
pickle.dump(lr,fileobject)
fileobject.close()