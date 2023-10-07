#!/usr/bin/env python
# coding: utf-8

# In[25]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

#reading the dataset
df=pd.read_csv('Car Insurance.csv',sep=',',header=0)
df

# checking any null values in the dataset
df.iloc[:,:].isnull().values.any()

#checking the count of null values in each attribute
df.iloc[:,:].isnull().sum()

#performing data imputation --> numerical data NaN replaced by using the strategy of mean 
numImputer=SimpleImputer(missing_values=np.nan,strategy='mean')
numImputer=numImputer.fit(df[['Age','Mileage']])
df[['Age','Mileage']]=numImputer.transform(df[['Age','Mileage']])

#Categorical data NaN replaced by using the strategy of most frequent
catImputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
catImputer=catImputer.fit(df[['Fuel','Gearbox','Colour']])
df[['Fuel','Gearbox','Colour']]=catImputer.transform(df[['Fuel','Gearbox','Colour']])

#obtaining dataset after imputation
df

#data-preprocessing 
#Claimed column pre-processed using Ordinal Encoder
#reshape(-1,1) function purpose makes 2D array with one column and as many rows as needed 
enc=preprocessing.OrdinalEncoder()
df['Claimed']=enc.fit_transform(df['Claimed'].values.reshape(-1,1))

# Make,Fuel,Gearbox,Colour preprocessed using One Hot Encoder where the first binary value is dropped
enc=preprocessing.OneHotEncoder(drop='first')
#writing the one hot encoded values to the array onehots
onehots=enc.fit_transform(df[['Make','Fuel','Gearbox','Colour']]).toarray()

#Creating the column names for new df
cols=[]
#calling the 'Make','Fuel','Gearbox','Colour' categories
for i in enc.categories_:
    
# deleting the first columns of the enc.categories(given below) namely 'Ford','D','A','Black'

#[array(['Ford', 'Nissan', 'Toyota'], dtype=object),
#array(['D', 'P'], dtype=object),
#array(['A', 'M'], dtype=object),
#array(['Black', 'Blue', 'Green', 'Red', 'White'], dtype=object)]
    i=np.delete(i,0)
    
# Putting the rest of the names such as 'Nissan','Toyota','P','M','Blue','Green','Red','White' into the array named cols
    cols.extend(i)
    
#joining the onehots 
df=df.join(pd.DataFrame(onehots,columns=cols))

#dropping the columns named 'Make','Fuel','Gearbox','Colour' from the original df dataset 
#axis=1 indicates dropping columns
#axis =0 indicates dropping rows
df =df.drop(['Make','Fuel','Gearbox','Colour'],axis=1)

#new dataset
df

#Standardization of the new dataset df

#making array from df: X= df all rows without last column, Y df all rows with only the last column
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

#training data is X -- this contains the mean and standard deviation values needed for scaling
scaler=preprocessing.StandardScaler().fit(X)

#all attributes becomes mean of that column
scaler.mean_

#represents the scaling factor (standard deviation) for 
#each feature that was computed when fitting the StandardScaler to the input features X
scaler.scale_

#used to apply the scaling transformation to 
#your input features X using the StandardScaler object scaler that you previously fitted. 
X_scaled=scaler.transform(X)

#initializes the MinMaxScaler from the preprocessing module. 
min_max_scaler=preprocessing.MinMaxScaler()

#applies the min-max scaling transformation to your input features X using the fit_transform method of the MinMaxScaler
X_minmax=min_max_scaler.fit_transform(X)

#Almost cleaned and standardized dataset
X_minmax

