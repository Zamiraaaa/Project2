import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from flask import Flask,request,jsonify

import numpy as np
from flask import Flask
from flask import render_template
from flask import request


# Load the file
df = pd.read_csv(r'C:\Users\User\Desktop\Model-Deploying\dataset .csv')
print(df.head())


# Select independent and dependent variable
# X = df[[""]]

from sklearn.model_selection import train_test_split

X = df.drop(['Row ID','Order ID','Order Date','Ship Date','Customer ID','Customer Name','Country','City','State',
              'Postal Code','Region','Category',
               'Product ID','Sub-Category','Product Name'],axis = 1)
y = df['Profit']


# Data Exploration
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

train_data = X_train.merge(y_train)
train_data


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.40,random_state = 101)
X_train

# print(sns.pairplot(df))


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

final = df.drop(['Row ID','Order ID','Order Date','Ship Date','Customer ID','Customer Name','Country','City','State',
              'Postal Code','Region','Category',
               'Product ID','Sub-Category','Product Name'],axis = 1)
final

from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
le = LabelEncoder()

# Encode the categorical data
df['Category'] = le.fit_transform(df['Category'])

df['Segment'] = le.fit_transform(df['Segment'])
df['Ship Mode'] = le.fit_transform(df['Ship Mode'])
df['State'] = le.fit_transform(df['State'])
df

X = df.drop(['Row ID','Order ID','Order Date','Ship Date','Sub-Category','Product Name','City','Customer ID','Customer Name','Country','Postal Code','Region','Product ID'],axis = 1)
y = df['Profit']
X

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.40,random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)



# Make pickle file of our model
pickle.dump(df,open("model.pkl","wb"))
pickle.dump(df,open("model_le_category.pkl","wb"))
pickle.dump(df,open('model_le_segment.pkl',"wb"))
pickle.dump(df,open('model_le_shipmode.pkl',"wb"))
pickle.dump(df,open("model_le_state.pkl","wb"))



# le = pickle.load(open('model_pickle.pkl', 'rb'))
# df['Profit'] = le.transform(df['Profit'])




# # Create a sample DataFrame with some data
# data = {'X': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
# df = pd.DataFrame(data)
#
# # Split data into features (X) and target (y)
# X = df[['X']]
# y = df['y']
#
# # Create and train a model
# model = LinearRegression()
# model.fit(X, y)
#
# # Use the predict method on the model to make predictions
# new_data = {'X': [6, 7, 8]}
# new_df = pd.DataFrame(new_data)
# predictions = model.predict(new_df[['X']])
# print(predictions)



from sklearn.linear_model import LinearRegression
import pandas as pd

# Create a sample DataFrame
data = {'X': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['X']]
y = df['y']

# Create and train a model
model = LinearRegression()
model.fit(X, y)

# Now, let's use the model's predict method
new_data = {'X': [6, 7, 8]}
new_df = pd.DataFrame(new_data)

# Use the model to make predictions
predictions = model.predict(new_df[['X']])

print(predictions)







