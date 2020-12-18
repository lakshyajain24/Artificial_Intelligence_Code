#Importing Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#Reading CSV 
dataset = pd.read_csv('Churn_Modelling - Churn_Modelling.csv')

#Assigning Feature and Label
feature = dataset.iloc[:, 3:-1].values
label = dataset.iloc[:, -1].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
feature[:, 2] = le.fit_transform(feature[:, 2])
#OnehotEncoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_Transform = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
feature = np.array(column_Transform.fit_transform(feature), dtype = np.str) #fitting the dataset

#Spliting the dataset as train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2, random_state = 0)

#Normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train[:,:12])
X_test = sc.transform(X_test)

# Creating Layes for Trainng Model(ANN)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 30, epochs = 100) #Selecting size for trainning

#Testing the model 
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 50000, 2, 1, 1, 45000]])) > 0.5)
