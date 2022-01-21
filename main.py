
"""
-- *********************************************
-- Author       :	Sharlotte Manganye
-- Create date  :   21 January 2022
-- Description  :  Deep Learning Model predicting churn for the bank
-- File Name    :  ANN.py
--*********************************************
"""

''' Libraries '''


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix

''' Importing the dataset '''
df = pd.read_csv('Churn_Modelling.csv', delimiter=',')
print(f'rows: columns \n  {df.shape}')   # rows and columns

''' Check columns list and missing values '''
print(f' Check columns list and missing values \n {df.isnull().sum()}')

''' Get unique count for each variable '''
print(f'Get unique count for each variable \n  {df.nunique()}')


'''Remove  RowNumber, CustomerId and Surname since these are specific to a customer '''

df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
print(f' New dataset \n {df.head()}')

''' Churn Data Analysis '''
labels = 'Churned', 'Retained'
sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(9, 7))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customers who churned and retained", size = 20)
plt.show()

''' Split dataset '''

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values



'''  Encoding categorical columns '''

# print(df['Gender'].unique())
# print(df['Geography'].unique())


'''  Label Encoding the "Gender" column '''
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

''' One Hot Encoding the "Geography" column '''


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


''' Splitting the dataset into the Training set and Test set '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

''' Scaling  Features '''
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

n_input = X_train.shape[1]
hidden_units = int(n_input)
fan_in = n_input


''' Initialising the ANN '''
ann = tf.keras.models.Sequential()

''' Adding the input layer and the first hidden layer '''
ann.add(tf.keras.layers.Dense(units = 6 ,kernel_initializer = 'he_uniform',activation='relu',input_dim = n_input))

''' Adding the second hidden layer '''
ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
''' Adding the output layer '''
ann.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'relu'))

''' Compiling the ANN '''
ann.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


'''Fitting the ANN to the Training set '''
model_history=ann.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0, batch_size=32)


'''list all data in history'''

print(model_history.history.keys())
''' summarize history for accuracy'''
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

''' Summarize history for loss  '''
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

''' Making the predictions and evaluating the model '''

''' Predicting the Test set results '''
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

'''Plotting Confusion Matrix '''
cm = confusion_matrix(y_test, y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['0', '1']);
ax.yaxis.set_ticklabels(['0', '1']);


''' Evaluate model '''

_, train_acc = ann.evaluate(X_train, y_train, verbose=0)
_, test_acc = ann.evaluate(X_test, y_test, verbose=0)
print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
