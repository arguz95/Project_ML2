# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <img src="https://nserc-hi-am.ca/2020/wp-content/uploads/sites/18/2019/12/McGill.png" width="500" height="400" align="left">
#
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
#
#
# # INSY 695: Final Group Project
# #### Arnaud Guzman-Annès | ID: 260882529
# #### Jules Zielinski Babu | ID: 
# #### Ram Babu | ID: 260958970
# #### Dorothy Zou | ID: 260950477
# #### Rameez Rosul | ID: 
# #### Johnny Qiao | ID: 
#
# <br>
# <br>
#
# **Date: April 1st, 2021**
# <br>
# <br>

# +
## Initial setup

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import pandas as pd
import seaborn as sns

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
pd.options.mode.chained_assignment = None  # default='warn'

#display all columns
pd.set_option('display.max_columns', None)

# +
# Sets seed for the entire notebook

np.random.seed(42)
# -

# Import data
#df = pd.read_csv("bankruptcy.csv")
df = pd.read_csv(r"C:\Users\Dell\Desktop\bankrupcy.csv")
bankruptcy = df.copy() # we can use this as raw data afterwards

df.head()

df.info()

# +
# Some more information about the dataset

display(df.shape)
display(df.isnull().sum())
display(df.describe())

# +
# Cheking for unique values

display(df[' Liability-Assets Flag'].nunique())
display(df[' Net Income Flag'].nunique())

# +
# Dropping these 2 columnds

# Liability-Assets Flag
# Net Income Flag

df[[' Liability-Assets Flag']].value_counts() 
df[' Liability-Assets Flag'].corr(df['Bankrupt?'])

#df = df.drop([' Liability-Assets Flag'], axis=1)
df = df.drop([' Net Income Flag'], axis=1)

# +
# We are checking for imbalanced data
# Print figure

import seaborn as sns

plt.figure(figsize=(5,5))
splot = sns.countplot(data = df,x = 'Bankrupt?',palette = 'Blues')
sns.set_style('ticks')
total = float(len(df))
for p in splot.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    splot.annotate(percentage, (x,y), ha = 'center', va = 'center')
plt.title("Bankrupt?")
plt.xlabel("Bankrupt")
plt.ylabel("Number of companies")

# +
# We are going to work later on on this issue.

# +
# Correlation

corr = df.corr()
fig, ax = plt.subplots(figsize = (15,15))
sns.heatmap(corr, ax = ax, cmap = 'viridis', linewidth = 0.1)


# -

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return(dataset)



correlation(df, 0.65)

filtered_col = correlation(df, 0.65).columns
df = df[filtered_col]
df

corr = df.corr()
fig, ax = plt.subplots(figsize = (15,15))
sns.heatmap(corr, ax = ax, cmap = 'viridis', linewidth = 0.1)

# Variable creation
X = df.drop(['Bankrupt?'], axis=1)
y = df['Bankrupt?']

# +
# feature selection (tree-based)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

clf = ExtraTreesClassifier(n_estimators=50, random_state=45)
clf = clf.fit(X, y)
feature_importance = clf.feature_importances_
  
# Normalizing the individual importances 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        clf.estimators_], 
                                        axis = 0) 

tbfs = pd.DataFrame(
    {"X": X.columns, "FI": feature_importance_normalized}
)

tbfs = tbfs.sort_values('FI',ascending=True)


# Plotting a Bar Graph to compare the models 
plt.figure(figsize=(15,25))
plt.barh(y=tbfs['X'],width=tbfs['FI']) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()

# -

new_col = tbfs['X'][-10:]
X = X[new_col]
X

display(X.shape)
display(y.shape)

#Scaling
from sklearn.preprocessing import StandardScaler
x_col = X.columns
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
X.columns = x_col
#X.apply(pd.to_numeric)
display(X.head())

# +
#Anomaly Detection
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100,contamination=0.01)
pred = iforest.fit_predict(X)
score = iforest.decision_function(X)
from numpy import where
anom_index = where(pred==-1)
values = X.iloc[anom_index]

for i in values.index:
    X = X.drop(i)
    y = y.drop(i)
# -

display(X.shape)
display(y.shape)

# Splitting the data set to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# +
# pip install imblearn
# #!pip install imblearn
# #!pip install delayed
# #!pip install delayed

# +
# Dealing with imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy = 0.2)
under = RandomUnderSampler(sampling_strategy = 0.6)

steps = [('o',over),('u',under)]
pipeline = Pipeline(steps = steps)

X_train, y_train = pipeline.fit_resample(X_train, y_train)
# -

display(y_train.shape)
display(X_train.shape)

count = 0
for i in y_train:
    if i == 1:
        count +=1
print("There are",count,"cases of bankruptcy and",len(y_train)-count,"of non-bankruptcy")

y_train = pd.DataFrame(y_train) 

plt.figure(figsize=(5,5))
splot = sns.countplot(data = y_train, x = 'Bankrupt?', palette = 'Blues')
sns.set_style('ticks')
total = float(len(df))
plt.title("Bankrupt?")
plt.xlabel("Bankrupt")
plt.ylabel("Number of companies")

"""# Modelling with balanced target 

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
model.fit(X_train, y_train)

sel = SelectFromModel(model)
sel.fit(X_train, y_train)

selected_feat= X_train.columns[(sel.get_support())]"""

# +
#X_train = selected_feat
# -

# ### Base Model

# +
# Find some classification models and check their baseline accuracy with 10 CV folds

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


def get_model():
    models = []
    models.append(('LR' , LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('RF' , RandomForestClassifier(n_estimators=100)))
    models.append(('MLP', MLPClassifier()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    return models

def bl_performance(X_train, y_train,models):
    results = []
    names = []
    acc = []
    f1 = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') 
        cv_f1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
        results.append(cv_results)
        names.append(name)
        print('{}: CV accuracy mean: {:.4}'.format(name, cv_results.mean()))
        print('{}: CV F1 score mean: {:.4}'.format(name, cv_f1.mean()))
        acc += [cv_results.mean()]
        f1 += [cv_f1.mean()]
        
    result_df = pd.DataFrame()
    result_df['Model'] = ['LR','KNN','SVM','GBC','RF','MLP','LDA']
    result_df['Accuracy'] = acc
    result_df['F1'] = f1
        
    return names, results, result_df


# +
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

models = get_model()
names,results, result_df= bl_performance(X_train, y_train,models)
# -

result_df

# ## MLflow

# +
import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
# -

# ### Keras Neural Network Model

# +
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[10]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(2, activation="softmax"))

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
               metrics=['acc',f1_m])

history = model.fit(X_train, y_train, epochs=8,  # epoch=30
                    validation_data=(X_test, y_test))
# -

X11= X_train[["Debt ratio %","ROA(C) before interest and depreciation before interest","Working Capital to Total Assets","Borrowing dependency","Total debt/Total net worth","Net Value Per Share (B)"]]

# +
#Preprocessing 
sns.heatmap(X11.isnull(), cbar=False)
X11 = X11.dropna()
X11.info()


# -

#X11.set_index('index', inplace=True)
# Printing the dataframe
X11.reset_index(level=0, inplace=True)
X11

#Elbow process and standarization
from sklearn.preprocessing import StandardScaler
import numpy
scaler = StandardScaler()
X_std11= scaler.fit_transform(X11) 
from sklearn.cluster import KMeans
withinss = []
for i in range (2,8):
     kmeans = KMeans(n_clusters=i)
     model = kmeans.fit(X_std11)
     withinss.append(model.inertia_)
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],withinss)


# +
#silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(X_std11)
labels = model.labels_
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std11,labels)
silhouette_avg = silhouette_score(X_std11,labels)
print("The average silhouette_score is :", silhouette_avg)

import pandas
df4 = pandas.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 0: ',np.average(df4[df4['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',np.average(df4[df4['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(df4[df4['label'] == 2].silhouette))
#print('Average Silhouette Score for Cluster 3: ',np.average(df4[df4['label'] == 3].silhouette))
#print('Average Silhouette Score for Cluster 4: ',np.average(df4[df4['label'] == 4].silhouette))
#print('Average Silhouette Score for Cluster 5: ',np.average(df4[df4['label'] == 5].silhouette))

# -

#silhouette score
from sklearn.metrics import silhouette_score
silhouette_score(X_std11,labels)


#Data retrival and merger 
def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])


ab0=ClusterIndicesComp(0, labels)
dataframe0=pd.DataFrame(ab0, columns=["index"]) 
print (dataframe0)
dataframe0.info()
result0 = pd.merge(dataframe0,X11,on='index',how="left")

m0=result0.median()
des0=result0.describe()
des0


ab1=ClusterIndicesComp(1, labels)
dataframe1=pd.DataFrame(ab1, columns=["index"]) 
print (dataframe1)
dataframe1.info()
result1 = pd.merge(dataframe1,X11,on='index',how="left")

m1=result1.median()
des1=result1.describe()
des1

ab2=ClusterIndicesComp(2, labels)
dataframe2=pd.DataFrame(ab2, columns=["index"]) 
print (dataframe2)
dataframe2.info()
result2 = pd.merge(dataframe2,X11,on='index',how="left")

m2=result2.median()
des2=result2.describe()
des2


