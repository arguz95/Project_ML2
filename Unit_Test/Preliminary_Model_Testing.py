# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import io
import requests

url = "https://raw.githubusercontent.com/arguz95/Project_ML2/main/Data/bankrupcy.csv"
download = requests.get(url).content

bankruptcy = pd.read_csv(io.StringIO(download.decode('utf-8')))
# -

bankruptcy

bankruptcy.info()

bankruptcy.isnull().sum().sort_values(ascending = False)
# It means no NaN values in the dataframe.

bankruptcy[' Net Income Flag'].nunique()
# Column ` Net Income Flag` only has one value, so we can drop it.

bankruptcy_drop = bankruptcy.drop([' Net Income Flag'], axis=1)
#need to drop all other useless/correlated columns too

# +
# #%matplotlib inline
#import seaborn as sns; sns.set()
#sns.pairplot(df, hue='Churn', height=2);
#this hoe takes forever to run so I just killed it. This is just to vizualize data
# -

#Variable creation
x_raw = bankruptcy_drop.drop(['Bankrupt?'], axis=1)
y = bankruptcy['Bankrupt?']
x_col = [x_raw.columns]

# +
#Scaling
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#x_scaled = scaler.fit_transform(x_raw)

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

x_scaled = num_pipeline.fit_transform(x_raw)
# -

x_pd = pd.DataFrame(x_scaled)
x_pd.columns = x_col
x_pd.apply(pd.to_numeric)

import numpy as np
corr_matrix = np.corrcoef(x_pd).round(decimals=2)
print(corr_matrix)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(-1, 1)
ax.grid(False)
#ax.xaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
#ax.yaxis.set(ticks=(0, 1, 2), ticklabels=('x', 'y', 'z'))
#ax.set_ylim(2.5, -0.5)
for i in range(len(x_pd.columns)):
    for j in range(len(x_pd.columns)):
        ax.text(j, i, corr_matrix[i, j], ha='center', va='center',
                color='r')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.show()

#Anomaly Detection
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100,contamination=0.01)
pred = iforest.fit_predict(x_pd)
score = iforest.decision_function(x_pd)
from numpy import where
anom_index = where(pred==-1)
values = x_pd.iloc[anom_index]
values
for i in values.index:
    x_pd = x_pd.drop(i)
    y = y.drop(i)

x_pd.head()

x_pd.shape

# +
# Tree-based Feature importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

clf = ExtraTreesClassifier(n_estimators=50, random_state=45)
clf = clf.fit(x_pd, y)
feature_importance = clf.feature_importances_
  
# Normalizing the individual importances 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        clf.estimators_], 
                                        axis = 0) 

tbfs = pd.DataFrame(
    {"X": x_raw.columns, "FI": feature_importance_normalized}
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

# show the table results of predictors and feature importance
clf_df = pd.DataFrame(x_raw.columns,columns=['X'])
clf_df['feature_importance']=feature_importance_normalized
clf_df = clf_df.sort_values(by='feature_importance',ascending=False)
clf_df.head(30)

clf_df.tail(10)

# ### Base Model

# confirm the columns of x 
X_finalized = x_pd.drop(columns=[' Quick Assets/Current Liability',' Liability-Assets Flag',' Revenue per person',
                                 ' Net Value Growth Rate',' Revenue Per Share (Yuan ¥)'])

X_finalized.shape

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_finalized, y, test_size=0.2, random_state=101)

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
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=101)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('{}: CV accuracy mean: {:.4}'.format(name, cv_results.mean()))
        
    return names, results


# +
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

models = get_model()
names,results = bl_performance(X_train, y_train,models)
# -

# ### Preliminary Model Testing

sel_features = clf_df['X'].head(5)

X_prelim = bankruptcy[sel_features]
y_prelim = bankruptcy['Bankrupt?']

X_prelim_scaled = num_pipeline.fit_transform(X_prelim)

X_prelim_scaled.shape

LDA = LinearDiscriminantAnalysis()

LDA.fit(X_prelim,y_prelim)

import pickle as pkl
pkl.dump(LDA, open('model_v1.pkl','wb'))

sel_features


