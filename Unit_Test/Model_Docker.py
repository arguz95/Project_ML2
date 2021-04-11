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
import io
import requests
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
pd.set_option('display.max_rows', None)

# +
# Sets seed for the entire notebook

np.random.seed(42)

# +
# Import data

url = "https://raw.githubusercontent.com/arguz95/Project_ML2/master/Data/bankrupcy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
bankruptcy = df.copy() # we can use this as raw data afterwards
# -

X = bankruptcy.drop(columns='Bankrupt?')
y = bankruptcy['Bankrupt?']

# ## Preparing Data for ML models
# *Splitting the data*

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve

# +
from sklearn.model_selection import train_test_split

X_train, X_test_final, y_train, y_test_final = train_test_split(X,y, test_size=0.2, random_state=42)

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.25, random_state=42)
# -

X_train.shape

# +
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([('std_scaler', StandardScaler())])

num_attribs = list(X_train)

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs)])
# -

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
X_test_final_prepared = full_pipeline.transform(X_test_final)

X_train_prepared = pd.DataFrame(X_train_prepared, columns=num_attribs)
X_test_prepared = pd.DataFrame(X_test_prepared, columns=num_attribs)
X_test_final_prepared = pd.DataFrame(X_test_final_prepared, columns=num_attribs)

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

y_train.reset_index(inplace=True)
y_train.drop(columns='index',axis=1,inplace=True)
y_test.reset_index(inplace=True)
y_test.drop(columns='index',inplace=True)
y_train.head()

# +
# Dealing with imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


over_sample=SMOTE()
X_train_prepared, y_train=over_sample.fit_resample(X_train_prepared,y_train)
# -

import rfpimp
from rfpimp import plot_corr_heatmap
limit = 0.85
corr = X_train_prepared.corr()

mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
corr_no_diag = corr.where(mask)

coll = [c for c in corr_no_diag.columns if any(abs(corr_no_diag[c]) > limit)]

# ### Variance threshold

# +
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

sel.fit(X_train_prepared)
var = sel.get_support()

col2 = []
for i in range(94):
    if not var[i]:
        col2.append(num_attribs[i])
# -

## Highly corelated columns with variance > 0.16
col_del = coll.copy()
for i in coll:
    if i not in col2:
        col_del.append(i)

## 31 columns removed
X_train_prepared_old = X_train_prepared.copy()
X_train_prepared.drop(columns=col_del,axis=1,inplace=True)

# ### Tree-Based Feature Selection

# +
# feature selection (tree-based)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

clf = ExtraTreesClassifier(n_estimators=50, random_state=45)
clf = clf.fit(X_train_prepared, y_train)
feature_importance = clf.feature_importances_
  
# Normalizing the individual importances 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        clf.estimators_], 
                                        axis = 0) 

tbfs = pd.DataFrame(
    {"X": X_train_prepared.columns, "FI": feature_importance_normalized}
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

chosen_features = tbfs.nlargest(5,'FI')

# #### Selected Features

chosen_features

## Training and test set with selected features
X_train_prepared = X_train_prepared[chosen_features['X']]
X_test_prepared = X_test_prepared[chosen_features['X']]
X_test_final_prepared = X_test_final_prepared[chosen_features['X']]



# ## MLflow

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from IPython.display import Image
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMModel,LGBMClassifier

hyperparameters = {"max_depth":scope.int(hp.quniform("max_depth",2,100,5)),
                "n_estimators":scope.int(hp.quniform("n_estimators",2,100,1)),
                "num_leaves": scope.int(hp.quniform("num_leaves",2,50,1)),
                "reg_alpha": hp.loguniform('reg_li',-5,5),
                "random_state":1,
                "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
                "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
                "boosting": hp.choice("boosting",["gbdt","dart","goss"]),
                "objective":"binary"}


# +
def train_model(parameters):
    mlflow.lightgbm.autolog()
    with mlflow.start_run(nested=True):
        booster = lgb.LGBMClassifier()
        booster.set_params(**parameters)
        booster.fit(X_train_prepared,y_train)
        
        mlflow.log_params(parameters)
        
        score = cross_val_score(booster, X_train_prepared, y_train, cv=5, scoring = "f1_macro",n_jobs=-1)
        mean_score = np.mean(score)
        
        mlflow.log_metric('f1_macro', mean_score)
        
        return{'status':STATUS_OK,
               "loss":mean_score,
               'booster':booster.get_params}
    
with mlflow.start_run(run_name='lightgbm_bankruptcy'):
    best_params = fmin(
        fn=train_model,
        space=hyperparameters,
        algo=tpe.suggest,
        max_evals = 50,
        trials = Trials(),
        rstate=np.random.RandomState(1))
# -

# ## Real Model

import mlflow
df = mlflow.search_runs(filter_string="metric.f1_macro > 0.8")

df

df.sort_values(by='metrics.f1_macro').iloc[0]

params = df.sort_values(by='metrics.f1_macro').iloc[0,7:16].to_dict()

import lightgbm

train_data = lightgbm.Dataset(X_train_prepared, label=y_train)
valid_data = lightgbm.Dataset(X_test_prepared, label=y_test)
test_data = lightgbm.Dataset(X_test_final_prepared, label=y_test_final)

X_train_prepared.shape

type(X_test_final_prepared.iloc[0,0])

parameters = {
    'objective': params["params.objective"],
    'boosting': params["params.boosting"],
    'num_leaves': int(params["params.num_leaves"]),
    'learning_rate': float(params["params.learning_rate"]),
    'n_estimators': int(params["params.n_estimators"]),
    'min_child_weight': float(params["params.min_child_weight"]),
    'random_state': 1,
    'reg_alpha':float(params["params.reg_alpha"]),
    'max_depth': int(params["params.max_depth"])
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=valid_data)

y_pred = model.predict(X_test_final_prepared)
y_pred=y_pred.round(0)
y_pred=y_pred.astype(int)

from sklearn.metrics import f1_score
f1_score(y_test_final, y_pred, average='weighted')

t = X_test[chosen_features['X']]

t.shape

t.reset_index(inplace=True)

t.drop(columns='index',inplace=True)

temp = model.predict(t.iloc[0:3,:])

temp

# ## Pickling the model

import pickle as pkl
pkl.dump(model, open('model.pkl','wb'))

# +
# 'weighted':
# Calculate metrics for each label, and find their average weighted by support 
#(the number of true instances for each label). This alters ‘macro’ to account 
#for label imbalance; it can result in an F-score that is not between precision and recall.
