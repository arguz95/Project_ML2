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
import matplotlib
matplotlib.use('TkAgg')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
pd.options.mode.chained_assignment = None  # default='warn'

# +
# Sets seed for the entire notebook

np.random.seed(42)
# -

# ## Fair ML

from sklearn.linear_model import LogisticRegression
from fairml import audit_model
from fairml import plot_dependencies

# +
# Import data

url = "https://raw.githubusercontent.com/arguz95/Project_ML2/master/Data/bankrupcy.csv"
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
bankruptcy = df.copy() # we can use this as raw data afterwards
bankruptcy['Bankrupt']=bankruptcy['Bankrupt?']
bankruptcy.drop("Bankrupt?", 1,inplace=True)
bankruptcy.head()

# +
# create feature and design matrix for model building.
compas_rating = bankruptcy.Bankrupt.values
propublica_data = bankruptcy.drop("Bankrupt", 1)


# this is just for demonstration, any classifier or regressor
# can be used here. fairml only requires a predict function
# to diagnose a black-box model.

# we fit a quick and dirty logistic regression sklearn
# model here.
clf = LogisticRegression(penalty='l2', C=0.01)
clf.fit(propublica_data.values, compas_rating)
clf.fit(propublica_data.values, compas_rating)

# +
#  call audit model with model
total, _ = audit_model(clf.predict, propublica_data)

# print feature importance
print(total)

plt.figure(figsize=(20,10))

# generate feature dependence plot
fig = plot_dependencies(
    total.median(),
    reverse_values=False,
    title="FairML feature dependence"
)
#plt.savefig("fairml_ldp.eps", transparent=False, bbox_inches='tight')
plt
# -

fig.set_size_inches(15,25)
fig


