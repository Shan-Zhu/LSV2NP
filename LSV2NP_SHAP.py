
#coding: utf-8

'''
SHAP
'''

import csv
import math
import pandas as pd
import numpy as np
from numpy import *
import sklearn as sk
from sklearn import preprocessing, svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 16
colors = ['#488CE5','#FB6F6F']  # 开始颜色和结束颜色
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

# Read data
input_data = pd.read_csv("LSV2NP_database.csv", sep=',')

# Determine feature and target variables
features = input_data.drop('Particle_size (nm)', axis=1).drop('DOI', axis=1).drop('Composition', axis=1)
target = input_data['Particle_size (nm)']

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

exported_pipeline = GradientBoostingRegressor()

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
results_train = exported_pipeline.predict(X_train)

shap.initjs()
explainer = shap.Explainer(exported_pipeline)

y_base = explainer.expected_value

predictt = exported_pipeline.predict(X_train)

shap_values = explainer.shap_values(features)

shap_exp=shap.Explanation(shap_values)

shap.plots.bar(shap_exp)

plt.show()






